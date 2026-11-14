"""Layer 6: fit a DrumPreset to a reference recording by least squares.

    y, sr = load_mono(path)
    res = fit_drum(y, sr, template_preset)      # f0, K, coupling, d0, d1
    res = fit_wires(y, sr, res)                 # snare only: threshold, brightness, level
    print(report(res))

The template preset fixes everything that is NOT fitted: strike/pickup radii, M, N,
mallet, reso/batter f0 ratio, air-column length, rim click. Fit only to the reference.
"""
from dataclasses import dataclass, replace
from itertools import product

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import butter, find_peaks, sosfilt, sosfiltfilt, stft
from scipy.special import jn_zeros

from drums import DrumPreset, band_time_energy, drum_tables, f0_for_pitch, render_drum


@dataclass
class FitResult:
    preset: DrumPreset        # template with fitted values written in
    pitch: float              # Hz, fitted sounding (0,1) frequency of the batter head
    meas_f: np.ndarray        # Hz, measured peaks used for the fit
    pred_f: np.ndarray        # Hz, nearest predicted partial for each measured peak
    cents: np.ndarray         # 1200*log2(pred/meas)
    meas_decay: np.ndarray    # 1/s, measured decay rate per peak (nan = not measurable)
    screen: "Screen" = None   # quality check of the reference file
    wire_err_db: float = float("nan")   # rms band error after fit_wires


@dataclass
class Screen:
    """Quality check of a reference hit. The thresholds are screening heuristics,
    not physics; they only decide which fitted numbers to trust."""
    clipped: bool         # flat-topped samples at full scale
    n_hits: int           # onsets found (should be 1)
    snr_db: float         # peak frame level over noise floor
    tail_db: float        # level of the last 50 ms relative to the peak frame
    decay_ratio: float    # median late/early decay slope of the strongest partials;
                          # 1 = exponential, <1 = tail too slow (reverb), >1 = start too slow (compression)
    trust_freq: bool      # pitch, K, coupling
    trust_decay: bool     # d0, d1 and wire parameters
    notes: list


# ------------------------------------------------------------------ measurement

def load_mono(path: str):
    import soundfile as sf
    y, sr = sf.read(path, always_2d=True)
    return y.mean(axis=1), sr


def extract_peaks(y, sr, fmin=30.0, fmax=None, prominence_db=10.0, floor_db=-60.0, pad=4):
    """Spectral peaks of the whole hit. Returns (freqs_hz, level_db) sorted by freq.
    Zero-padded FFT, no window (a decayed hit has no truncation leakage),
    parabolic interpolation of the log magnitude."""
    n = int(2 ** np.ceil(np.log2(len(y) * pad)))
    mag = np.abs(np.fft.rfft(y, n))
    db = 20 * np.log10(mag / mag.max() + 1e-15)
    idx = find_peaks(db, prominence=prominence_db, height=floor_db)[0]
    idx = idx[(idx > 0) & (idx < len(db) - 1)]
    a, b, c = db[idx - 1], db[idx], db[idx + 1]
    off = 0.5 * (a - c) / (a - 2 * b + c)
    f = (idx + off) * sr / n
    keep = (f >= fmin) & (f <= (fmax or sr / 2))
    return f[keep], (b - 0.25 * (a - c) * off)[keep]


def _partial_decays(y, sr, freqs, nperseg, span_db):
    """For each partial: (t, log envelope) of its STFT bin from its maximum down to
    span_db below it. Shared by measure_decays and decay_ratio."""
    hop = nperseg // 8
    _, t, Z = stft(y, sr, nperseg=nperseg, noverlap=nperseg - hop, boundary=None, padded=False)
    for f in freqs:
        env = np.abs(Z[int(round(f * nperseg / sr))])
        i0 = int(np.argmax(env))
        below = np.nonzero(env[i0:] < env[i0] * 10 ** (-span_db / 20))[0]
        i1 = i0 + (below[0] if len(below) else len(env) - i0)
        yield t[i0:i1], np.log(env[i0:i1] + 1e-300)


def measure_decays(y, sr, freqs, nperseg=4096, span_db=30.0):
    """Decay rate (1/s) of each partial: slope of log |STFT bin| after its maximum,
    over the first span_db of decay. nan where fewer than 4 frames are usable."""
    return np.array([-np.polyfit(t, e, 1)[0] if len(t) >= 4 else np.nan
                     for t, e in _partial_decays(y, sr, freqs, nperseg, span_db)])


def decay_ratio(y, sr, freqs, nperseg=4096, span_db=30.0):
    """Median over partials of (late decay slope / early decay slope), splitting each
    partial's first span_db of decay at its halfway level. 1 = exponential decay."""
    ratios = []
    for t, e in _partial_decays(y, sr, freqs, nperseg, span_db):
        if len(t) < 6:
            continue
        early = e > e[0] - 0.5 * span_db * np.log(10) / 20
        k = int(np.argmin(early)) if not early.all() else len(t)
        if k >= 3 and len(t) - k >= 3:
            a, b = np.polyfit(t[:k], e[:k], 1)[0], np.polyfit(t[k:], e[k:], 1)[0]
            if a < 0:
                ratios.append(b / a)
    return float(np.median(ratios)) if ratios else float("nan")


FRAME_S, LEAD_FRAMES = 0.005, 10   # 5 ms hop; 50 ms of silence prepended to every file


def frame_db(y, sr):
    """Frame levels in dB re the loudest frame (10 ms rms, 5 ms hop). 50 ms of silence
    is prepended so a file that starts on the attack still shows a jump:
    frame i starts at sample (i - LEAD_FRAMES) * hop."""
    hop = int(FRAME_S * sr)
    pad = np.concatenate([np.zeros(LEAD_FRAMES * hop), y])
    n = (len(pad) - 2 * hop) // hop
    c = np.concatenate([[0.0], np.cumsum(pad ** 2)])
    i = np.arange(n) * hop
    rms = np.sqrt(np.maximum(c[i + 2 * hop] - c[i], 0) / (2 * hop))
    return 20 * np.log10(rms / rms.max() + 1e-10)


def onset_frames(y, sr):
    """Frame index of each hit. An onset is a 10 dB level jump within 30 ms, to within
    30 dB of the loudest frame; jumps closer than 50 ms are one hit.
    Counted above 1 kHz: an attack is broadband, while the slow beating of two close low
    partials (coupled heads) also makes the level jump but only down low."""
    def jumps(d):
        low = np.array([d[max(0, i - 6):i].min() if i else d[0] for i in range(len(d))])
        return np.nonzero((d - low > 10) & (d > -30))[0]
    on = jumps(frame_db(sosfilt(butter(4, 1000.0 / (sr / 2), "high", output="sos"), y), sr))
    if len(on) == 0:
        on = jumps(frame_db(y, sr))
    return on[np.concatenate([[True], np.diff(on) > 10])] if len(on) else on


def slice_hits(y, sr, min_s=0.3, max_s=4.0):
    """Cut a long recording into candidate single hits. Each slice runs from 5 ms before
    an onset to 20 ms before the next one (at most max_s); slices shorter than min_s are
    dropped. Returns [(start_sample, end_sample)]. Screen each slice before fitting."""
    hop = int(FRAME_S * sr)
    starts = np.maximum((onset_frames(y, sr) - LEAD_FRAMES - 1) * hop, 0)
    ends = np.minimum(np.concatenate([starts[1:] - 3 * hop, [len(y)]]), starts + int(max_s * sr))
    return [(int(a), int(b)) for a, b in zip(starts, ends) if b - a >= min_s * sr]


def screen(y, sr, nperseg=4096, min_snr_db=40.0, max_tail_db=-40.0, ratio_range=(0.6, 1.6)) -> Screen:
    """Check a reference hit before trusting a fit to it."""
    peak = np.max(np.abs(y))
    # clipping: runs of samples stuck at full scale
    flat = np.abs(y) >= peak * (1 - 1e-4)
    edges = np.diff(np.concatenate([[0], flat.astype(int), [0]]))
    runs = np.nonzero(edges == -1)[0] - np.nonzero(edges == 1)[0]
    clipped = bool(np.sum(runs >= 3) >= 2 or np.any(runs >= 8))

    db = frame_db(y, sr)
    on = onset_frames(y, sr)
    n_hits = len(on)

    first = on[0] if len(on) else 0
    tail_db = float(db[-10:].mean())
    # noise floor: before the attack if the file has a lead-in, else the tail
    lead = db[10:first - 2]
    noise_db = float(lead.mean()) if len(lead) >= 4 else tail_db
    snr_db = -noise_db

    f, level = extract_peaks(y, sr)
    ratio = decay_ratio(y, sr, f[np.argsort(-level)[:6]], nperseg=nperseg) if len(f) else float("nan")

    notes = []
    if clipped:
        notes.append("clipped: adds false harmonics")
    if n_hits != 1:
        notes.append("%d hits found, need exactly 1" % n_hits)
    if snr_db < min_snr_db:
        notes.append("noise floor only %.0f dB below peak (want %.0f)" % (snr_db, min_snr_db))
    if tail_db > max_tail_db:
        notes.append("tail still at %.0f dB at end of file: cut off, noisy or reverberant" % tail_db)
    if not ratio_range[0] <= ratio <= ratio_range[1]:
        notes.append("decay not exponential (late/early slope %.2f): %s" % (
            ratio, "reverb or room" if ratio < 1 else "compression" if ratio > 1 else "not measurable"))
    trust_freq = not clipped and snr_db >= 30.0
    trust_decay = not notes
    return Screen(clipped, n_hits, snr_db, tail_db, ratio, trust_freq, trust_decay, notes)


# ------------------------------------------------------------------ frequency fit

def with_tuning(d: DrumPreset, pitch: float, K: float, coupling: float) -> DrumPreset:
    """Template with new sounding pitch, air loading and coupling.
    The resonant head keeps the template's reso/batter f0 ratio and gets the same K."""
    f0 = f0_for_pitch(pitch, K, d.pitch_mode, d.batter.f01_scale)
    reso = None if d.reso is None else replace(d.reso, f0=f0 * d.reso.f0 / d.batter.f0, K=K)
    return replace(d, batter=replace(d.batter, f0=f0, K=K), reso=reso, coupling=coupling)


def audible_freqs(d: DrumPreset, rel=0.01) -> np.ndarray:
    """Predicted partial frequencies with |amp| above rel * strongest, sorted."""
    t = drum_tables(d)[0]
    return t.freq[np.abs(t.amp) > rel * np.max(np.abs(t.amp))]


def _cents(a, b):
    return 1200 * np.log2(a / b)


def _nearest(pool, x):
    return pool[np.argmin(np.abs(pool[None, :] - np.asarray(x)[:, None]), axis=1)]


def freq_residuals(theta, d, meas_f, all_f, n_low=4, clip=200.0):
    """Cents errors, clipped so one unmatched peak cannot dominate:
    - each measured peak -> nearest audible predicted partial
    - each of the n_low lowest audible predicted partials -> nearest measured peak
      (stops the fit from hiding in a dense forest of high modes)"""
    pred = audible_freqs(with_tuning(d, *theta))
    r1 = _cents(_nearest(pred, meas_f), meas_f)
    low = pred[:n_low]
    r2 = _cents(low, _nearest(all_f, low))
    return np.clip(np.concatenate([r1, r2]), -clip, clip)


def ladder_freqs(f11, K, n_rungs=5):
    """Air-loaded (m,1) frequencies for m = 0..n_rungs, anchored so (1,1) = f11."""
    a = np.array([jn_zeros(m, 1)[0] for m in range(n_rungs + 1)])
    g = a / np.sqrt(1 + K / a)
    return f11 * g / g[1]


def fit_ladder(all_f, all_db, n_rungs=5, anchor_db=10.0, good_cents=20.0, clip=100.0):
    """Fit K to the (m,1) ladder, m = 2..n_rungs, with (1,1) pinned to an anchor peak:
    the evenly stepping row of peaks an edge-struck single head shows.
    Returns (f11, K, rms_cents).

    (1,1) is not fitted: its peak is measured directly, and letting it float lets weak
    upper rungs (often buried in noise) pull the pitch.
    Anchor: the strongest peak is taken as (1,1), unless a lower strong peak (within
    anchor_db of it) has a ladder that fits within good_cents rms AND has the strongest
    peak as one of its rungs; then the lowest such peak is (1,1) and the strongest peak
    is a higher rung. A ladder started on an unrelated peak can fit by chance, but it
    will not pass through the strongest peak.
    Robust loss: one rung matched to a neighbouring mode or a noise peak must not drag K.
    All rungs count equally: weighting by peak level was tried and made real hits worse
    (real upper rungs are weak but genuine). The price: when the upper rungs are buried in
    noise, K can be off by about 20% (seen on the model's own 587 Hz render with noise).
    Known ambiguity: for K near 0 the (m,2) modes also form a ladder that looks like a
    high-K (m,1) ladder; the fit cannot tell them apart from frequencies alone."""
    def fit_anchor(f11):
        def resid(k):
            pred = ladder_freqs(f11, k[0], n_rungs)[2:]
            return np.clip(_cents(pred, _nearest(all_f, pred)), -clip, clip)
        r = min((least_squares(resid, [k0], bounds=([0.0], [12.0]), diff_step=1e-3,
                               loss="soft_l1", f_scale=8.0)
                 for k0 in (0.5, 2.0, 4.0, 7.0)), key=lambda r: r.cost)
        return f11, float(r.x[0]), float(np.sqrt(np.mean(resid(r.x) ** 2)))

    top = all_f[np.argmax(all_db)]
    for f11 in all_f[(all_db >= all_db.max() - anchor_db) & (all_f < top)]:
        fit = fit_anchor(f11)
        rungs = ladder_freqs(fit[0], fit[1], n_rungs)[2:]
        if fit[2] < good_cents and np.min(np.abs(_cents(rungs, top))) < 30.0:
            return fit
    return fit_anchor(top)


def find_f01(all_f, all_db, f11, K, window=(0.3, 0.65)):
    """(0,1) frequency as a multiple of where the K curve puts it: the strongest peak
    between window[0]*f11 and window[1]*f11. Returns 1.0 if there is no peak there.
    Low confidence on a multi-drum set: neighbouring drums ring in the same range."""
    sel = (all_f > window[0] * f11) & (all_f < window[1] * f11)
    if not sel.any():
        return 1.0
    return float(all_f[sel][np.argmax(all_db[sel])] / ladder_freqs(f11, K)[0])


def fit_drum(y, sr, template: DrumPreset, n_peaks=10, fmin=30.0, fmax=None, nperseg=4096,
             floor_db=-60.0) -> FitResult:
    """Fit pitch (-> f0), K, coupling to peak frequencies, then d0, d1 to peak decays.
    Raise fmin above mains hum / rumble and floor_db above the noise floor for real takes."""
    all_f, all_db = extract_peaks(y, sr, fmin=fmin, fmax=fmax, floor_db=floor_db)
    strongest = np.sort(np.argsort(-all_db)[:n_peaks])
    meas_f = all_f[strongest]
    two = template.reso is not None

    if template.pitch_mode == (1, 1):
        # edge-struck single head: lock onto the (m,1) ladder, then place (0,1) separately
        pitch, K, _ = fit_ladder(all_f, all_db)
        scale = find_f01(all_f, all_db, pitch, K)
        fitted = with_tuning(replace(template, batter=replace(template.batter, f01_scale=scale)),
                             pitch, K, 0.0)
        return _finish(y, sr, fitted, pitch, meas_f, nperseg)

    # coarse grid (the cost is not smooth: nearest-partial matching), then refine
    # pitch candidates: the lowest peaks, plus the strongest-set's lowest peak read as
    # each of the first ideal ratios (an edge strike can leave the (0,1) mode weak)
    ideal = np.array([1.0, 1.594, 2.136, 2.296, 2.653, 2.918])
    pitches = np.unique(np.concatenate([all_f[:3], meas_f[0] / ideal]))
    grid = product(pitches, [0.0, 0.5, 1.0, 2.0, 3.0, 4.0],
                   [0.0, 0.1, 0.2, 0.4, 0.8] if two else [0.0])
    cost = lambda th: np.sum(freq_residuals(th, template, meas_f, all_f) ** 2)
    starts = sorted(grid, key=cost)[:3]
    lo, hi = [pitches.min() * 0.7, 0.0, 0.0], [pitches.max() * 1.3, 10.0, 5.0 if two else 1e-9]
    best = min((least_squares(freq_residuals, np.clip(s, lo, hi), bounds=(lo, hi),
                              args=(template, meas_f, all_f), diff_step=1e-3)
                for s in starts), key=lambda r: r.cost)
    pitch, K, coupling = best.x
    fitted = with_tuning(template, pitch, K, coupling if two else 0.0)

    return _finish(y, sr, fitted, pitch, meas_f, nperseg)


def _finish(y, sr, fitted, pitch, meas_f, nperseg) -> FitResult:
    """Decay fit (linear least squares d = d0 + d1*f over the measurable peaks),
    predicted-vs-measured table and reference screening."""
    two = fitted.reso is not None
    dec = measure_decays(y, sr, meas_f, nperseg=nperseg)
    ok = np.isfinite(dec)
    d1, d0 = np.polyfit(meas_f[ok], dec[ok], 1) if ok.sum() >= 2 else (0.0, np.nanmean(dec))
    fitted = replace(fitted, batter=replace(fitted.batter, d0=d0, d1=d1),
                     reso=None if not two else replace(fitted.reso, d0=d0, d1=d1))
    pred_f = _nearest(audible_freqs(fitted), meas_f)
    return FitResult(fitted, pitch, meas_f, pred_f, _cents(pred_f, meas_f), dec,
                     screen(y, sr, nperseg=nperseg))


# ------------------------------------------------------------------ tenor ladder profile

def measure_rungs(y, sr, f11, K, f01_scale, n_rungs=8, nperseg=2048, match_cents=40.0, **peak_kw):
    """Initial amplitude (dB) and decay (1/s) of the ladder rungs (m,1), m = 0..n_rungs-1.
    A rung with no peak within match_cents is nan. The FFT peak of a decaying sine cut
    off after T seconds is a*(1 - exp(-d*T))/(2*d), so the amplitude is the peak level
    times d/(1 - exp(-d*T)): slow-decaying rungs look louder in a spectrum than they start."""
    f, level = extract_peaks(y, sr, **peak_kw)
    rung = ladder_freqs(f11, K, n_rungs - 1)
    rung[0] *= f01_scale
    near = _nearest(f, rung)
    ok = np.abs(_cents(near, rung)) < match_cents
    decay = np.where(ok, measure_decays(y, sr, near, nperseg=nperseg, span_db=25.0), np.nan)
    T = len(y) / sr
    with np.errstate(invalid="ignore", divide="ignore"):
        amp_db = level[np.searchsorted(f, near)] + 20 * np.log10(decay / (1 - np.exp(-decay * T)))
    return np.where(ok & (decay > 0), amp_db, np.nan), decay


def measure_pop(y, sr, f11, body_decay=15.0, window_s=0.012, lowpass_hz=2500.0):
    """Attack pop: least-squares fit of two damped sines to the first window_s of a hit,
    a short burst (free amplitude, decay, frequency, phase) plus the ringing pitch mode
    (frequency f11, decay body_decay, free amplitude and phase).
    Returns (pop_amp / body_amp, pop_decay 1/s, pop_freq / f11, relative_fit_error).
    Rough by nature: check relative_fit_error before trusting the numbers."""
    x = sosfiltfilt(butter(4, lowpass_hz / (sr / 2), "low", output="sos"), y)
    on = max(int(np.argmax(np.abs(x) > 0.2 * np.max(np.abs(x)))) - int(0.0005 * sr), 0)
    seg = x[on:on + int(window_s * sr)]
    seg = seg / np.max(np.abs(seg))
    t = np.arange(len(seg)) / sr

    def resid(p):
        a1, d1, ratio, ph1, a2, ph2, t0 = p
        tt = np.maximum(t - t0, 0)
        model = (a1 * np.exp(-d1 * tt) * np.sin(2 * np.pi * ratio * f11 * tt + ph1)
                 + a2 * np.exp(-body_decay * tt) * np.sin(2 * np.pi * f11 * tt + ph2)) * (t >= t0)
        return model - seg

    lo, hi = [0, 30, 0.4, -7, 1e-3, -7, 0.0], [5, 3000, 2.5, 7, 2, 7, 0.002]
    best = min((least_squares(resid, [1.0, 400.0, r0, ph, 0.4, 0.0, 0.0005], bounds=(lo, hi))
                for r0 in (0.7, 0.85, 1.0, 1.2) for ph in (0.0, 1.5, 3.0, 4.5)), key=lambda r: r.cost)
    a1, d1, ratio, _, a2, _, _ = best.x
    return float(a1 / a2), float(d1), float(ratio), float(np.sqrt(np.mean(best.fun ** 2) / np.mean(seg ** 2)))


def extract_attack(y, sr, rung_decay, n_rungs=8, lead_s=0.002, fit_from_s=0.015,
                   fade=(0.020, 0.040), fit_len_s=0.3, highpass_hz=150.0, **peak_kw):
    """Slice the attack out of a single tenor hit for the hybrid model.
    1. high-pass (hum, rumble), align so the hit starts lead_s into the slice, peak = 1
    2. ladder fit -> the rung frequencies (the measured peak where a rung has one)
    3. linear least squares for amplitude and phase of every rung on
       fit_from_s .. fit_len_s, i.e. after the attack is over
    4. run those rungs back to the start and subtract: the residual, faded out over
       `fade`, is the attack
    The attack is returned in units where the fitted (1,1) rung has amplitude 1, so it can
    be scaled onto any ring. rung_decay: fallback decay per rung where none is measurable.
    Returns dict(attack, lead_s, fade, rung_phase, start, peak_amp, f11, K, f01_scale,
    explained, attack_db): start = sample index in y where the slice begins (rung_phase is
    relative to start + lead_s), peak_amp = fitted (1,1) amplitude in the normalised slice,
    explained = share of the 15-300 ms energy the rungs account for, attack_db = attack
    energy re the whole slice."""
    x = sosfiltfilt(butter(4, highpass_hz / (sr / 2), "high", output="sos"), y)
    on = max(int(np.argmax(np.abs(x) > 0.2 * np.max(np.abs(x)))) - int(lead_s * sr), 0)
    x = x[on:on + int(fit_len_s * sr)]
    x = x / np.max(np.abs(x))
    t = np.arange(len(x)) / sr

    f, level = extract_peaks(x, sr, **peak_kw)
    f11, K, _ = fit_ladder(f, level)
    scale = find_f01(f, level, f11, K)
    rung = ladder_freqs(f11, K, n_rungs - 1)
    rung[0] *= scale
    near = _nearest(f, rung)
    freq = np.where(np.abs(_cents(near, rung)) < 40.0, near, rung)
    dec = measure_decays(x, sr, freq, nperseg=2048, span_db=25.0)
    dec = np.where(np.isfinite(dec) & (dec > 1.0), dec, np.asarray(rung_decay, dtype=float)[:n_rungs])

    def basis(tt):
        env = np.exp(-dec[:, None] * tt)
        return np.concatenate([env * np.cos(2 * np.pi * freq[:, None] * tt),
                               env * np.sin(2 * np.pi * freq[:, None] * tt)]).T

    late = t >= fit_from_s
    coef, *_ = np.linalg.lstsq(basis(t[late] - lead_s), x[late], rcond=None)
    ring = basis(np.maximum(t - lead_s, 0)) @ coef * (t >= lead_s)
    resid = x - ring
    keep = 0.5 - 0.5 * np.cos(np.pi * np.clip((fade[1] - t) / (fade[1] - fade[0]), 0, 1))
    attack = (resid * keep)[:int(fade[1] * sr)]
    amp = np.hypot(coef[:n_rungs], coef[n_rungs:])          # a*sin(w*t + phase)
    return dict(attack=attack / amp[1], lead_s=lead_s, fade=tuple(fade), start=on, peak_amp=float(amp[1]),
                rung_phase=tuple(float(v) for v in np.arctan2(coef[:n_rungs], coef[n_rungs:])),
                f11=float(f11), K=float(K), f01_scale=float(scale),
                explained=float(1 - np.sum(resid[late] ** 2) / np.sum(x[late] ** 2)),
                attack_db=float(10 * np.log10(np.sum(attack ** 2) / np.sum(x ** 2))))


def attack_tones(a: dict, sr, n_tones=60, until_s=0.020, max_decay=3000.0, min_decay=np.log(10) / 0.040):
    """Model a sliced attack (extract_attack result) as short damped tones, so that no audio
    has to be stored. ESPRIT on the part of the attack from the ring start (lead_s) to
    until_s (before its fade-out begins): Hankel matrix -> signal subspace -> poles, then
    linear least squares for amplitude and phase. Unstable, non-oscillating and very fast
    poles are dropped. With n_tones = 60 about 35-50 survive; listening test by the user:
    60 is indistinguishable from the sample, 24 sounds "compressed".
    Returns (rows, explained): rows of (freq / f11, amplitude re the (1,1) rung, decay 1/s,
    start phase rad) for a*exp(-d*t)*sin(2*pi*f*t + phase), t from the ring start;
    explained = share of the analysed attack energy the tones rebuild."""
    i0, i1 = int(round(a["lead_s"] * sr)), int(until_s * sr)
    x = np.asarray(a["attack"], dtype=float)[i0:i1]
    L = len(x) // 2
    H = np.lib.stride_tricks.sliding_window_view(x, L).T
    U = np.linalg.svd(H, full_matrices=False)[0][:, :2 * n_tones]
    z = np.linalg.eigvals(np.linalg.lstsq(U[:-1], U[1:], rcond=None)[0])
    freq, decay = np.angle(z) * sr / (2 * np.pi), -np.log(np.abs(z)) * sr
    ok = (freq > 20.0) & (freq < 0.45 * sr) & (decay > 0) & (decay < max_decay)
    # An attack tone must be over when the attack is: at least 20 dB down 40 ms in (the
    # sampled attack is faded out by then). 18 ms of data cannot tell a slow decay from a
    # very slow one, and a tone fitted with almost none would ring on for seconds.
    freq, decay = freq[ok], np.maximum(decay[ok], min_decay)
    t = np.arange(len(x)) / sr
    env = np.exp(-decay[None, :] * t[:, None])
    B = np.hstack([env * np.sin(2 * np.pi * freq[None, :] * t[:, None]),
                   env * np.cos(2 * np.pi * freq[None, :] * t[:, None])])
    c = np.linalg.lstsq(B, x, rcond=None)[0]
    k = len(freq)
    amp, phase = np.hypot(c[:k], c[k:]), np.arctan2(c[k:], c[:k])
    explained = float(1 - np.sum((B @ c - x) ** 2) / np.sum(x ** 2))
    order = np.argsort(freq)
    rows = tuple((float(freq[i] / a["f11"]), float(amp[i]), float(decay[i]), float(phase[i])) for i in order)
    return rows, explained


def align_to_attack(y, sr, lead_s=0.001):
    """Trim so the hit starts lead_s before the first sample above 20% of the peak."""
    on = int(np.argmax(np.abs(y) > 0.2 * np.max(np.abs(y))))
    return y[max(on - int(lead_s * sr), 0):]


def measure_noise(y, sr, modal, bands, edges, floor_db=-80.0):
    """Noise map of a hit: what it has in each band and time slice beyond the modal part.
    `modal` is the rendered ladder model of the same drum (no noise). Both are referred to
    their own energy below bands[0][0] inside edges[-1] seconds, so the result is in dB
    re the modal energy. Returns [band][slice] dB, floored at floor_db."""
    T = int(edges[-1] * sr)
    y = align_to_attack(y, sr)[:T]
    low = butter(4, bands[0][0] / (sr / 2), "low", output="sos")
    real = band_time_energy(y, sr, bands, edges) / np.sum(sosfilt(low, y) ** 2) * sr
    model = band_time_energy(modal[:T], sr, bands, edges) / np.sum(sosfilt(low, modal[:T]) ** 2) * sr
    return np.maximum(10 * np.log10(np.maximum(real - model, 1e-30)), floor_db)


def tenor_profile(hits, sr, n_rungs=8, bands=((1200.0, 2400.0), (2400.0, 4800.0), (4800.0, 9600.0)),
                  edges=(0.0, 0.003, 0.01, 0.03, 0.08, 0.2, 0.3), **peak_kw):
    """Pool several single tenor hits (any drums of one set) into one profile.
    Pass 1: each hit is fitted on its own (ladder, (0,1)); rung amplitudes (re that hit's
    strongest rung) and rung decays are combined by median.
    Pass 2: each hit is compared with the ladder-only model of its own drum to get the
    noise map; maps are combined by median (in dB).
    Returns dict(rung_amp_db, rung_decay, noise_bands, noise_edges, noise_db,
                 fits=[(f11, K, f01_scale)])."""
    import drums
    amps, decays, fits = [], [], []
    for y in hits:
        f, level = extract_peaks(y, sr, **peak_kw)
        f11, K, _ = fit_ladder(f, level)
        scale = find_f01(f, level, f11, K)
        a, d = measure_rungs(y, sr, f11, K, scale, n_rungs, **peak_kw)
        amps.append(a - np.nanmax(a))
        decays.append(d)
        fits.append((f11, K, scale))
    amp = np.nanmedian(amps, axis=0)
    rung_amp = tuple(float(v) for v in np.round(amp - np.nanmax(amp), 1))
    rung_decay = tuple(float(v) for v in np.round(np.nanmedian(decays, axis=0), 1))

    maps = []
    for y, (f11, K, scale) in zip(hits, fits):
        d = drums.tenor(f11, 0.0, K=K)
        d = replace(d, batter=replace(d.batter, f01_scale=scale, sr=sr, duration=edges[-1] + 0.05),
                    rung_amp_db=rung_amp, rung_decay=rung_decay)
        maps.append(measure_noise(y, sr, render_drum(with_tuning(d, f11, K, 0.0)), bands, edges))
    noise_db = tuple(tuple(float(v) for v in row) for row in np.round(np.median(maps, axis=0), 1))
    return dict(rung_amp_db=rung_amp, rung_decay=rung_decay, noise_bands=tuple(bands),
                noise_edges=tuple(edges), noise_db=noise_db, fits=fits)


# ------------------------------------------------------------------ wire fit (snare)

def band_map(y, sr, n_bands=6, n_frames=40, fmin=200.0, floor_db=-60.0):
    """Log-band x time energy map in dB relative to its maximum.
    Wide bands average over the wire noise; many time frames are needed because
    wire_threshold shows up as WHEN the noise cuts off (level alone cannot tell
    threshold from wire_level)."""
    _, _, Z = stft(y, sr, nperseg=1024, boundary=None, padded=False)
    P = np.abs(Z) ** 2
    f = np.fft.rfftfreq(1024, 1 / sr)
    f_edges = np.geomspace(fmin, sr / 2, n_bands + 1)
    t_edges = np.unique(np.geomspace(1, P.shape[1], n_frames + 1).astype(int))
    t_edges[0] = 0
    E = np.array([[P[(f >= f_edges[i]) & (f < f_edges[i + 1]), t_edges[j]:t_edges[j + 1]].mean()
                   for j in range(len(t_edges) - 1)] for i in range(n_bands)])
    return np.maximum(10 * np.log10(E / E.max() + 1e-15), floor_db)


def fit_wires(y, sr, res: FitResult) -> FitResult:
    """Fit wire_threshold, wire_brightness, wire_level to the band map of the hit.
    Run after fit_drum. The reference is trimmed/padded to the model's length."""
    n = int(res.preset.batter.sr * res.preset.batter.duration)
    target = band_map(np.pad(y, (0, max(0, n - len(y))))[:n], sr)

    def model(th):
        return replace(res.preset, wires=True, wire_threshold=th[0],
                       wire_brightness=th[1], wire_level=th[2])

    def resid(th):
        return (band_map(render_drum(model(th)), sr) - target).ravel()

    grid = product([0.02, 0.1, 0.3], [2000.0, 5000.0, 10000.0], [0.1, 0.5, 2.0])
    start = min(grid, key=lambda th: np.sum(resid(th) ** 2))
    lo, hi = [0.0, 500.0, 0.0], [0.95, 0.45 * sr, 20.0]
    best = least_squares(resid, start, bounds=(lo, hi), diff_step=0.05, x_scale=[0.1, 5000.0, 1.0])
    return replace(res, preset=model(best.x), wire_err_db=float(np.sqrt(np.mean(best.fun ** 2))))


# ------------------------------------------------------------------ report

def report(res: FitResult) -> str:
    d = res.preset
    lines = ["%s: pitch=%.2f Hz on mode %s  f0=%.2f Hz  K=%.3f  coupling=%.3f  d0=%.2f  d1=%.5f"
             % (d.name, res.pitch, d.pitch_mode, d.batter.f0, d.batter.K, d.coupling,
                d.batter.d0, d.batter.d1)]
    if d.batter.f01_scale != 1.0:
        lines.append("(0,1) mode at %.2f x its K-curve frequency (low confidence)" % d.batter.f01_scale)
    if d.wires:
        lines.append("wires: threshold=%.3f  brightness=%.0f Hz  level=%.3f  band rms err=%.2f dB"
                     % (d.wire_threshold, d.wire_brightness, d.wire_level, res.wire_err_db))
    lines.append("%10s %10s %8s %10s %10s" % ("meas Hz", "pred Hz", "cents", "decay 1/s", "model 1/s"))
    for fm, fp, c, dm in zip(res.meas_f, res.pred_f, res.cents, res.meas_decay):
        lines.append("%10.2f %10.2f %8.1f %10.2f %10.2f" % (fm, fp, c, dm, d.batter.d0 + d.batter.d1 * fm))
    lines.append("rms freq error: %.1f cents" % np.sqrt(np.mean(res.cents ** 2)))
    if res.screen is not None:
        sc = res.screen
        lines.append("reference check: hits=%d  snr=%.0f dB  tail=%.0f dB  decay ratio=%.2f  clipped=%s"
                     % (sc.n_hits, sc.snr_db, sc.tail_db, sc.decay_ratio, sc.clipped))
        lines.append("TRUST  pitch/K/coupling: %s   decay d0/d1 and wires: %s"
                     % ("yes" if sc.trust_freq else "NO", "yes" if sc.trust_decay else "NO"))
        lines.extend("  - " + n for n in sc.notes)
    return "\n".join(lines)
