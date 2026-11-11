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
from scipy.signal import butter, find_peaks, sosfilt, stft

from drums import DrumPreset, drum_tables, f0_for_pitch, render_drum


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
    f0 = f0_for_pitch(pitch, K)
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


def fit_drum(y, sr, template: DrumPreset, n_peaks=10, fmax=None, nperseg=4096) -> FitResult:
    """Fit pitch (-> f0), K, coupling to peak frequencies, then d0, d1 to peak decays."""
    all_f, all_db = extract_peaks(y, sr, fmax=fmax)
    strongest = np.sort(np.argsort(-all_db)[:n_peaks])
    meas_f = all_f[strongest]
    two = template.reso is not None

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

    # decay: linear least squares d = d0 + d1*f over the measurable peaks
    dec = measure_decays(y, sr, meas_f, nperseg=nperseg)
    ok = np.isfinite(dec)
    d1, d0 = np.polyfit(meas_f[ok], dec[ok], 1) if ok.sum() >= 2 else (0.0, np.nanmean(dec))
    fitted = replace(fitted, batter=replace(fitted.batter, d0=d0, d1=d1),
                     reso=None if not two else replace(fitted.reso, d0=d0, d1=d1))

    pred_f = _nearest(audible_freqs(fitted), meas_f)
    return FitResult(fitted, pitch, meas_f, pred_f, _cents(pred_f, meas_f), dec,
                     screen(y, sr, nperseg=nperseg))


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
    lines = ["%s: pitch=%.2f Hz  f0=%.2f Hz  K=%.3f  coupling=%.3f  d0=%.2f  d1=%.5f"
             % (d.name, res.pitch, d.batter.f0, d.batter.K, d.coupling, d.batter.d0, d.batter.d1)]
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
