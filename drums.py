"""Drum types built on the drum_modal core (Layer 5).

DrumPreset -> drum_tables() -> render_drum()
Preset builders: snare(), tenor(), bass(). Every number in the builders is a
PLACEHOLDER until Layer 6 fits it to reference recordings; nothing is tuned by ear.

Non-membrane partials (air column, rim click) are stored in the same ModeTable
with m = -1 (column) or m = -2 (rim), so the whole drum is still one mode table.
"""
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
from scipy.signal import butter, lfilter, sosfilt
from scipy.special import jn_zeros

from drum_modal import DrumParams, ModeTable, mode_table, render, two_head_tables

COLUMN, RIM = -1, -2   # values of ModeTable.m marking non-membrane partials


@dataclass
class DrumPreset:
    name: str
    batter: DrumParams
    reso: Optional[DrumParams] = None  # None = single head (tenor)
    pitch_mode: tuple = (0, 1)         # (m, n) of the batter mode that carries the tuned pitch
    coupling: float = 0.0              # two-head air coupling (Layer 4)
    # open-tube air column (tenors): f = n*c/(2*L_eff)
    column_L: float = 0.0              # m, effective tube length; 0 = off
    column_n: int = 3                  # number of column partials
    column_level: float = 0.0          # amp relative to strongest membrane mode
    column_decay: float = 20.0         # 1/s
    c: float = 343.0                   # m/s, speed of sound
    # rimshot click: one damped partial
    rim_level: float = 0.0             # amp relative to strongest membrane mode; 0 = off
    rim_hz: float = 2000.0
    rim_decay: float = 300.0           # 1/s
    # snare wires: filtered noise gated by resonant-head displacement
    wires: bool = False
    wire_threshold: float = 0.1        # wire tension: fraction of peak reso displacement
    wire_brightness: float = 6000.0    # Hz, low-pass cutoff of the wire noise
    wire_level: float = 0.5            # relative to peak of the modal signal
    wire_seed: int = 0
    # measured ladder profile (tenors). When rung_amp_db is set, the drum is ONLY the (m,1)
    # modes m = 0..len-1: frequencies still come from the physics (f0, K, f01_scale), but
    # amplitude and decay come from these reference-measured tables instead of the
    # strike/pickup formula and d0 + d1*f. r_s, r_p, d0, d1 and the mallet are then unused.
    rung_amp_db: tuple = ()            # initial amplitude of (m,1), dB re the strongest
    rung_decay: tuple = ()             # 1/s per rung
    # measured noise map: stick attack and dense upper content that the ladder does not
    # carry. Band-limited noise per band, with an energy envelope given per time slice.
    noise_bands: tuple = ()            # ((lo_hz, hi_hz), ...); empty = off
    noise_edges: tuple = ()            # time-slice edges in s, e.g. (0, 0.003, 0.01, ...)
    noise_db: tuple = ()               # [band][slice] energy, dB re the modal energy
                                       # inside the first noise_edges[-1] seconds
    noise_seed: int = 0


# ------------------------------------------------------------------ helpers

def f0_for_pitch(pitch_hz: float, K: float, mode=(0, 1), f01_scale: float = 1.0) -> float:
    """In-vacuo f0 such that the air-loaded (m, n) mode sounds at pitch_hz."""
    m, n = mode
    alpha = jn_zeros(m, n)[-1]
    f0 = pitch_hz * jn_zeros(0, 1)[0] / alpha * np.sqrt(1 + K / alpha)
    return f0 / f01_scale if mode == (0, 1) else f0


def retune(d: "DrumPreset", pitch_hz: float) -> "DrumPreset":
    """Same drum, new tuning. The batter's pitch mode (d.pitch_mode) moves to pitch_hz.
    Held fixed (properties of the drum, not the tuning): K, the reso/batter f0 ratio,
    the air column, and the ABSOLUTE air spring between the heads. `coupling` is
    relative to the head frequencies (k = coupling * fb_01 * fr_01), so it is rescaled
    by 1/s^2 to keep k unchanged. Decay (d0, d1) is NOT adjusted: it is only known at
    the fitted tuning."""
    f0 = f0_for_pitch(pitch_hz, d.batter.K, d.pitch_mode, d.batter.f01_scale)
    s = f0 / d.batter.f0
    reso = None if d.reso is None else replace(d.reso, f0=d.reso.f0 * s)
    return replace(d, batter=replace(d.batter, f0=f0), reso=reso, coupling=d.coupling / s ** 2)


def column_freqs(L_eff: float, n: int, c: float = 343.0) -> np.ndarray:
    """Open-open tube resonances f = n*c/(2*L_eff), n = 1..n."""
    return np.arange(1, n + 1) * c / (2 * L_eff)


def add_partials(t: ModeTable, kind: int, freq, amp, decay) -> ModeTable:
    """Append non-membrane partials (m = kind, n = 1..) and re-sort by freq."""
    freq = np.atleast_1d(np.asarray(freq, dtype=float))
    k = len(freq)
    m = np.concatenate([t.m, np.full(k, kind)])
    n = np.concatenate([t.n, np.arange(1, k + 1)])
    f = np.concatenate([t.freq, freq])
    a = np.concatenate([t.amp, np.broadcast_to(amp, k)])
    d = np.concatenate([t.decay, np.broadcast_to(decay, k)])
    o = np.argsort(f)
    return ModeTable(m[o], n[o], f[o], a[o], d[o])


def m_gt0_fraction(t: ModeTable) -> float:
    """Share of membrane amp^2 carried by m>0 modes."""
    return np.sum(t.amp[t.m > 0] ** 2) / np.sum(t.amp[t.m >= 0] ** 2)


# ------------------------------------------------------------------ tables + render

def drum_tables(d: DrumPreset):
    """Returns (out_table, reso_table). reso_table is None for single-head drums."""
    if d.reso is None:
        out, reso = mode_table(d.batter), None
        if d.rung_amp_db:
            k = len(d.rung_amp_db)
            rung = (out.n == 1) & (out.m < k)
            m = out.m[rung]
            out = ModeTable(m, out.n[rung], out.freq[rung],
                            10 ** (np.asarray(d.rung_amp_db)[m] / 20), np.asarray(d.rung_decay)[m])
    else:
        out, reso = two_head_tables(d.batter, d.reso, d.coupling)
    ref = np.max(np.abs(out.amp))
    if d.column_L > 0 and d.column_level != 0:
        out = add_partials(out, COLUMN, column_freqs(d.column_L, d.column_n, d.c),
                           d.column_level * ref, d.column_decay)
    if d.rim_level != 0:
        out = add_partials(out, RIM, d.rim_hz, d.rim_level * ref, d.rim_decay)
    return out, reso


def wire_signal(reso_disp: np.ndarray, modal_peak: float, d: DrumPreset, sr: int) -> np.ndarray:
    """Snare wires: low-passed noise that sounds only while the resonant head's
    |displacement| exceeds the threshold, scaled by how far it exceeds it."""
    peak = np.max(np.abs(reso_disp))
    if peak == 0:
        return np.zeros_like(reso_disp)
    noise = np.random.default_rng(d.wire_seed).standard_normal(len(reso_disp))
    b, a = butter(2, min(d.wire_brightness / (sr / 2), 0.99))
    noise = lfilter(b, a, noise)
    noise /= np.sqrt(np.mean(noise ** 2))
    gate = np.maximum(np.abs(reso_disp) / peak - d.wire_threshold, 0.0)
    return d.wire_level * modal_peak * noise * gate


def modal_energy(t: ModeTable, T: float = np.inf) -> float:
    """Energy (integral of y^2 dt over the first T seconds) of a sum of decaying sines,
    cross terms ignored: a^2 * (1 - exp(-2*d*T)) / (4*d) per partial. Known before the
    hit is rendered, so anything scaled by it stays causal."""
    return float(np.sum(t.amp ** 2 * (1 - np.exp(-2 * t.decay * T)) / (4 * t.decay)))


def band_filter(x, sr, lo, hi, order=8):
    """Band-pass used for BOTH measuring and synthesising the noise map, so the two agree.
    Steep (order 8) because neighbouring bands can differ by 15 dB or more: with gentler
    skirts the louder band leaks into the quieter one and is counted twice."""
    return sosfilt(butter(order, [lo / (sr / 2), min(hi / (sr / 2), 0.99)], "band", output="sos"), x)


def _envelope(t, mid, log_a):
    """Log-amplitude through the nodes (mid, log_a): held before the first node, linear
    between nodes, last slope continued after the last node but never rising."""
    slope = min((log_a[-1] - log_a[-2]) / (mid[-1] - mid[-2]), 0.0)
    return np.where(t <= mid[-1], np.interp(t, mid, log_a), log_a[-1] + slope * (t - mid[-1]))


def band_time_energy(y, sr, bands, edges):
    """Energy [band][slice]: band-pass (band_filter), then sum of squares inside each
    time slice. This is THE definition of a noise map: fit.measure_noise uses it on
    recordings and noise_signal calibrates against it."""
    out = []
    for lo, hi in bands:
        x = band_filter(y, sr, lo, hi)
        out.append([np.sum(x[int(a * sr):int(b * sr)] ** 2) / sr for a, b in zip(edges[:-1], edges[1:])])
    return np.array(out)


NOISE_INSET = 2 ** (1 / 6)   # each synthesis band stops 1/6 octave short of its edges


def noise_signal(d: DrumPreset, table: ModeTable, n_iter: int = 15) -> np.ndarray:
    """Noise part of the hit from the measured noise map. Per band: white noise through a
    band-pass, times an envelope whose log-amplitude is interpolated between the slice
    centres (_envelope). The envelope nodes are calibrated by analysis-by-synthesis:
    the noise is measured with band_time_energy, exactly as a recording would be, and
    the nodes are corrected until it shows the map. That absorbs filter delay, the
    steep-fall overshoot inside long slices and what leaks between bands. Synthesis bands
    are inset from the band edges, where two analysis bands overlap and a flat band of
    noise would be counted in both. The calibrated nodes depend only on the preset, so a
    real-time port can compute them offline."""
    p = d.batter
    n = int(p.sr * p.duration)
    t = np.arange(n) / p.sr
    edges = np.asarray(d.noise_edges, dtype=float)
    mid, dur = 0.5 * (edges[:-1] + edges[1:]), np.diff(edges)
    target = modal_energy(table, edges[-1]) * 10 ** (np.asarray(d.noise_db) / 10)
    rng = np.random.default_rng(d.noise_seed)
    lead = int(0.05 * p.sr)                      # let the filter settle before t = 0
    raw = []
    for lo, hi in d.noise_bands:
        x = band_filter(rng.standard_normal(n + lead), p.sr, lo * NOISE_INSET, hi / NOISE_INSET)[lead:]
        raw.append(x / np.sqrt(np.mean(x ** 2)))
    log_a = 0.5 * np.log(target / dur)           # unit-rms noise at amplitude A: energy A^2 * dur

    def synth():
        return sum(x * np.exp(_envelope(t, mid, la)) for x, la in zip(raw, log_a))

    for _ in range(n_iter):
        got = band_time_energy(synth(), p.sr, d.noise_bands, edges)
        log_a = log_a + np.clip(0.5 * np.log(target / np.maximum(got, 1e-300)), -1.5, 1.5)
    return synth()


def render_drum(d: DrumPreset) -> np.ndarray:
    out, reso = drum_tables(d)
    y = render(out, d.batter)
    if d.noise_bands:
        y = y + noise_signal(d, out)
    if d.wires and reso is not None:
        y = y + wire_signal(render(reso, d.batter), np.max(np.abs(y)), d, d.batter.sr)
    return y


# ------------------------------------------------------------------ presets (placeholders)

SNARE_STRIKE_R = {"center": 0.0, "edge": 0.8, "rimshot": 0.3}


def snare(strike: str = "center", wires: bool = True) -> DrumPreset:
    """Marching snare: high tension, short decays, many upper modes, two heads."""
    common = dict(M=10, N=6, d0=25.0, d1=0.004, duration=1.0, contact_s=0.0003, fc=9000.0)
    batter = DrumParams(f0=380.0, r_s=SNARE_STRIKE_R[strike], r_p=0.3, **common)
    reso = DrumParams(f0=430.0, r_p=0.3, **common)
    return DrumPreset(name="snare_" + strike, batter=batter, reso=reso, coupling=0.2,
                      wires=wires, rim_level=0.5 if strike == "rimshot" else 0.0)


def tenor(pitch_hz: float, L_eff: float, K: float = 0.5) -> DrumPreset:
    """Tenor: single head, open-bottom shell as an open tube of length L_eff.
    Tuned by the (1,1) mode: on real tenors that is the strongest, pitch-carrying
    partial (reference fit, see PROGRESS.md), not the (0,1) mode. L_eff = 0: no column."""
    batter = DrumParams(f0=f0_for_pitch(pitch_hz, K, (1, 1)), K=K, M=8, N=4, r_s=0.5, r_p=0.3,
                        d0=8.0, d1=0.003, duration=1.5, contact_s=0.0005, fc=6000.0)
    return DrumPreset(name="tenor_%gHz" % pitch_hz, batter=batter, pitch_mode=(1, 1),
                      column_L=L_eff, column_level=0.3)


def bass(pitch_hz: float, K: float = 2.0, coupling: float = 0.2) -> DrumPreset:
    """Pitched bass drum: two identical heads, strong air loading, soft mallet.
    With identical heads the lowest (in-phase) partial sounds at pitch_hz."""
    head = DrumParams(f0=f0_for_pitch(pitch_hz, K), K=K, M=6, N=4, r_s=0.2, r_p=0.3,
                      d0=4.0, d1=0.01, duration=2.0, contact_s=0.004, fc=1500.0)
    return DrumPreset(name="bass_%gHz" % pitch_hz, batter=head, reso=head, coupling=coupling)


# Fitted to the first real reference (TAMA Fieldstar tenor tuning tutorial, low-bitrate MP3,
# see PROGRESS.md). Per drum: (1,1) pitch in Hz, air loading K, (0,1) scale; mean over the
# clean hits found (count in the comment). A sixth, higher drum (~757 Hz) was heard in one
# ambiguous hit and is left out. The air column is off (shell length unknown).
REFERENCE_TENORS = [
    (587.0, 3.0, 0.78),   # 3 hits, K 2.7-3.3
    (522.0, 2.2, 0.72),   # 2 hits, K 2.2
    (485.0, 1.8, 0.76),   # 1 hit
    (439.0, 2.15, 0.78),  # 2 hits, K 2.1-2.2
    (390.0, 1.75, 0.81),  # 1 hit
]
# Ladder profile and noise map, median over the same 8 hits (fit.tenor_profile).
# One shared profile for all five drums: too few hits per drum to fit each separately.
# Low confidence on decay (hits follow each other every 0.3-0.6 s, tails are cut).
TENOR_RUNG_AMP_DB = (-9.5, 0.0, -10.7, -20.4, -31.3, -26.6, -31.3, -28.5)   # dB, m = 0..7
TENOR_RUNG_DECAY = (13.3, 15.3, 9.1, 6.7, 6.1, 8.2, 6.3, 5.3)             # 1/s, m = 0..7
TENOR_NOISE_BANDS = ((1200.0, 2400.0), (2400.0, 4800.0), (4800.0, 9600.0))
TENOR_NOISE_EDGES = (0.0, 0.003, 0.01, 0.03, 0.08, 0.2, 0.3)
TENOR_NOISE_DB = (
    (-27.0, -20.8, -17.7, -19.6, -20.7, -30.7),   # 1.2-2.4 kHz
    (-26.8, -24.8, -21.2, -23.2, -28.5, -41.8),   # 2.4-4.8 kHz
    (-33.6, -32.7, -29.6, -35.1, -46.5, -58.6),   # 4.8-9.6 kHz
)


def reference_tenors():
    """The five reference-fitted tenors, highest first."""
    out = []
    for pitch, K, scale in REFERENCE_TENORS:
        d = tenor(pitch, 0.0, K=K)
        batter = replace(d.batter, f01_scale=scale, duration=1.0)
        out.append(replace(d, batter=batter, rung_amp_db=TENOR_RUNG_AMP_DB,
                           rung_decay=TENOR_RUNG_DECAY, noise_bands=TENOR_NOISE_BANDS,
                           noise_edges=TENOR_NOISE_EDGES, noise_db=TENOR_NOISE_DB))
    return out
