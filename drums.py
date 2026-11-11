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
from scipy.signal import butter, lfilter
from scipy.special import jn_zeros

from drum_modal import DrumParams, ModeTable, mode_table, render, two_head_tables

COLUMN, RIM = -1, -2   # values of ModeTable.m marking non-membrane partials


@dataclass
class DrumPreset:
    name: str
    batter: DrumParams
    reso: Optional[DrumParams] = None  # None = single head (tenor)
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


# ------------------------------------------------------------------ helpers

def f0_for_pitch(pitch_hz: float, K: float) -> float:
    """In-vacuo f0 such that the air-loaded (0,1) mode sounds at pitch_hz."""
    return pitch_hz * np.sqrt(1 + K / jn_zeros(0, 1)[0])


def retune(d: "DrumPreset", pitch_hz: float) -> "DrumPreset":
    """Same drum, new tuning. The batter's air-loaded (0,1) mode moves to pitch_hz.
    Held fixed (properties of the drum, not the tuning): K, the reso/batter f0 ratio,
    the air column, and the ABSOLUTE air spring between the heads. `coupling` is
    relative to the head frequencies (k = coupling * fb_01 * fr_01), so it is rescaled
    by 1/s^2 to keep k unchanged. Decay (d0, d1) is NOT adjusted: it is only known at
    the fitted tuning."""
    f0 = f0_for_pitch(pitch_hz, d.batter.K)
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


def render_drum(d: DrumPreset) -> np.ndarray:
    out, reso = drum_tables(d)
    y = render(out, d.batter)
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
    """Tenor: single head, open-bottom shell as an open tube of length L_eff."""
    batter = DrumParams(f0=f0_for_pitch(pitch_hz, K), K=K, M=8, N=4, r_s=0.5, r_p=0.3,
                        d0=8.0, d1=0.003, duration=1.5, contact_s=0.0005, fc=6000.0)
    return DrumPreset(name="tenor_%gHz" % pitch_hz, batter=batter,
                      column_L=L_eff, column_level=0.3)


def bass(pitch_hz: float, K: float = 2.0, coupling: float = 0.2) -> DrumPreset:
    """Pitched bass drum: two identical heads, strong air loading, soft mallet.
    With identical heads the lowest (in-phase) partial sounds at pitch_hz."""
    head = DrumParams(f0=f0_for_pitch(pitch_hz, K), K=K, M=6, N=4, r_s=0.2, r_p=0.3,
                      d0=4.0, d1=0.01, duration=2.0, contact_s=0.004, fc=1500.0)
    return DrumPreset(name="bass_%gHz" % pitch_hz, batter=head, reso=head, coupling=coupling)
