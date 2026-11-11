"""Modal synthesis core for marching percussion. Pure functions, no globals.

Pipeline: DrumParams -> mode_table() -> render() -> write_wav() / plot_spectrum()
See SPEC.md for the layer plan. Layers implemented so far: 1, 2, 3 (no doublets), 4.
"""
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from scipy.special import jn_zeros, jv


@dataclass
class DrumParams:
    # Layer 1: ideal membrane
    f0: float = 200.0      # Hz, frequency of the (0,1) mode
    M: int = 6             # m = 0..M nodal diameters
    N: int = 4             # n = 1..N nodal circles
    r_s: float = 0.4       # strike radius, 0..1
    r_p: float = 0.3       # pickup radius, 0..1 (same angle as strike)
    # Layer 2: render
    d0: float = 2.0        # 1/s, frequency-independent decay rate
    d1: float = 0.001      # 1/s per Hz, frequency-dependent decay rate
    sr: int = 48000        # Hz
    duration: float = 2.0  # s
    # Layer 3: shared physics knobs (defaults = off, i.e. identical to Layers 1-2)
    tension: float = 1.0       # relative head tension; f scales as sqrt(tension)
    K: float = 0.0             # air loading; f = f_ideal / sqrt(1 + K/alpha_mn)
    contact_s: float = 0.0     # s, raised-cosine force pulse duration (0 = impulse)
    fc: float = float("inf")   # Hz, stick rolloff exp(-(f/fc)^2) on amps
    drop: float = 0.0          # pitch drop depth; f(t) = f*(1 + drop*exp(-t/tau_d))
    tau_d: float = 0.05        # s, pitch drop time constant
    # Fitted from real tenors: the (0,1) mode sits below what the K curve allows
    f01_scale: float = 1.0     # multiplies the (0,1) frequency only (1 = on the K curve)


class ModeTable(NamedTuple):
    """One entry per mode; all arrays have the same length, sorted by freq."""
    m: np.ndarray      # nodal diameters
    n: np.ndarray      # nodal circles
    freq: np.ndarray   # Hz
    amp: np.ndarray    # signed linear amplitude
    decay: np.ndarray  # 1/s, envelope = exp(-decay * t)


# ---------------------------------------------------------------- Layer 3

def pulse_gain(freq: np.ndarray, contact_s: float) -> np.ndarray:
    """Spectrum of a raised-cosine force pulse of duration contact_s, unity at DC.
    F(x) = sinc(x) / (1 - x^2), x = f * contact_s (limit 1/2 at x = 1).
    Signed: negative lobes flip the phase of that mode."""
    x = np.asarray(freq, dtype=float) * contact_s
    den = 1 - x ** 2
    safe = np.abs(den) > 1e-9
    return np.where(safe, np.sinc(x) / np.where(safe, den, 1.0), 0.5)


def strike_gain(freq: np.ndarray, p: DrumParams) -> np.ndarray:
    """Per-mode amp factor from the mallet: contact pulse spectrum * rolloff."""
    return pulse_gain(freq, p.contact_s) * np.exp(-(np.asarray(freq) / p.fc) ** 2)


# ---------------------------------------------------------------- Layer 1

def mode_table(p: DrumParams) -> ModeTable:
    """Circular membrane: Bessel-zero frequencies, strike/pickup amps.
    Layer 3 knobs applied here: tension, air loading K, mallet (contact_s, fc)."""
    alpha_01 = jn_zeros(0, 1)[0]
    f0 = p.f0 * np.sqrt(p.tension)
    ms, ns, freqs, amps = [], [], [], []
    for m in range(p.M + 1):
        alpha = jn_zeros(m, p.N)                       # alpha_m1..alpha_mN
        norm = (np.pi if m == 0 else np.pi / 2) * jv(m + 1, alpha) ** 2
        ms.append(np.full(p.N, m))
        ns.append(np.arange(1, p.N + 1))
        f = f0 * alpha / alpha_01 / np.sqrt(1 + p.K / alpha)
        if m == 0:
            f[0] *= p.f01_scale
        freqs.append(f)
        amps.append(jv(m, alpha * p.r_s) * jv(m, alpha * p.r_p) / norm)
    m, n, freq, amp = (np.concatenate(x) for x in (ms, ns, freqs, amps))
    order = np.argsort(freq)
    m, n, freq, amp = m[order], n[order], freq[order], amp[order]
    amp = amp * strike_gain(freq, p)
    return ModeTable(m, n, freq, amp, p.d0 + p.d1 * freq)


# ---------------------------------------------------------------- Layer 4

def two_head_tables(pb: DrumParams, pr: DrumParams, coupling: float):
    """Batter (pb) + resonant (pr) head coupled through the enclosed air.
    Returns (batter_table, reso_table): motion at the batter pickup (pb.r_p) and at
    the resonant-head pickup (pr.r_p), for a strike on the batter at pb.r_s.

    Each (0,n) pair is two oscillators joined by an air spring, in f^2 units:
        [[fb^2 + k, -k], [-k, fr^2 + k]],   k = coupling * fb_01 * fr_01
    Eigenvalues -> two partials (in-phase: lower, out-of-phase: raised).
    Eigenvector V[:, j] = (batter share, reso share) of partial j.
    m>0 modes displace no net air volume and stay uncoupled; the resonant head's
    m>0 modes are never excited (amp 0). Each table has 2N more rows than a
    single head: every (0,n) appears twice."""
    assert pb.N == pr.N
    tb, tr = mode_table(pb), mode_table(pr)
    fb, fr = tb.freq[tb.m == 0], tr.freq[tr.m == 0]        # ordered n = 1..N
    alpha = jn_zeros(0, pb.N)
    norm = np.pi * jv(1, alpha) ** 2
    strike = jv(0, alpha * pb.r_s) / norm
    shape_b = strike * jv(0, alpha * pb.r_p)               # strike -> batter pickup
    shape_r = strike * jv(0, alpha * pr.r_p)               # strike -> reso pickup
    k = coupling * fb[0] * fr[0]

    n_out, f_out, ab, ar, dec = [], [], [], [], []
    for i in range(pb.N):
        w, V = np.linalg.eigh(np.array([[fb[i] ** 2 + k, -k], [-k, fr[i] ** 2 + k]]))
        for j in range(2):
            f = np.sqrt(w[j])
            g = strike_gain(f, pb)
            n_out.append(i + 1)
            f_out.append(f)
            ab.append(shape_b[i] * V[0, j] ** 2 * g)
            ar.append(shape_r[i] * V[0, j] * V[1, j] * g)
            # loss of each partial = energy-weighted mix of the two heads' losses
            dec.append(V[0, j] ** 2 * (pb.d0 + pb.d1 * f) + V[1, j] ** 2 * (pr.d0 + pr.d1 * f))
    n_out, f_out, dec = np.array(n_out), np.array(f_out), np.array(dec)

    def merge(single: ModeTable, amp_m0, keep_amp: bool) -> ModeTable:
        hi = single.m > 0
        m = np.concatenate([single.m[hi], np.zeros_like(n_out)])
        n = np.concatenate([single.n[hi], n_out])
        freq = np.concatenate([single.freq[hi], f_out])
        amp = np.concatenate([single.amp[hi] * keep_amp, amp_m0])
        decay = np.concatenate([single.decay[hi], dec])
        o = np.argsort(freq)
        return ModeTable(m[o], n[o], freq[o], amp[o], decay[o])

    return merge(tb, np.array(ab), True), merge(tr, np.array(ar), False)


# ---------------------------------------------------------------- Layer 2

def render(table: ModeTable, p: DrumParams) -> np.ndarray:
    """Sum of exponentially decaying sines. Modes above Nyquist are skipped.
    Output is NOT normalized (raw sum of table amps).
    Pitch drop: f(t) = f*(1 + drop*exp(-t/tau_d)); phase is its exact integral,
    2*pi*f*tw with warped time tw = t + drop*tau_d*(1 - exp(-t/tau_d))."""
    t = np.arange(int(p.sr * p.duration)) / p.sr
    tw = t if p.drop == 0 else t + p.drop * p.tau_d * (1 - np.exp(-t / p.tau_d))
    y = np.zeros_like(t)
    for f, a, d in zip(table.freq, table.amp, table.decay):
        if f * (1 + max(p.drop, 0.0)) < p.sr / 2:
            y += a * np.exp(-d * t) * np.sin(2 * np.pi * f * tw)
    return y


def spectrum(y: np.ndarray, sr: int):
    """Magnitude spectrum (no window). Returns (freqs_hz, magnitude)."""
    return np.fft.rfftfreq(len(y), 1 / sr), np.abs(np.fft.rfft(y))


def top_modes(table: ModeTable, k: int) -> np.ndarray:
    """Indices of the k modes with the largest |amp|."""
    return np.argsort(-np.abs(table.amp))[:k]


def write_wav(path: str, y: np.ndarray, sr: int) -> None:
    """Write WAV, peak-normalized to 0.9 (file level only, not physics)."""
    import soundfile as sf
    sf.write(path, 0.9 * y / np.max(np.abs(y)), sr)


def plot_spectrum(path: str, y: np.ndarray, table: ModeTable, p: DrumParams,
                  title: str = "") -> None:
    """Spectrum in dB with predicted mode frequencies as vertical lines."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f, mag = spectrum(y, p.sr)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(f, 20 * np.log10(mag / mag.max() + 1e-12), lw=0.8)
    for fm in table.freq:
        ax.axvline(fm, color="r", lw=0.5, alpha=0.5)
    ax.set(xlim=(0, min(p.sr / 2, table.freq.max() * 1.1)), ylim=(-100, 5),
           xlabel="Hz", ylabel="dB", title=title or "spectrum vs predicted modes")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
