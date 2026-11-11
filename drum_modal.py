"""Modal synthesis core for marching percussion. Pure functions, no globals.

Pipeline: DrumParams -> mode_table() -> render() -> write_wav() / plot_spectrum()
See SPEC.md for the layer plan. Layers implemented so far: 1, 2, 3 (no doublets).
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
        freqs.append(f0 * alpha / alpha_01 / np.sqrt(1 + p.K / alpha))
        amps.append(jv(m, alpha * p.r_s) * jv(m, alpha * p.r_p) / norm)
    m, n, freq, amp = (np.concatenate(x) for x in (ms, ns, freqs, amps))
    order = np.argsort(freq)
    m, n, freq, amp = m[order], n[order], freq[order], amp[order]
    amp = amp * strike_gain(freq, p)
    return ModeTable(m, n, freq, amp, p.d0 + p.d1 * freq)


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
