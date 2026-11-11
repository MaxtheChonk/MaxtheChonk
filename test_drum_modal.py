import numpy as np
from scipy.special import jn_zeros

from drum_modal import DrumParams, mode_table


# ---------------------------------------------------------------- Layer 1

def test_l1_ratios():
    t = mode_table(DrumParams())
    ratios = t.freq / t.freq[0]
    expected = [1.000, 1.594, 2.136, 2.296, 2.653, 2.918]
    assert np.allclose(ratios[:6], expected, atol=0.001)


def test_l1_center_strike_only_m0():
    t = mode_table(DrumParams(r_s=0.0, r_p=0.5))
    assert np.all(t.amp[t.m > 0] == 0)
    assert np.any(np.abs(t.amp[t.m == 0]) > 1e-3)


def test_l1_nodal_circle_strike():
    # Mode (0,2) has its nodal circle at r = alpha_01 / alpha_02.
    a = jn_zeros(0, 2)
    t = mode_table(DrumParams(r_s=a[0] / a[1], r_p=0.5))
    hit = (t.m == 0) & (t.n == 2)
    assert abs(t.amp[hit][0]) < 1e-12
    assert np.max(np.abs(t.amp)) > 1e-3   # other modes still excited


# ---------------------------------------------------------------- Layer 2

def test_l2_fft_peaks_match_top8_modes():
    from scipy.signal import find_peaks
    from drum_modal import render, spectrum, top_modes
    p = DrumParams()
    t = mode_table(p)
    f, mag = spectrum(render(t, p), p.sr)
    bin_hz = f[1] - f[0]
    peak_freqs = f[find_peaks(mag)[0]]          # every local maximum
    for i in top_modes(t, 8):
        nearest = peak_freqs[np.argmin(np.abs(peak_freqs - t.freq[i]))]
        assert abs(nearest - t.freq[i]) <= bin_hz, (t.m[i], t.n[i], t.freq[i], nearest)
