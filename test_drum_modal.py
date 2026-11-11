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


# ---------------------------------------------------------------- Layer 3

def test_l3_tension_keeps_ratios():
    a, b = mode_table(DrumParams()), mode_table(DrumParams(tension=2.0))
    assert np.allclose(b.freq / b.freq[0], a.freq / a.freq[0], rtol=1e-12)
    assert np.isclose(b.freq[0], a.freq[0] * np.sqrt(2.0))
    assert np.array_equal(a.amp, b.amp)


def test_l3_air_loading():
    ideal = mode_table(DrumParams())
    assert np.array_equal(mode_table(DrumParams(K=0.0)).freq, ideal.freq)   # K=0 == Layer 1
    loaded = mode_table(DrumParams(K=2.0))
    assert np.array_equal(loaded.m, ideal.m) and np.array_equal(loaded.n, ideal.n)
    shift = loaded.freq / ideal.freq
    assert np.all(shift < 1)                      # every mode lowered
    assert np.all(np.diff(shift) > -1e-12)        # low modes lowered the most
    assert shift[0] < shift[-1]
    r_ideal, r_loaded = ideal.freq / ideal.freq[0], loaded.freq / loaded.freq[0]
    assert np.all(r_loaded[1:] > r_ideal[1:])     # ratios stretch upward


def test_l3_longer_contact_less_hf():
    def hf_energy(contact_s):
        t = mode_table(DrumParams(contact_s=contact_s))
        return np.sum(t.amp[t.freq > 1000.0] ** 2)
    assert hf_energy(0.0) > hf_energy(0.0005) > hf_energy(0.003)
    # knobs off == Layer 1 amps; fc rolloff only ever reduces |amp|
    base = mode_table(DrumParams())
    soft = mode_table(DrumParams(fc=800.0))
    assert np.all(np.abs(soft.amp) <= np.abs(base.amp))
    assert np.allclose(soft.amp, base.amp * np.exp(-(base.freq / 800.0) ** 2))


def test_l3_pitch_drop_zero_matches_layer2():
    from drum_modal import render
    p = DrumParams(drop=0.0)
    tb = mode_table(p)
    t = np.arange(int(p.sr * p.duration)) / p.sr
    ref = sum(a * np.exp(-d * t) * np.sin(2 * np.pi * f * t)
              for f, a, d in zip(tb.freq, tb.amp, tb.decay))
    assert np.allclose(render(tb, p), ref, atol=1e-12)
    assert not np.allclose(render(tb, DrumParams(drop=0.05)), ref, atol=1e-3)


# ---------------------------------------------------------------- Layer 4

def test_l4_zero_coupling_gives_independent_tables():
    from drum_modal import two_head_tables
    pb, pr = DrumParams(f0=200.0), DrumParams(f0=260.0, r_p=0.5, d0=5.0)
    tb, tr = mode_table(pb), mode_table(pr)
    bt, rt = two_head_tables(pb, pr, 0.0)
    live = bt.amp != 0
    # batter pickup hears exactly the single-head batter table
    assert np.allclose(bt.freq[live], tb.freq, rtol=1e-12)
    assert np.allclose(bt.amp[live], tb.amp, rtol=1e-12)
    assert np.allclose(bt.decay[live], tb.decay, rtol=1e-12)
    assert np.array_equal(bt.m[live], tb.m) and np.array_equal(bt.n[live], tb.n)
    # resonant head: its own uncoupled frequencies, and never excited
    assert np.all(rt.amp == 0)
    assert np.allclose(rt.freq[rt.m > 0], tr.freq[tr.m > 0], rtol=1e-12)
    assert np.allclose(np.sort(rt.freq[rt.m == 0]),
                       np.sort(np.concatenate([tb.freq[tb.m == 0], tr.freq[tr.m == 0]])), rtol=1e-12)
    # the silent partials in the batter table are the resonant (0,n) modes
    assert np.allclose(bt.freq[~live], tr.freq[tr.m == 0], rtol=1e-12)


def test_l4_coupling_splits_m0_only():
    from drum_modal import two_head_tables
    p = DrumParams()
    t = mode_table(p)
    c = 0.3
    bt, rt = two_head_tables(p, p, c)                 # identical heads
    f0n = t.freq[t.m == 0]
    k = c * f0n[0] ** 2
    pair = bt.freq[bt.m == 0]
    lo = np.array([pair[bt.n[bt.m == 0] == n].min() for n in range(1, p.N + 1)])
    hi = np.array([pair[bt.n[bt.m == 0] == n].max() for n in range(1, p.N + 1)])
    assert np.allclose(lo, f0n)                       # in-phase: unchanged
    assert np.allclose(hi, np.sqrt(f0n ** 2 + 2 * k)) # out-of-phase: raised
    assert np.allclose(bt.freq[bt.m > 0], t.freq[t.m > 0])   # m>0 untouched
    assert np.allclose(bt.amp[bt.m > 0], t.amp[t.m > 0])
    assert np.any(rt.amp[rt.m == 0] != 0) and np.all(rt.amp[rt.m > 0] == 0)
