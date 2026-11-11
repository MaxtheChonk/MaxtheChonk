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


# ---------------------------------------------------------------- Layer 5

def test_l5_snare_wires_off_is_pure_modal():
    import drums
    from drum_modal import render, two_head_tables
    d = drums.snare("center", wires=False)
    modal = render(two_head_tables(d.batter, d.reso, d.coupling)[0], d.batter)
    assert np.array_equal(drums.render_drum(d), modal)
    wet = drums.render_drum(drums.snare("center", wires=True))
    assert not np.allclose(wet, modal)


def test_l5_snare_edge_more_m_gt0_than_center():
    import drums
    center = drums.drum_tables(drums.snare("center", wires=False))[0]
    edge = drums.drum_tables(drums.snare("edge", wires=False))[0]
    assert drums.m_gt0_fraction(center) == 0.0
    assert drums.m_gt0_fraction(edge) > drums.m_gt0_fraction(center)


def test_l5_snare_rimshot_adds_one_click_partial():
    import drums
    t = drums.drum_tables(drums.snare("rimshot", wires=False))[0]
    assert np.sum(t.m == drums.RIM) == 1
    assert np.sum(drums.drum_tables(drums.snare("edge", wires=False))[0].m == drums.RIM) == 0


def test_l5_tenor_column_modes_at_predicted_freqs():
    import drums
    from dataclasses import replace
    from scipy.signal import find_peaks
    from drum_modal import spectrum
    L, c = 0.25, 343.0
    d = drums.tenor(300.0, L)
    # low decay so lines are narrow enough for a 1-bin check (as in the Layer 2 test)
    d = replace(d, batter=replace(d.batter, d0=2.0, d1=0.001, duration=2.0), column_decay=2.0)
    table = drums.drum_tables(d)[0]
    predicted = np.arange(1, d.column_n + 1) * c / (2 * L)
    assert np.allclose(table.freq[table.m == drums.COLUMN], predicted)
    f, mag = spectrum(drums.render_drum(d), d.batter.sr)
    peaks = f[find_peaks(mag)[0]]
    for fc in predicted:
        assert abs(peaks[np.argmin(np.abs(peaks - fc))] - fc) <= f[1] - f[0], fc


def test_l5_tenor_and_bass_tuned_to_pitch():
    import drums
    for pitch in (233.0, 277.0, 311.0, 370.0):
        t = drums.drum_tables(drums.tenor(pitch, 0.25))[0]
        assert np.isclose(t.freq[(t.m == 1) & (t.n == 1)][0], pitch)   # tenors tune by (1,1)
    for pitch in (55.0, 65.0, 73.0, 82.0, 98.0):
        d = drums.bass(pitch)
        assert d.batter.K > 0
        assert np.isclose(drums.drum_tables(d)[0].freq[0], pitch)   # lowest partial


def test_l5_bass_mallet_softer_than_snare_stick():
    import drums
    b, s = drums.bass(65.0).batter, drums.snare().batter
    assert b.contact_s > s.contact_s and b.fc < s.fc


# ---------------------------------------------------------------- Layer 6
# No reference recordings yet, so the fitter is checked on synthetic hits: render a
# preset with known parameters, fit from a deliberately wrong template, recover them.

def test_l6_fit_recovers_tenor():
    # Truth values in the range the real tenor reference showed (K about 2-3, (0,1) at
    # about 0.7-0.8 of its K-curve frequency). For K near 0 the ladder fit is ambiguous
    # (see fit_ladder) and is not expected to recover K.
    import drums, fit
    from dataclasses import replace
    for pitch, K, scale, L in ((440.0, 2.2, 0.75, 0.0), (590.0, 3.0, 0.70, 0.2)):
        truth = drums.tenor(pitch, L, K=K)
        truth = fit.with_tuning(replace(truth, batter=replace(truth.batter, f01_scale=scale)),
                                pitch, K, 0.0)
        r = fit.fit_drum(drums.render_drum(truth), truth.batter.sr, drums.tenor(250.0, L, K=1.5))
        b = r.preset.batter
        assert np.isclose(r.pitch, pitch, rtol=3e-3)
        assert np.isclose(b.K, K, rtol=0.10)
        assert abs(b.f01_scale - scale) < 0.02
        assert np.isclose(b.d0, truth.batter.d0, rtol=0.05) and np.isclose(b.d1, truth.batter.d1, rtol=0.05)


def test_l6_fit_recovers_bass_with_coupling():
    import drums, fit
    truth = drums.bass(98.0, K=3.0, coupling=0.4)
    r = fit.fit_drum(drums.render_drum(truth), truth.batter.sr, drums.bass(80.0, K=0.5, coupling=0.05))
    b = r.preset.batter
    assert np.isclose(b.f0, truth.batter.f0, rtol=5e-3)
    assert abs(b.K - 3.0) < 0.1 and abs(r.preset.coupling - 0.4) < 0.02
    assert np.isclose(b.d0, truth.batter.d0, rtol=0.1) and np.isclose(b.d1, truth.batter.d1, rtol=0.1)
    assert "rms freq error" in fit.report(r)


def test_l6_fit_recovers_snare_modal_and_wires():
    import drums, fit
    from dataclasses import replace
    dry = drums.snare("edge", wires=False)
    r = fit.fit_drum(drums.render_drum(dry), dry.batter.sr, fit.with_tuning(dry, 330.0, 1.0, 0.5))
    assert np.isclose(r.preset.batter.f0, 380.0, rtol=0.01)
    assert abs(r.preset.coupling - 0.2) < 0.02
    assert np.sqrt(np.mean(r.cents ** 2)) < 5.0
    # wires: different noise seed than the model uses, so only statistics can match
    wet = replace(dry, wires=True, wire_threshold=0.15, wire_brightness=4000.0,
                  wire_level=0.8, wire_seed=7)
    w = fit.fit_wires(drums.render_drum(wet), wet.batter.sr, r).preset
    assert abs(w.wire_threshold - 0.15) < 0.05
    assert np.isclose(w.wire_brightness, 4000.0, rtol=0.25)
    assert np.isclose(w.wire_level, 0.8, rtol=0.3)


# ---------------------------------------------------------------- reference screening

def test_screen_passes_clean_hits():
    import drums, fit
    for d in (drums.snare("edge", wires=False), drums.snare("center", wires=True),
              drums.tenor(370.0, 0.2), drums.bass(98.0), drums.bass(65.0)):
        s = fit.screen(drums.render_drum(d), d.batter.sr)
        assert s.trust_freq and s.trust_decay and s.n_hits == 1 and not s.notes, (d.name, s.notes)


def test_screen_flags_bad_references():
    import drums, fit
    from scipy.signal import fftconvolve
    sr = 48000
    y = drums.render_drum(drums.tenor(370.0, 0.2))
    peak = np.max(np.abs(y))

    s = fit.screen(np.clip(y, -0.5 * peak, 0.5 * peak), sr)             # clipped
    assert s.clipped and not s.trust_freq and not s.trust_decay

    two = y.copy()
    two[int(0.3 * sr):] += 0.8 * y[:len(y) - int(0.3 * sr)]               # second hit
    s = fit.screen(two, sr)
    assert s.n_hits == 2 and s.trust_freq and not s.trust_decay

    s = fit.screen(y[:len(y) // 10], sr)                                  # tail cut off
    assert s.tail_db > -40 and not s.trust_decay

    noisy = y + peak * 10 ** (-25 / 20) * np.random.default_rng(0).standard_normal(len(y))
    s = fit.screen(noisy, sr)                                             # noise floor
    assert s.snr_db < 30 and not s.trust_freq and not s.trust_decay

    w = int(0.01 * sr)                                                    # 4:1 compressor
    env = np.sqrt(np.maximum(fftconvolve(y ** 2, np.ones(w) / w, mode="same"), 0)) + 1e-12
    over = np.maximum(20 * np.log10(env / env.max()) + 30, 0)
    s = fit.screen(y * 10 ** (-over * 0.75 / 20), sr)
    assert s.decay_ratio > 1.6 and s.trust_freq and not s.trust_decay

    # reverb longer and louder than the drum's own tail (snare: short decay)
    ys = drums.render_drum(drums.snare("edge", wires=False))
    n = int(1.2 * sr)
    ir = np.random.default_rng(1).standard_normal(n) * np.exp(-6.9 * np.arange(n) / n)
    ir *= 0.3 / np.sqrt(np.sum(ir ** 2))
    ir[0] = 1.0
    s = fit.screen(fftconvolve(ys, ir), sr)
    assert s.decay_ratio < 0.6 and s.trust_freq and not s.trust_decay


def test_slice_hits_finds_each_hit_in_a_long_take():
    import drums, fit
    sr = 48000
    pitches = [370.0, 311.0, 277.0, 233.0]
    gap = np.zeros(int(0.2 * sr))
    take = np.concatenate([np.concatenate([gap, drums.render_drum(drums.tenor(p, 0.25))]) for p in pitches])
    take += 1e-4 * np.max(np.abs(take)) * np.random.default_rng(0).standard_normal(len(take))
    cuts = fit.slice_hits(take, sr)
    assert len(cuts) == len(pitches)
    for (a, b), pitch in zip(cuts, pitches):
        s = fit.screen(take[a:b], sr)
        assert s.n_hits == 1 and s.trust_freq and s.trust_decay, s.notes
        f, level = fit.extract_peaks(take[a:b], sr)
        assert np.min(np.abs(f - pitch)) < 2e-3 * pitch  # the tuned (1,1) mode is among the peaks


# ---------------------------------------------------------------- retuning

def test_retune_moves_pitch_and_keeps_the_drum():
    import drums

    def split(d):   # f_hi^2 - f_lo^2 of the (0,1) pair = 2 * absolute air spring
        t = drums.drum_tables(d)[0]
        pair = t.freq[(t.m == 0) & (t.n == 1)]
        return pair.max() ** 2 - pair.min() ** 2

    low = drums.bass(65.0)
    high = drums.retune(low, 98.0)
    assert np.isclose(drums.drum_tables(high)[0].freq[0], 98.0)
    assert high.batter.K == low.batter.K
    assert np.isclose(split(high), split(low), rtol=1e-9)
    assert np.isclose(drums.drum_tables(drums.retune(high, 65.0))[0].freq[0], 65.0)   # round trip

    t1 = drums.tenor(300.0, 0.25)
    a, b = drums.drum_tables(t1)[0], drums.drum_tables(drums.retune(t1, 350.0))[0]
    assert np.isclose(b.freq[(b.m == 1) & (b.n == 1)][0], 350.0)
    assert np.allclose(a.freq[a.m == drums.COLUMN], b.freq[b.m == drums.COLUMN])  # shell unchanged
    mem = a.m >= 0
    assert np.allclose(b.freq[b.m >= 0] / a.freq[mem], 350.0 / 300.0)             # head ratios unchanged


# ---------------------------------------------------------------- tenor pitch mode + (0,1) offset

def test_f01_scale_moves_only_the_01_mode():
    a = mode_table(DrumParams(K=2.0))
    assert np.array_equal(mode_table(DrumParams(K=2.0, f01_scale=1.0)).freq, a.freq)   # off = unchanged
    b = mode_table(DrumParams(K=2.0, f01_scale=0.75))
    is01 = (a.m == 0) & (a.n == 1)
    assert np.isclose(b.freq[(b.m == 0) & (b.n == 1)][0], 0.75 * a.freq[is01][0])
    assert np.allclose(np.sort(b.freq[~((b.m == 0) & (b.n == 1))]), np.sort(a.freq[~is01]))


def test_f0_for_pitch_by_mode():
    import drums
    for mode in ((0, 1), (1, 1), (2, 1)):
        p = DrumParams(f0=drums.f0_for_pitch(440.0, 2.0, mode), K=2.0)
        t = mode_table(p)
        assert np.isclose(t.freq[(t.m == mode[0]) & (t.n == mode[1])][0], 440.0)
    p = DrumParams(f0=drums.f0_for_pitch(100.0, 2.0, (0, 1), 0.8), K=2.0, f01_scale=0.8)
    assert np.isclose(mode_table(p).freq[0], 100.0)
    assert drums.tenor(440.0, 0.25).pitch_mode == (1, 1) and drums.bass(65.0).pitch_mode == (0, 1)


def test_fit_ladder_picks_the_11_anchor():
    import fit
    truth = fit.ladder_freqs(440.0, 2.1)                  # (0,1) .. (5,1)
    f = np.concatenate([[0.76 * truth[0]], truth[1:]])    # low (0,1) plus the ladder
    level = np.array([-14.0, -6.0, 0.0, -15.0, -16.0, -18.0])   # (2,1) is the loudest peak
    f11, K, rms = fit.fit_ladder(f, level)
    assert np.isclose(f11, 440.0, rtol=1e-3) and np.isclose(K, 2.1, rtol=0.02) and rms < 1.0
    assert np.isclose(fit.find_f01(f, level, f11, K), 0.76, atol=0.01)


# ---------------------------------------------------------------- measured tenor profile + attack

def test_rung_profile_replaces_formula_amps_but_keeps_physics_freqs():
    import drums
    from dataclasses import replace
    d = replace(drums.reference_tenors()[0], pop_amp=0.0)
    t = drums.drum_tables(d)[0]
    k = len(drums.TENOR_RUNG_AMP_DB)
    assert np.all(t.n == 1) and sorted(t.m) == list(range(k))            # only the ladder
    formula = drums.drum_tables(replace(d, rung_amp_db=(), rung_decay=()))[0]
    for m in range(k):
        assert t.freq[t.m == m][0] == formula.freq[(formula.m == m) & (formula.n == 1)][0]
        assert np.isclose(20 * np.log10(t.amp[t.m == m][0]), drums.TENOR_RUNG_AMP_DB[m])
        assert t.decay[t.m == m][0] == drums.TENOR_RUNG_DECAY[m]
    up = drums.drum_tables(drums.retune(d, 650.0))[0]                    # still retunable
    assert np.isclose(up.freq[up.m == 1][0], 650.0) and np.array_equal(up.amp, t.amp[np.argsort(t.m)][up.m])


def test_noise_off_is_pure_modal_and_on_matches_its_map():
    import drums, fit
    from dataclasses import replace
    from drum_modal import render
    d = replace(drums.reference_tenors()[1], pop_amp=0.0)
    dry = replace(d, noise_bands=())
    table = drums.drum_tables(dry)[0]
    modal = render(table, dry.batter)
    assert np.array_equal(drums.render_drum(dry), modal)
    # the noise alone carries the energies its map says, per band and time slice
    noise = drums.render_drum(d) - modal
    e = fit.band_time_energy(noise, d.batter.sr, d.noise_bands, d.noise_edges)
    got = 10 * np.log10(e / drums.modal_energy(table, d.noise_edges[-1]))
    assert np.allclose(got, d.noise_db, atol=3.0), np.round(got - np.array(d.noise_db), 1)


def test_tenor_profile_round_trip():
    # Render the reference tenors, measure them like a recording, get the profile back.
    import drums, fit
    from dataclasses import replace
    ref = [replace(d, pop_amp=0.0) for d in drums.reference_tenors()]   # ladder + noise only
    amp, dec = np.array(drums.TENOR_RUNG_AMP_DB), np.array(drums.TENOR_RUNG_DECAY)

    # (a) ladder part alone (noise off): everything comes back tightly
    p = fit.tenor_profile([drums.render_drum(replace(d, noise_bands=())) for d in ref], 48000)
    for (f11, K, scale), (pitch, K0, scale0) in zip(p["fits"], drums.REFERENCE_TENORS):
        assert np.isclose(f11, pitch, rtol=2e-3) and np.isclose(K, K0, rtol=0.05) and abs(scale - scale0) < 0.02
    assert np.allclose(p["rung_amp_db"], amp, atol=1.5)
    assert np.allclose(p["rung_decay"], dec, rtol=0.1)

    # (b) full hits (noise on): pitch, (0,1) and the noise map come back tightly, and so do
    # the rungs that stand above the noise. KNOWN LIMIT, not asserted tightly: rungs weaker
    # than -25 dB sit inside the 1.2-2.4 kHz noise, so their amplitude and decay, and the K
    # of a drum whose upper rungs fall there (587 Hz: 3.5 vs 3.0), are only roughly recovered.
    p = fit.tenor_profile([drums.render_drum(d) for d in ref], 48000)
    for (f11, K, scale), (pitch, K0, scale0) in zip(p["fits"], drums.REFERENCE_TENORS):
        assert np.isclose(f11, pitch, rtol=2e-3) and abs(scale - scale0) < 0.02
        assert np.isclose(K, K0, rtol=0.2)
    strong = amp > -25.0
    assert np.allclose(np.array(p["rung_amp_db"])[strong], amp[strong], atol=1.5)
    assert np.allclose(np.array(p["rung_decay"])[strong], dec[strong], rtol=0.1)
    assert np.allclose(p["noise_db"], drums.TENOR_NOISE_DB, atol=3.0)


# ---------------------------------------------------------------- attack pop + room

def test_pop_is_one_extra_partial_and_leaves_the_rest_alone():
    import drums
    from dataclasses import replace
    d = drums.reference_tenors()[2]
    off = drums.drum_tables(replace(d, pop_amp=0.0))[0]
    on = drums.drum_tables(d)[0]
    pop = on.m == drums.POP
    assert pop.sum() == 1 and not np.any(off.m == drums.POP)
    pitch = off.freq[off.m == 1][0]
    assert np.isclose(on.freq[pop][0], d.pop_ratio * pitch)
    assert np.isclose(on.amp[pop][0], d.pop_amp * np.max(off.amp)) and on.decay[pop][0] == d.pop_decay
    assert np.array_equal(on.freq[~pop], off.freq) and np.array_equal(on.amp[~pop], off.amp)
    # the noise is referred to the ringing part only, so the pop does not change it
    dry_on, dry_off = replace(d, noise_bands=()), replace(d, noise_bands=(), pop_amp=0.0)
    noise_on = drums.render_drum(d) - drums.render_drum(dry_on)
    noise_off = drums.render_drum(replace(d, pop_amp=0.0)) - drums.render_drum(dry_off)
    assert np.allclose(noise_on, noise_off, atol=1e-9)


def test_measure_pop_recovers_a_two_component_hit():
    import fit
    sr, f11 = 48000, 480.0
    t = np.arange(int(0.05 * sr)) / sr
    y = 2.0 * np.exp(-200.0 * t) * np.sin(2 * np.pi * 0.72 * f11 * t) + np.exp(-15.0 * t) * np.sin(2 * np.pi * f11 * t)
    amp, decay, ratio, err = fit.measure_pop(y, sr, f11)
    assert np.isclose(amp, 2.0, rtol=0.1) and np.isclose(decay, 200.0, rtol=0.1)
    assert np.isclose(ratio, 0.72, rtol=0.02) and err < 0.05


def test_room_off_is_dry_and_on_has_the_set_energy_and_decay():
    import drums
    from dataclasses import replace
    d = replace(drums.reference_tenors()[0], noise_bands=())
    d = replace(d, batter=replace(d.batter, duration=2.0))
    dry = drums.render_drum(d)
    assert np.array_equal(drums.render_drum(replace(d, room_db=float("-inf"))), dry)
    wet = drums.render_drum(replace(d, room_db=-10.0, room_rt60=0.6)) - dry
    assert np.isclose(10 * np.log10(np.sum(wet ** 2) / np.sum(dry ** 2)), -10.0, atol=0.01)
    # tail: energy 0.9-1.2 s vs 0.3-0.6 s should have fallen by 60 dB * 0.6 s / rt60 = 60 dB
    sr = d.batter.sr
    late = np.sum(wet[int(0.9 * sr):int(1.2 * sr)] ** 2)
    early = np.sum(wet[int(0.3 * sr):int(0.6 * sr)] ** 2)
    assert late < early                                   # decays
    wet_only = drums.add_room(np.r_[1.0, np.zeros(2 * sr - 1)], sr, 0.6, 0.0) - np.r_[1.0, np.zeros(2 * sr - 1)]
    e = [np.sum(wet_only[int(a * sr):int((a + 0.1) * sr)] ** 2) for a in (0.1, 0.4)]
    assert np.isclose(10 * np.log10(e[1] / e[0]), -60.0 * 0.3 / 0.6, atol=2.0)   # rt60 slope
