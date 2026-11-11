# PROGRESS

## Done
- Layer 1: ideal membrane mode table (`drum_modal.mode_table`)
- Layer 2: render, WAV, spectrum plot (`render`, `spectrum`, `top_modes`, `write_wav`, `plot_spectrum`)
- Layer 3: tension, air loading K, mallet (contact_s, fc), pitch drop (drop, tau_d).
  Doublets skipped (budget mode, optional in SPEC).
- `demo.py` writes `out/ideal_membrane.wav` + `.png` only (out/ is gitignored); not extended for Layer 3
- `drums.py` not started (Layer 5)

## Test results
`python -m pytest -q` -> 8 passed (run once per layer, no failures, no retries)
- L1: test_l1_ratios, test_l1_center_strike_only_m0, test_l1_nodal_circle_strike
- L2: test_l2_fft_peaks_match_top8_modes
- L3: test_l3_tension_keeps_ratios, test_l3_air_loading, test_l3_longer_contact_less_hf,
  test_l3_pitch_drop_zero_matches_layer2

## Code map
- `DrumParams` dataclass: every knob. Add new knobs here with defaults that mean "off"
  so old tests keep passing.
- `mode_table(p) -> ModeTable(m, n, freq, amp, decay)`, sorted by freq. Pure.
  Applies tension, K and mallet gain (`strike_gain` = `pulse_gain` * rolloff).
- `render(table, p) -> np.ndarray`, raw un-normalized sum. Applies pitch drop.

## Assumptions added beyond SPEC (none are tuned by ear)
- Partials are sines starting at phase 0 (velocity-impulse response). Spec says "sines".
- Amps keep their sign (Bessel product can be negative). "Top 8" = largest |amp|.
- Modes above Nyquist are skipped in render.
- WAV is peak-normalized to 0.9 on write only; `render` output is raw.
- Default param values (f0=200, r_s=0.4, r_p=0.3, d0=2, d1=0.001, M=6, N=4, 2 s, tau_d=0.05)
  are placeholders for testing, not drum presets. Real values come from Layer 6 fitting.
- L2 test: no FFT window, 2 s -> bin = 0.5 Hz; peak = nearest local max to predicted freq.
- L3 tension: `tension` is relative head tension and f scales as sqrt(tension) (ideal membrane,
  f ~ sqrt(T/sigma)). SPEC only says "scales f0".
- L3 air loading: `f0` is the in-vacuo (0,1) frequency. With K>0 the sounding fundamental is
  f0/sqrt(1+K/alpha_01), lower than f0. Layer 5 "solve f0 for a pitch" must invert this.
- L3 mallet: the raised-cosine pulse is applied in the mode table as its exact Fourier
  transform, sinc(x)/(1-x^2) with x = f*contact_s, not by time-domain convolution. Signed
  (negative lobes flip a mode's phase); the constant contact_s/2 delay is dropped. Modes
  at spectral nulls of the pulse get ~0 amp, which is physical for this pulse shape.
- L3 pitch drop: one global drop/tau_d for all modes; decay still uses the undropped freq.

## Known issues
- None failing. L2 test relies on low decay (linewidth ~ bin width). With heavy decay
  (snare presets) neighbouring modes merge and peak-picking will need a longer/zero-padded
  FFT; do not loosen the 1-bin tolerance.
- Dependencies not pinned (no requirements.txt): numpy scipy matplotlib soundfile pytest.

## Exact next step
Layer 4: two-head coupling. Add a coupling knob (default 0) plus resonant-head params; build
batter and resonant mode tables, split (0,n) modes into in-phase / out-of-phase pairs.
TEST: coupling=0 gives two independent mode tables. SPEC gives no formula for the split, so
state the coupled-oscillator assumption (and get it approved) before coding it.
Run pytest once after Layer 4.
