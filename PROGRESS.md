# PROGRESS

## Done
- Layer 1: ideal membrane mode table (`drum_modal.mode_table`)
- Layer 2: render, WAV, spectrum plot (`render`, `spectrum`, `top_modes`, `write_wav`, `plot_spectrum`)
- `demo.py` writes `out/ideal_membrane.wav` + `.png` (out/ is gitignored)
- `drums.py` not started (Layer 5)

## Test results
`python -m pytest -q` -> 4 passed (run once per layer, no failures, no retries)
- test_l1_ratios, test_l1_center_strike_only_m0, test_l1_nodal_circle_strike
- test_l2_fft_peaks_match_top8_modes

## Code map
- `DrumParams` dataclass: every knob. Add new knobs here with defaults that mean "off"
  (e.g. K=0, drop=0) so old tests keep passing.
- `mode_table(p) -> ModeTable(m, n, freq, amp, decay)`, sorted by freq. Pure.
- `render(table, p) -> np.ndarray`, raw un-normalized sum.

## Assumptions added beyond SPEC (none are tuned by ear)
- Partials are sines starting at phase 0 (velocity-impulse response). Spec says "sines".
- Amps keep their sign (Bessel product can be negative). "Top 8" = largest |amp|.
- Modes above Nyquist are skipped in render.
- WAV is peak-normalized to 0.9 on write only; `render` output is raw.
- Default param values (f0=200, r_s=0.4, r_p=0.3, d0=2, d1=0.001, M=6, N=4, 2 s) are
  placeholders for testing, not drum presets. Real values come from Layer 6 fitting.
- L2 test: no FFT window, 2 s -> bin = 0.5 Hz; peak = nearest local max to predicted freq.

## Known issues
- None failing. L2 test relies on low decay (linewidth ~ bin width). With heavy decay
  (snare presets) neighbouring modes merge and peak-picking will need a longer/zero-padded
  FFT; do not loosen the 1-bin tolerance.
- Dependencies not pinned (no requirements.txt): numpy scipy matplotlib soundfile pytest.

## Exact next step
Layer 3, one knob at a time, each with its test, in this order:
1. Tension: add `tension_scale` (multiplies f0). TEST ratios unchanged.
2. Air loading: `K`, f = f_ideal / sqrt(1 + K/alpha_mn). Needs alpha per mode: compute inside
   `mode_table` where `alpha` is already available. TEST K=0 == Layer 1, K>0 stretches ratios.
3. Mallet: `contact_s`, `fc`; raised-cosine pulse + exp(-(f/fc)^2) on amps.
4. Pitch drop: `drop`, `tau_d` in `render` (integrate phase). TEST drop=0 == Layer 2 output.
5. Doublets: skip (budget mode).
Run pytest once after Layer 3.
