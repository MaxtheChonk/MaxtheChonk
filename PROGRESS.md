# PROGRESS

## Done
- Layer 1: ideal membrane mode table (`drum_modal.mode_table`)
- Layer 2: render, WAV, spectrum plot (`render`, `spectrum`, `top_modes`, `write_wav`, `plot_spectrum`)
- Layer 3: tension, air loading K, mallet (contact_s, fc), pitch drop (drop, tau_d). Doublets skipped.
- Layer 4: two-head coupling (`two_head_tables(pb, pr, coupling) -> (batter_table, reso_table)`)
- Layer 5: `drums.py` with `DrumPreset`, `snare()`, `tenor()`, `bass()`, `drum_tables`, `render_drum`
- Layer 6: `fit.py` (`extract_peaks`, `measure_decays`, `fit_drum`, `fit_wires`, `report`).
  Checked on synthetic hits only: no reference recordings supplied yet.
- Reference screening (not in SPEC, user-approved): `fit.screen(y, sr) -> Screen`. Runs inside
  `fit_drum`; `report` prints a TRUST line for pitch/K/coupling and for decay/wires.
- Hit slicer (not in SPEC): `fit.slice_hits(y, sr) -> [(start, end)]` cuts a long take into
  candidate single hits; screen each slice before fitting. Tested on a synthetic 4-hit take.
- Retuning (not in SPEC): `drums.retune(preset, pitch_hz)` moves the pitch and holds K, the
  reso/batter ratio, the air column and the absolute air spring (rescales `coupling` by 1/s^2).
  Decay is not adjusted. Named tuning sets NOT built: need real pitches per style from the user.
- `demo.py`: all SPEC deliverables render into `out/` (gitignored). Add paths to `REFERENCES`
  in demo.py to fit and print reports.

## Test results
Last run `python -m pytest -q` -> 23 passed (re-run after adding fmin/floor_db to fit_drum)
- L1 (3), L2 (1), L3 (4), L4 (2), L5 (6), L6 (3), screening (2), slicer (1), retune (1)
- History: test_l6_fit_recovers_snare_modal_and_wires failed once (wire threshold 0.098 vs
  0.15). Fixed in the fitter, tolerance unchanged: `band_map` went from 10 bands x 8 frames to
  6 bands x 40 frames, because the threshold shows up as WHEN the wire noise cuts off.

## Code map
- `DrumParams` dataclass: every knob. Add new knobs here with defaults that mean "off"
  so old tests keep passing.
- `mode_table(p) -> ModeTable(m, n, freq, amp, decay)`, sorted by freq. Pure.
  Applies tension, K and mallet gain (`strike_gain` = `pulse_gain` * rolloff).
- `render(table, p) -> np.ndarray`, raw un-normalized sum. Applies pitch drop.
- `two_head_tables(pb, pr, coupling)`: batter + resonant DrumParams -> two ModeTables
  (what each head's pickup hears). Each has 2N extra rows: every (0,n) appears twice.
  Render either with `render(table, pb)`. The reso table is what Layer 5 snare wires gate on.

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

- L4 coupling model (SPEC gives no formula): each (0,n) batter/reso pair is two oscillators
  joined by an air spring. Matrix in f^2 units [[fb^2+k, -k], [-k, fr^2+k]],
  k = coupling * fb_01 * fr_01. In-phase partial is unshifted for identical heads,
  out-of-phase is raised to sqrt(f^2 + 2k). Simplifications:
  - same k for every n (per-mode volume-displacement weighting ignored)
  - only same-n pairs couple (no (0,1)-(0,2) cross coupling)
  - equal modal mass on both heads (symmetric matrix)
  - m>0 modes uncoupled (no net volume change); reso m>0 modes never excited (amp 0)
  - strike on batter only; partial decay = energy-weighted mix of both heads' d0 + d1*f
  - `coupling` is dimensionless; its real value must come from Layer 6 fitting

- L5 pitch: "tuned to pitch" means the sounding (0,1) mode = pitch (tenor), or the lowest
  in-phase partial = pitch (bass, identical heads).
- L5 non-membrane partials live in the ModeTable with m=-1 (air column) / m=-2 (rim click).
  SPEC gives only column frequencies; their amp (`column_level`, relative to strongest
  membrane mode) and `column_decay` are knobs. Mallet gain is not applied to them.
- L5 rimshot = head strike + ONE damped sine (`rim_hz`, `rim_decay`, `rim_level`).
- L5 wires: low-passed noise * max(|reso disp|/peak - threshold, 0). Threshold is relative to
  peak resonant-head displacement (no absolute hit dynamics). Brightness = low-pass cutoff.
  Only (0,n) modes move the resonant head (Layer 4), so they alone drive the wires.
- L5 strike radii: center 0.0, edge 0.8, rimshot 0.3. Placeholders, like every preset number
  and the demo pitches / shell lengths.
- L6 fits pitch (-> f0), K, coupling to peak frequencies (cents, nearest-partial matching,
  grid search + scipy least_squares), then d0, d1 by linear least squares on STFT decay slopes.
  NOT fitted, taken from the template: r_s, r_p, M, N, mallet, reso/batter f0 ratio, column L.
  Amplitudes are not fitted. Wires fitted to a coarse band x time energy map in dB.
- L6 test tolerances were set from what the fitter achieves on synthetic hits (SPEC has no
  Layer 6 test). They are regression bounds, not physics.

- Screening thresholds (snr 40 dB, tail -40 dB, decay ratio 0.6..1.6, onset = 10 dB jump in
  30 ms above 1 kHz) are heuristics, validated only on synthetic degradations.

## Known issues
- User has only SAMPLES (no raw recordings). Expect pitch, K, coupling to be usable and
  decay / wire numbers to be flagged untrusted on processed samples.
- Screening misses reverb that is quieter or shorter than the drum's own tail (synthetic:
  caught on snare, missed on tenor and bass). It also cannot see EQ, or clipping that was
  followed by other processing.
- Snare modal fit must use a snares-OFF hit. With wires on, noise peaks pollute the peak list
  (synthetic check: ~100 cents rms error).
- Wire fit on synthetic: threshold within 0.01, brightness within ~13%, level within ~25%.
- Decay fit is biased when two partials share an STFT bin (beating). Seen on bass 65 Hz
  synthetic: d0 2.9 vs 4.0, d1 0.0122 vs 0.0100. Bass 98 Hz recovered within 1%.
- Snare center strike excites few partials: coupling recovered as 0.17 vs 0.20 (11.7 cents rms).
- L2-style 1-bin peak tests need low decay; heavy-decay presets merge neighbouring lines.
- Dependencies not pinned: numpy scipy matplotlib soundfile pytest.

## First real reference: findings (tenors)
Source: user-supplied MP3 of "Roger Carter Fieldstar Tenor Tuning Tutorial" (TAMA), 1041 s,
84 kbps, stereo. Audio is NOT in the repo (lives only in the session scratchpad; ask the user
to re-upload in a new session). Mostly speech; constant 120 Hz mains hum (use fmin=150).
Drum-only passages: ~669-671 s, 716-724 s, 750-768 s, 870-881 s, 1000-1022 s.
- `slice_hits` + `screen` alone cannot separate drum taps from speech on this file; the
  passages were found by eye on a spectrogram. Hits follow each other every 0.3-0.6 s, so
  tails are cut short: decay numbers from this file are NOT trusted.
- Six drums (a sextet). Strongest peak of each: ~390, 440, 487, 522, 590, 757 Hz.
- Every drum shows the same peak ladder from its strongest peak: 1 : 1.43 : 1.82 : 2.21 : 2.57.
  This matches the air-loaded (m,1) series, m = 1..5, anchored at (1,1):
  fitting (f11, K) to those 5 peaks gives 3-13 cents rms on 10 hits, with K = 1.7 .. 3.3
  (2 parameters, 5 peaks, K scatters between hits of the same drum).
- So on real tenors the STRONGEST / pitch-defining mode is (1,1), not (0,1). The presets and
  `f0_for_pitch` currently tune by (0,1): this is wrong for tenors and must change.
- The (0,1) mode is not explained: the model puts it at 224-432 Hz, the recordings show low
  peaks around 183-271 Hz (or nothing clear). Likely extra lowering by the air in the shell,
  which f = f_ideal/sqrt(1+K/alpha) cannot produce (it caps (1,1)/(0,1) at 2.01).
  Peaks near 195/213/245/271 Hz also appear in hits on OTHER drums (sympathetic ringing).
- Generic `fit_drum` on these hits: 15-60 cents rms and inconsistent K (0.7 vs 2.1 for the same
  drum), because nearest-partial matching in a dense mode set is ambiguous. Not usable as is.
- `fit_drum` gained `fmin` and `floor_db` arguments for real takes.

## Exact next step
Decisions needed from the user before changing the model (not in SPEC):
1. Tune tenors by the (1,1) mode instead of (0,1).
2. Give (0,1) its own frequency offset knob (fitted), instead of forcing it onto the K curve.
3. Make `fit_drum` for tenors fit the (m,1) ladder first (anchor at the strongest peak), then
   assign the remaining peaks.
After that: write fitted K and the six pitches into a tenor preset, flagged "decay untrusted".
Still open: named tuning sets, bass and snare references, cleaner tenor audio for decay.
