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
- Tenor model changes after the first real reference (user-approved, not in SPEC):
  1. Tenors tune by the (1,1) mode: `DrumPreset.pitch_mode`, `f0_for_pitch(pitch, K, mode)`.
     Bass and snare still use (0,1).
  2. `DrumParams.f01_scale`: multiplies the (0,1) frequency only (1 = on the K curve).
  3. `fit_drum` on a `pitch_mode == (1,1)` template uses `fit_ladder` (anchor + K from the
     (m,1) rungs) then `find_f01`. Other drums use the old grid + least-squares path.
  `drums.REFERENCE_TENORS` / `reference_tenors()` hold the fitted pitch, K, f01_scale of five
  drums. Decay, mallet, radii, column are still placeholders there.
- Measured tenor profile (user-approved after A/B/C listening, departs from SPEC amps/decay):
  `DrumPreset.rung_amp_db` / `rung_decay`: when set, the drum is only the (m,1) ladder,
  frequencies from the physics, amplitude and decay from reference-measured tables.
  `DrumPreset.noise_bands` / `noise_edges` / `noise_db`: band x time noise map for the stick
  attack and the dense 1.2-9.6 kHz content (`drums.noise_signal`, scaled causally from
  `modal_energy`). Measured by `fit.tenor_profile` (uses `measure_rungs`, `measure_noise`).
  Constants: `drums.TENOR_RUNG_AMP_DB`, `TENOR_RUNG_DECAY`, `TENOR_NOISE_*`; used by
  `reference_tenors()`. `ladder` anchor bound tightened to +-0.5% in `fit_ladder`.
- `demo.py`: all SPEC deliverables render into `out/` (gitignored). Add paths to `REFERENCES`
  in demo.py to fit and print reports.

## Test results
Last run `python -m pytest -q` -> 27 passed, 2 FAILED (both new this session, failed twice,
stopped per the "fails twice" rule; tolerances NOT loosened, tests NOT skipped)
- FAIL test_noise_off_is_pure_modal_and_on_matches_its_map: noise energy in the 4.8-9.6 kHz
  band is 3-7.5 dB above its map in the late time slices (other cells within 2.5 dB).
  Most likely cause: crosstalk. The 4th-order band-pass skirts let the 2.4-4.8 kHz noise,
  ~15 dB louder there, leak into the band above. `measure_noise` on real audio has the same
  leak, so the stored map already contains it and the synthesis adds it twice.
  Likely fix: steeper band filters (order 8+), or solve the small band-leakage matrix.
- FAIL test_tenor_profile_round_trip: fitted (1,1) of the 590 Hz drum is 587.8 Hz (0.37% off,
  tolerance 0.2%). Most likely cause: the weak upper rungs (-27..-35 dB) are now buried under
  the 1.2-2.4 kHz noise, get matched to noise peaks, and pull the joint (f11, K) fit.
  Likely fix: fit the ladder only on rungs that stand clear of the local noise level, or
  take f11 straight from the anchor peak and fit K alone.
- Passing: L1 (3), L2 (1), L3 (4), L4 (2), L5 (6), L6 (3), screening (2), slicer (1),
  retune (1), tenor pitch mode / (0,1) offset / ladder (3), rung profile (1).

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
- With the ladder fitter (`fit_ladder`) the same hits give, per drum:
  590 Hz: K 2.9-3.5 (3 hits), (0,1) at 0.74-0.80 of its K-curve frequency
  522 Hz: K 2.1-2.2 (2 hits), 0.72      487 Hz: K 1.7 (1 hit), 0.76
  440 Hz: K 2.1 (2 hits), 0.77          390 Hz: K 1.8 (1 hit), 0.81
  Ladder rms 2-17 cents. Two hits stay unexplained: 876.44 s (peaks 520/756, maybe a sixth
  drum at ~757 Hz) and 1019.48 s (strongest 271 Hz, no ladder).
- CAVEAT: the same peak ratios (1 : 1.43 : 1.82 : 2.21 : 2.57) also equal the K=0 series
  (1,1),(0,2),(1,2),(2,2),(3,2). The air-loaded (m,1) reading was chosen because an edge
  strike should excite (2,1),(3,1) strongly and no peaks sit where K=0 would put them, but
  frequencies alone cannot rule the other reading out. Same ambiguity makes `fit_ladder`
  unreliable on drums whose true K is near 0 (synthetic K=0.5 was fitted as K=4.4).
- Synthetic recovery with the ladder fitter: pitch within 0.3%, f01_scale within 0.01,
  K biased about 8% low (rung 5 merges with a neighbouring mode in the placeholder presets).
- (0,1) identification is low confidence: neighbouring drums ring in the same range.
- Generic (non-ladder) `fit_drum` on these hits was unusable: 15-60 cents, inconsistent K.

## Why the fitted tenors do not sound like tenors yet (measured, same 8 hits)
User verdict on `tenor_sweep.wav`: right notes, not a tenor. Measured against the model:
- Ladder rung levels, median dB for m = 0..7: -11, 0, -7, -10, -26, -18, -24, -21.
  Ladder rungs carry 58-83% of the energy (32-40% on the 590 Hz drum). Peaks that are not
  rungs sit around -34 dB. The model (point strike x point pickup, SPEC Layer 1 amps) makes
  (m, n>=2) modes far too loud between 1 and 4 kHz.
- Fitting r_s, r_p to the levels: 17 dB rms (placeholder) -> 14.5 dB, flat minimum. Adding a
  per-nodal-circle suppression: best 11.4 dB at 10 dB/circle. Neither is a usable fit: the
  SPEC amplitude formula does not describe these recordings.
- Ladder rung decay, median 1/s for m = 0..7: 13, 15, 9, 7, 5.5, 8, 6, 5. Low modes decay
  FASTER than high rungs, opposite to d0 + d1*f with d1 > 0. Low confidence (tails cut).
- Energy above 4 kHz: -22 dB of the total, decaying at ~28 1/s. The model has about -50 dB
  there (M=8, N=4 stops near 3 kHz and has no attack component).
- User verdict on A/B/C/D: C (measured rungs + attack) closest, "not close enough".
- Remaining gap measured (real minus model with the first attack version, median dB):
  1.2-2.4 kHz +6..10, 2.4-4.8 kHz +15..29 (sustained), above 9.6 kHz model too loud; and the
  real envelope spikes in the first 3 ms then drops ~10 dB, the model started flat.
  The band x time noise map was built to cover both. NOT yet re-measured or heard.
- Rung amplitudes are now corrected for decay (FFT peak of a truncated decaying sine is
  a*(1-exp(-dT))/(2d)): m = 0..7 -> -9.5, 0, -10.7, -20.4, -31.8, -26.6, -34.7, -26.2 dB.
  The "C" file the user liked used uncorrected peak levels, so upper rungs are now quieter.

## Exact next step
1. Fix the two failing tests (causes above), one pytest run.
2. Re-run the real-vs-model band/time comparison (scratch `gap.py` logic: band x time energy
   of aligned real hits vs `reference_tenors()` renders) and get the user's ears on the new
   `out/tenor_sweep.wav` against the real hits.
3. Still unmodelled: sympathetic ringing of neighbouring drums (195-271 Hz cluster), shell
   air column (length unknown), per-drum profiles (one shared profile for all five now).
Tenors also need cleaner audio for decay (isolated hits that ring out > 1 s), shell depth for the
air column, and a hit recorded with a known strike spot to set r_s / r_p.
Then bass drums (two heads, uses the old fit path: expect the same nearest-partial ambiguity
on real audio, consider a ladder-style fit there too), then snare (needs a snares-off hit).
Still open: named tuning sets per corps style; golden-file export for the C++ port; the
wire model must be made causal for real time.
