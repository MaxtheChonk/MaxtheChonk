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
- `demo.py`: all SPEC deliverables render into `out/` (gitignored). Add paths to `REFERENCES`
  in demo.py to fit and print reports.

## Test results
Last run `python -m pytest -q` -> 22 passed
- L1 (3), L2 (1), L3 (4), L4 (2), L5 (6), L6 (3), screening (2), slicer (1)
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

## Exact next step
BLOCKED on audio. First reference chosen by the user: YouTube "Roger Carter Fieldstar Tenor
Tuning Tutorial" (TAMA - Bergerault USA, 1041 s), https://youtu.be/BpVNv8RxO0o
Cloud environment "MaxtheChonk" allows www.youtube.com, but the audio is served from
per-request *.googlevideo.com hosts and only exact hosts can be allowed from a session.
Unblock by either: (a) user adds `*.googlevideo.com` under Allowed domains in the
environment settings, or (b) user supplies the audio file. Keep audio OUT of the repo.
Then:
1. Convert to mono WAV, `fit.slice_hits`, `fit.screen` each slice, group slices by lowest peak
   (= drum pitch), list the kept slices with timestamps for the user to confirm which drum.
2. Fit one tenor first (`fit.fit_drum` with a `drums.tenor(pitch, L_eff)` template). Needs
   shell length L_eff from the user or the drum's spec sheet.
3. Write only trusted fitted numbers into drums.py.
Also pending (user has not said yes): tunable presets = store the air spring as an absolute
value so coupling survives retuning, `retune(preset, pitch)`, named tuning sets.
