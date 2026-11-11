BUDGET MODE: I have limited usage.
- First, save this entire prompt verbatim as SPEC.md in the project root.
- This session: do Layers 1 and 2 ONLY, then stop.
- Keep code minimal, no extra features or refactors beyond the spec.
- Don't rerun tests repeatedly; run pytest once per layer and show the result.
- If a test fails twice, stop and give me the single most likely cause instead of retrying.
- Before stopping, write PROGRESS.md: what's done, test results, known issues, exact next step.
- Future sessions will start with "Read PROGRESS.md and SPEC.md, do Layer X." Keep the code readable enough that this works.

==================================================

Build a validated modal synthesis prototype for MARCHING PERCUSSION in Python (numpy, scipy, matplotlib, soundfile). This is a test-driven physics build. Do not tune by ear or invent parameters; each layer must pass its checks before the next one is added.

GOAL
Physically modeled marching battery: snare, tenors (quads/quints), and pitched bass drums, all on one shared modal core using Bessel functions. Each drum gets a preset fit to my reference recordings. This will later be ported to C++/JUCE, so keep the core as pure functions that output a mode table (freq, amp, decay per mode). Use a params dataclass so I drive every knob myself.

LAYER 1: Ideal membrane mode table
- Modes (m, n): m = 0..M nodal diameters, n = 1..N nodal circles. alpha_mn = scipy.special.jn_zeros(m, N)
- f_mn = f0 * alpha_mn / alpha_01
- Strike at radius r_s, pickup at r_p (0..1 of radius, same angle):
  amp_mn = J_m(alpha_mn*r_s) * J_m(alpha_mn*r_p) / norm_mn
  norm_mn = pi*J_{m+1}(alpha_mn)^2 if m==0 else (pi/2)*J_{m+1}(alpha_mn)^2
TESTS:
- Sorted ratios start 1.000, 1.594, 2.136, 2.296, 2.653, 2.918 (tol 0.001)
- Center strike (r_s=0) excites only m=0 modes
- Strike on a nodal circle gives ~0 amp for that mode

LAYER 2: Render + spectral check
- Sum of exponentially decaying sines, 48 kHz, decay rate = d0 + d1*f
- Write WAV, plot spectrum with predicted mode lines overlaid
TEST: FFT peaks of the top 8 modes match the table within 1 FFT bin

LAYER 3: Shared physics knobs, one at a time, each with a test
- Tension: scales f0 only. TEST: ratios unchanged
- Air loading: f_mn = f_ideal / sqrt(1 + K/alpha_mn). TEST: K>0 lowers low modes more and ratios stretch upward; K=0 matches Layer 1
- Mallet/stick: raised-cosine force pulse with contact duration as a knob, plus rolloff exp(-(f/fc)^2) on amps. TEST: longer contact means less high-frequency energy
- Pitch drop: f(t) = f*(1 + drop*exp(-t/tau_d)). TEST: drop=0 matches Layer 2
- Doublets (optional, skip in budget mode): split m>0 modes by split_hz into two partials. TEST: beating period = 1/split_hz

LAYER 4: Two-head coupling (shared by snare and bass)
- Batter head + resonant head coupled through the enclosed air: (0,n) modes split into in-phase and out-of-phase pairs, coupling strength as a knob
TEST: coupling=0 gives two independent mode tables

LAYER 5: Drum types

MARCHING SNARE
- Very high tension, Kevlar batter: high f0, short decays, strong upper modes (the "crack")
- Two heads, using Layer 4
- Snare wires: nonlinear, NOT modal. Model as filtered noise gated by bottom-head displacement crossing a threshold (wires lift and slap). Knobs: wire tension (threshold) and wire brightness (noise filter). TEST: wires off = pure modal output
- Strike types: center, edge, rimshot (add a short rim/shell click transient; rimshot optional in budget mode). TEST: edge strike excites more m>0 energy than center, verified from the amp table

TENORS
- Single head, open-bottom cut shells, no resonant head
- Shell acts as an open tube: add air column resonance f = n*c/(2*L_eff). TEST: column modes appear at the predicted frequencies
- Each drum tuned to a pitch I specify: solve for f0 per drum

BASS DRUMS
- Two heads, large diameter, lower tension, tuned as a pitched set of 5 (pitches I specify)
- Air loading K matters most here, since low modes get pulled down hardest
- Soft wide mallet: longer contact duration and lower fc

LAYER 6: Reference fitting
- For each drum, load my reference WAVs (dry, close-mic'd single hits per drum and stroke type; I will supply paths)
- Extract spectral peaks and decay rates; fit f0, K, d0, d1, coupling (and wire params for snare) by least squares
- Report a predicted vs measured peak table and error per drum. Fit only to the reference, never to taste

DELIVERABLES
- drum_modal.py: core functions, no globals
- drums.py: snare, tenor and bass presets built on the core
- test_drum_modal.py: pytest covering every test above
- demo.py rendering WAVs + spectrum PNGs: snare center/edge/rimshot with and without wires, a tenor sweep across 4 drums, the bass drum set, and K=0 vs fitted K comparisons

RULES
- Run pytest and show results after each layer
- If a test fails, stop and report. Do not loosen tolerances to make it pass
- Explain any physics assumption you add that isn't in this spec before using it

References: Rossing, "Science of Percussion Instruments" ch. 3; Christian et al., JASA 76(5) 1984 (air loading); Julius O. Smith, "Physical Audio Signal Processing" (CCRMA).
