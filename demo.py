"""Render demo WAVs + spectrum PNGs into out/.

    python demo.py

Tenor pitch, K and (0,1) offset come from the first reference (drums.REFERENCE_TENORS).
Everything else, including bass pitches, is a PLACEHOLDER until references are supplied. To fit: put paths in REFERENCES below; each drum is then
fitted (fit.py), its report printed, and the fitted preset used for rendering.
"""
import os
from dataclasses import replace

import numpy as np

import drums
import engine
import fit
from drum_modal import plot_spectrum, spectrum, write_wav

OUT = "out"
BASS_PITCHES_HZ = [98.0, 82.0, 73.0, 65.0, 55.0]  # placeholder, drum 1..5

# preset name -> path of a dry, close-mic'd single hit. Snare modal parameters should be
# fitted from a snares-OFF hit; add "<name>_wires" for a snares-ON hit to fit the wires.
REFERENCES = {
    # "snare_center": "refs/snare_center_off.wav",
    # "snare_center_wires": "refs/snare_center_on.wav",
    # "tenor_370Hz": "refs/tenor1.wav",
    # "bass_98Hz": "refs/bass1.wav",
}


# Hybrid tenors: sampled attack + modelled ring. The attacks are cut from a recording that
# is NOT in the repo (refs/ is git-ignored). Put the mono 48 kHz WAV at TENOR_SOURCE; the
# attacks are then extracted once into TENOR_ATTACKS. Without either file the tenors render
# model-only. TENOR_CUTS: (start s, length s) of one clean hit per drum of
# drums.REFERENCE_TENORS, highest drum first. These cuts are for the TAMA Fieldstar tenor
# tuning tutorial audio; attacks cut from someone else's recording are for private testing.
TENOR_SOURCE = "refs/tenor_tutorial.wav"
TENOR_ATTACKS = "refs/tenor_attacks.npz"
TENOR_CUTS = [(875.84, 0.58), (874.84, 0.69), (720.43, 0.51), (870.62, 0.48), (669.96, 0.30)]


def hybrid_tenors():
    """Reference tenors with their sampled attacks if available, else model-only."""
    tenors = drums.reference_tenors()
    if not os.path.exists(TENOR_ATTACKS) and os.path.exists(TENOR_SOURCE):
        y, sr = fit.load_mono(TENOR_SOURCE)
        assert sr == tenors[0].batter.sr, "resample the source to %d Hz" % tenors[0].batter.sr
        attacks = []
        for (start, length), d in zip(TENOR_CUTS, tenors):
            a = fit.extract_attack(y[int((start - 0.05) * sr):int((start + length) * sr)], sr,
                                   drums.TENOR_RUNG_DECAY, fmin=150, fmax=6000, floor_db=-50)
            print("%s: attack cut at %.2f s, rungs explain %.0f%% of the ring, attack %.1f dB re hit"
                  % (d.name, start, 100 * a["explained"], a["attack_db"]))
            attacks.append(a)
        drums.save_attacks(TENOR_ATTACKS, attacks)
    if not os.path.exists(TENOR_ATTACKS):
        print("no %s: tenors rendered model-only" % TENOR_ATTACKS)
        return tenors
    return [drums.with_attack(d, a) for d, a in zip(tenors, drums.load_attacks(TENOR_ATTACKS))]


def build_tenor_presets() -> None:
    """Measure the tenor presets from the source recording and write presets/*.json:
    per drum its pitch, K, (0,1) offset, the attack as damped tones and the rung phases.
    Only numbers are written, no audio. Needs TENOR_SOURCE."""
    y, sr = fit.load_mono(TENOR_SOURCE)
    rows = []
    for (start, length), (pitch, K, scale) in zip(TENOR_CUTS, drums.REFERENCE_TENORS):
        a = fit.extract_attack(y[int((start - 0.05) * sr):int((start + length) * sr)], sr,
                               drums.TENOR_RUNG_DECAY, lead_s=0.0015, fmin=150, fmax=6000, floor_db=-50)
        tones, explained = fit.attack_tones(a, sr)
        print("tenor %g Hz: %d attack tones rebuild %.0f%% of the sliced attack" % (pitch, len(tones), 100 * explained))
        rows.append(dict(name="tenor_%gHz" % pitch, pitch_hz=pitch, K=K, f01_scale=scale,
                         attack_tones=[list(t) for t in tones], rung_phase=list(a["rung_phase"])))
    engine.save_presets(engine.TENOR_PRESETS, "TAMA Fieldstar tenor tuning tutorial audio, 84 kbps MP3", rows)


def fitted(d: drums.DrumPreset) -> drums.DrumPreset:
    """Fit d to its reference if one is listed, else return d unchanged."""
    if d.name not in REFERENCES:
        return d
    y, sr = fit.load_mono(REFERENCES[d.name])
    res = fit.fit_drum(y, sr, replace(d, wires=False))
    if d.name + "_wires" in REFERENCES:
        res = fit.fit_wires(*fit.load_mono(REFERENCES[d.name + "_wires"]), res)
    print(fit.report(res), "\n")
    return replace(res.preset, wires=d.wires)


def save(d: drums.DrumPreset, name: str) -> np.ndarray:
    y = drums.render_drum(d)
    write_wav("%s/%s.wav" % (OUT, name), y, d.batter.sr)
    plot_spectrum("%s/%s.png" % (OUT, name), y, drums.drum_tables(d)[0], d.batter, name)
    return y / np.max(np.abs(y))


def compare_K(d: drums.DrumPreset, name: str) -> None:
    """Overlay spectra of the same drum with K=0 and with its (fitted) K."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    k0 = replace(d, batter=replace(d.batter, K=0.0),
                 reso=None if d.reso is None else replace(d.reso, K=0.0))
    fig, ax = plt.subplots(figsize=(10, 4))
    for preset, label in ((k0, "K=0"), (d, "K=%.2f" % d.batter.K)):
        y = drums.render_drum(preset)
        write_wav("%s/%s_%s.wav" % (OUT, name, "K0" if preset is k0 else "K"), y, d.batter.sr)
        f, mag = spectrum(y, d.batter.sr)
        ax.plot(f, 20 * np.log10(mag / mag.max() + 1e-12), lw=0.8, label=label)
    ax.set(xlim=(0, 8 * d.batter.f0), ylim=(-80, 5), xlabel="Hz", ylabel="dB",
           title="%s: air loading K=0 vs K=%.2f (same f0)" % (name, d.batter.K))
    ax.legend()
    fig.tight_layout()
    fig.savefig("%s/%s_K_compare.png" % (OUT, name), dpi=120)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)

    for strike in ("center", "edge", "rimshot"):
        for wires in (False, True):
            d = fitted(drums.snare(strike, wires=wires))
            save(d, "snare_%s_%s" % (strike, "wires" if wires else "nowires"))

    # tenor presets on the generative engine (modelled attack, no audio needed)
    if os.path.exists(TENOR_SOURCE) and not os.path.exists(engine.TENOR_PRESETS):
        build_tenor_presets()
    tenors = engine.preset_drums() or hybrid_tenors()
    sweep = [save(d, "tenor_%d" % (i + 1)) for i, d in enumerate(tenors)]
    write_wav(OUT + "/tenor_sweep.wav", np.concatenate(sweep), tenors[0].batter.sr)

    # generated drums: knobs + rules only, no recording of these drums exists
    gen = [save(engine.generate(engine.DrumSpec(name="gen_%gHz" % p, pitch_hz=p)), "generated_%gHz" % p)
           for p in (700.0, 600.0, 500.0, 400.0, 300.0, 220.0, 160.0, 110.0)]
    write_wav(OUT + "/generated_pitch_sweep.wav", np.concatenate(gen), 48000)
    # strike position, rim to centre, on one preset
    pos = [save(d, "strike_%.2f" % d.batter.r_s) for d in
           (engine.preset_drums(strike_r=r)[1] for r in (0.9, 0.75, 0.65, 0.5, 0.35, 0.2, 0.05))] \
        if engine.load_presets() else []
    if pos:
        write_wav(OUT + "/strike_position_sweep.wav", np.concatenate(pos), 48000)
    # ring time on one generated drum
    ring = [save(engine.generate(engine.DrumSpec(name="ring", pitch_hz=330.0, ring=r, duration=1.5)), "ring_x%g" % r)
            for r in (0.4, 1.0, 2.5)]
    write_wav(OUT + "/ring_time_sweep.wav", np.concatenate(ring), 48000)

    basses = [fitted(drums.bass(p)) for p in BASS_PITCHES_HZ]
    hits = [save(d, "bass_%d" % (i + 1)) for i, d in enumerate(basses)]
    write_wav(OUT + "/bass_set.wav", np.concatenate(hits), basses[0].batter.sr)

    compare_K(basses[0], "bass_1")
    compare_K(basses[-1], "bass_5")
    compare_K(tenors[0], "tenor_1")


if __name__ == "__main__":
    main()
