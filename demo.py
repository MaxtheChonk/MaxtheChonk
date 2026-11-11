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

    tenors = [fitted(d) for d in drums.reference_tenors()]   # pitch, K, (0,1) fitted; rest placeholder
    sweep = [save(d, "tenor_%d" % (i + 1)) for i, d in enumerate(tenors)]
    write_wav(OUT + "/tenor_sweep.wav", np.concatenate(sweep), tenors[0].batter.sr)

    basses = [fitted(drums.bass(p)) for p in BASS_PITCHES_HZ]
    hits = [save(d, "bass_%d" % (i + 1)) for i, d in enumerate(basses)]
    write_wav(OUT + "/bass_set.wav", np.concatenate(hits), basses[0].batter.sr)

    compare_K(basses[0], "bass_1")
    compare_K(basses[-1], "bass_5")
    compare_K(tenors[0], "tenor_1")


if __name__ == "__main__":
    main()
