"""Render demo WAVs + spectrum PNGs into out/. Layers 1-2 only so far."""
import os

from drum_modal import DrumParams, mode_table, plot_spectrum, render, write_wav


def main():
    os.makedirs("out", exist_ok=True)
    p = DrumParams()
    table = mode_table(p)
    y = render(table, p)
    write_wav("out/ideal_membrane.wav", y, p.sr)
    plot_spectrum("out/ideal_membrane.png", y, table, p, "Ideal membrane, f0=%g Hz" % p.f0)


if __name__ == "__main__":
    main()
