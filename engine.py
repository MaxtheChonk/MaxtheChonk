"""Generative drum engine: physical knobs in, drum out. No recording needed.

    spec = DrumSpec(pitch_hz=300.0, strike_r=0.4)
    y = drums.render_drum(generate(spec))

Three layers:
  DrumSpec   the knobs a user turns
  Rules      how a real drum behaves, measured from recordings; the defaults behind the knobs
  preset     a DrumSpec plus measured extras for one specific drum (presets/*.json)

generate() returns an ordinary drums.DrumPreset, so everything downstream (mode table,
render, retune, room) is the same code for an invented drum and for a fitted preset.

What is physics and what is measured, for a generated single-head drum:
  frequencies of the ring      Bessel zeros + air loading K + (0,1) offset      physics + rule
  loudness vs strike position  membrane mode shapes J_m                          physics
  loudness of each rung        measured profile at the reference strike spot     rule
  ring time of each rung       measured profile                                  rule
  attack                       measured tones, scaled with the pitch             rule
  upper noise                  measured band x time map                          rule

The only Rules so far come from ONE tenor set (5 drums, 390-590 Hz, low-bitrate source).
Far from that range a generated drum is extrapolation, not a measured fact.
Two-head drums are not covered by the rules yet: use drums.snare() / drums.bass().
"""
import json
import os
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
from scipy.special import jn_zeros

import drums

PRESET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "presets")
TENOR_PRESETS = os.path.join(PRESET_DIR, "reference_tenors.json")


@dataclass
class Rules:
    """Defaults measured from recordings. Each field says where it comes from."""
    pitch_mode: tuple = (1, 1)            # measured: the strongest, pitch-carrying mode
    K: float = 2.15                       # measured: median air loading of the 5 tenors (1.75-3.0)
    f01_scale: float = 0.77               # measured: (0,1) sits at 0.72-0.81 of its K-curve frequency
    rung_amp_db: tuple = drums.TENOR_RUNG_AMP_DB    # measured, pooled over 8 hits
    rung_decay: tuple = drums.TENOR_RUNG_DECAY      # measured, pooled; low confidence (tails cut)
    strike_ref: float = 0.65              # ASSUMED: where the measured hits landed (normal tenor
                                          # playing zone, a few cm inside the rim). Not measured.
    circular_db: float = -34.0            # measured: median level of non-rung peaks at that spot
    circular_n: int = 3
    noise_bands: tuple = drums.TENOR_NOISE_BANDS    # measured
    noise_edges: tuple = drums.TENOR_NOISE_EDGES
    noise_db: tuple = drums.TENOR_NOISE_DB
    noise_ref_hz: float = 485.0           # median pitch of the drums the noise map came from
    attack_tones: tuple = ()              # measured tones of one representative drum
    rung_phase: tuple = ()                # start phases that go with those tones
    attack_fade: tuple = (0.020, 0.040)


@dataclass
class DrumSpec:
    """The knobs. None = use the rule."""
    name: str = "drum"
    pitch_hz: float = 440.0               # frequency of the pitch mode
    K: Optional[float] = None             # air loading
    f01_scale: Optional[float] = None     # (0,1) offset
    strike_r: Optional[float] = None      # strike position, 0 = centre .. 1 = rim
    ring: float = 1.0                     # ring time multiplier (2 = rings twice as long)
    attack: float = 1.0                   # attack level multiplier (0 = no attack tones)
    noise: float = 1.0                    # noise level multiplier (0 = no noise)
    attack_tracking: float = 1.0          # 1 = attack tones move with the pitch, 0 = stay at the
                                          # frequencies they were measured at. ASSUMED 1: the
                                          # reference has no drum at two tunings to tell.
    noise_tracking: float = 0.0           # same for the noise bands. 0 = as measured.
    column_L: float = 0.0                 # m, shell air column length; 0 = off
    column_level: float = 0.0
    room_db: float = float("-inf")        # not fitted; yours to set
    room_rt60: float = 0.5
    duration: float = 1.0
    sr: int = 48000


def pitch_from_tension(diameter_m: float, tension_n_per_m: float, density_kg_per_m2: float,
                       K: float = 0.0, mode=(0, 1), f01_scale: float = 1.0) -> float:
    """Frequency of a mode of a round head from its physical size, tension and mass:
    f_mn = alpha_mn / (2*pi*a) * sqrt(T / sigma), then air loading (and the (0,1) offset).
    Lets a spec be written as diameter + tension instead of a pitch."""
    m, n = mode
    alpha = jn_zeros(m, n)[-1]
    f = alpha / (np.pi * diameter_m) * np.sqrt(tension_n_per_m / density_kg_per_m2)
    f /= np.sqrt(1 + K / alpha)
    return float(f * f01_scale if mode == (0, 1) else f)


def generate(spec: DrumSpec, rules: Optional[Rules] = None, attack_tones=None, rung_phase=None) -> drums.DrumPreset:
    """Build a single-head drum from knobs + rules. attack_tones / rung_phase override the
    rule's generic attack with one measured on this specific drum (a preset's extras)."""
    rules = rules or tenor_rules()
    K = rules.K if spec.K is None else spec.K
    scale = rules.f01_scale if spec.f01_scale is None else spec.f01_scale
    strike = rules.strike_ref if spec.strike_r is None else spec.strike_r
    tones = rules.attack_tones if attack_tones is None else attack_tones
    phases = rules.rung_phase if rung_phase is None else rung_phase

    d = drums.tenor(spec.pitch_hz, spec.column_L, K=K)
    batter = replace(d.batter, f0=drums.f0_for_pitch(spec.pitch_hz, K, rules.pitch_mode),
                     f01_scale=scale, r_s=strike, sr=spec.sr, duration=spec.duration,
                     M=max(d.batter.M, len(rules.rung_amp_db) - 1), N=max(d.batter.N, rules.circular_n + 1))

    # attack tones are stored as multiples of the pitch they were measured at; with
    # tracking < 1 they are pulled back toward those measured frequencies
    if len(tones):
        t = np.array(tones, dtype=float)
        t[:, 0] *= (rules.noise_ref_hz / spec.pitch_hz) ** (1 - spec.attack_tracking)
        tones = tuple(map(tuple, t))
    shift = (spec.pitch_hz / rules.noise_ref_hz) ** spec.noise_tracking
    bands = tuple((lo * shift, hi * shift) for lo, hi in rules.noise_bands if lo * shift < 0.45 * spec.sr)
    noise_db = tuple(tuple(v + 20 * np.log10(spec.noise) for v in row)
                     for row in rules.noise_db[:len(bands)]) if spec.noise > 0 else ()

    return replace(
        d, name=spec.name, batter=batter, pitch_mode=rules.pitch_mode,
        column_level=spec.column_level,
        rung_amp_db=rules.rung_amp_db, rung_decay=tuple(v / spec.ring for v in rules.rung_decay),
        rung_phase=tuple(phases), strike_ref=rules.strike_ref,
        circular_db=rules.circular_db, circular_n=rules.circular_n,
        attack_tones=tuple(tones), attack_gain=spec.attack, attack_fade=rules.attack_fade,
        noise_bands=bands if spec.noise > 0 else (), noise_edges=rules.noise_edges, noise_db=noise_db,
        room_db=spec.room_db, room_rt60=spec.room_rt60)


# ------------------------------------------------------------------ presets on disk

def save_presets(path: str, source: str, rows: list) -> None:
    """rows: dicts with name, pitch_hz, K, f01_scale, attack_tones, rung_phase."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"source": source, "drums": rows}, f, indent=1)


def load_presets(path: str = TENOR_PRESETS) -> list:
    """Raw preset rows (dicts) from a preset file; [] if the file does not exist."""
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)["drums"]


def tenor_rules(path: str = TENOR_PRESETS, generic: int = 1) -> Rules:
    """The tenor rules. The generic attack is the measured attack of one representative
    preset drum (index `generic`: the 522 Hz drum, whose attack the tones rebuild best).
    Without a preset file the rules have no attack tones."""
    rows = load_presets(path)
    if not rows:
        return Rules()
    r = rows[min(generic, len(rows) - 1)]
    return Rules(attack_tones=tuple(map(tuple, r["attack_tones"])), rung_phase=tuple(r["rung_phase"]))


def preset_drums(path: str = TENOR_PRESETS, **spec_kw) -> list:
    """The preset drums of a file as DrumPresets: knobs from the file, each with its own
    measured attack. spec_kw overrides knobs for all of them (e.g. strike_r=0.3)."""
    rules = tenor_rules(path)
    return [generate(DrumSpec(name=r["name"], pitch_hz=r["pitch_hz"], K=r["K"], f01_scale=r["f01_scale"], **spec_kw),
                     rules, attack_tones=r["attack_tones"], rung_phase=r["rung_phase"])
            for r in load_presets(path)]
