from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import stormpy
import stormpy.simulator


@dataclass
class BuiltModel:
    """Convenience wrapper for a PRISM program and its Storm model."""
    prism_program: object
    model: object


def _build_with_options(
    prism_path: str,
    build_choice_labels: bool = False,
) -> BuiltModel:
    """
    Internal helper: parse PRISM file and build sparse model with
    - state valuations
    - all labels
    - all reward models
    - optional choice labels
    """
    prism_program = stormpy.parse_prism_program(prism_path)

    options = stormpy.BuilderOptions()
    options.set_build_state_valuations(True)
    options.set_build_all_labels(True)
    options.set_build_all_reward_models(True)
    if build_choice_labels:
        options.set_build_choice_labels(True)

    model = stormpy.build_sparse_model_with_options(prism_program, options)
    return BuiltModel(prism_program=prism_program, model=model)


def build_mc_with_valuations(prism_path: str) -> BuiltModel:
    """
    Build a Markov chain  with valuations, labels, rewards.
    Used e.g. for die.prism.
    """
    bm = _build_with_options(prism_path, build_choice_labels=False)
    print(type(bm))
    return bm


def build_mdp_with_valuations(prism_path: str) -> BuiltModel:
    """
    Build an MDP with valuations, labels, rewards (no choice labels).
    """
    bm = _build_with_options(prism_path, build_choice_labels=True)
    return bm


def build_mdp_with_valuations_and_choice_labels(prism_path: str) -> BuiltModel:
    """
    Build an MDP with:
      - state valuations
      - all labels
      - all reward models
      - choice labels (for PRISM action labels [E],[W],...)
    This is what the maze training / simulate_mdp_maze use.
    """
    bm = _build_with_options(prism_path, build_choice_labels=True)
    return bm


def create_simulator(model, seed: int = 0):
    """
    Uniform helper to create a Storm simulator.
    """
    return stormpy.simulator.create_simulator(model, seed=seed)
