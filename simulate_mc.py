#!/usr/bin/env python3
"""
simulate_mc.py

Simple simulator for PRISM Markov chains (e.g. die.prism) using stormpy.

- Builds the model with state valuations and labels.
- Runs a given number of episodes.
- At each step, prints:
    step, state id, state valuation, labels, reward vector.

Usage:

    python simulate_mc.py --prism-path die.prism --episodes 5 --max-steps 20
"""

from __future__ import annotations

import argparse
import random
from typing import List

import stormpy
import stormpy.simulator


def build_mc_with_valuations(prism_path: str):
    prism_program = stormpy.parse_prism_program(prism_path)

    options = stormpy.BuilderOptions()
    options.set_build_state_valuations(True)
    # if you have rewards/labels you care about:
    options.set_build_all_reward_models(True)
    options.set_build_all_labels(True)

    model = stormpy.build_sparse_model_with_options(prism_program, options)

    print("Model type:", model.model_type)
    print("Number of states:", model.nr_states)
    print("Reward models:", list(model.reward_models.keys()))
    print("Labels:", sorted(model.labeling.get_labels()))

    return prism_program, model



def simulate_mc(prism_path: str, episodes: int, max_steps: int, seed: int):
    prism_program, model = build_mc_with_valuations(prism_path)
    print(model)
    simulator = stormpy.simulator.create_simulator(model, seed=seed)

    labeling = model.labeling

    for ep in range(episodes):
        print(f"\n=== Episode {ep+1}/{episodes} ===")
        # For MCs, restart() typically returns (state, reward_vec, labels)
        state, reward_vec, labels = simulator.restart()

        for t in range(max_steps):
            label_names = labeling.get_labels_of_state(state)

            print(
                f"step {t:2d}: state={state:3d}, "
                f"labels={sorted(label_names)}, "
                f"reward={list(reward_vec) if reward_vec else []}"
            )

            if simulator.is_done():
                print("  -> simulator is done (absorbing state).")
                break

            # For Markov chains, there is no action choice: just step.
            # On many stormpy versions, `step()` works without an argument.
            next_state, reward_vec, labels = simulator.step()
            state = next_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prism-path",
        type=str,
        required=True,
        help="Path to PRISM model (e.g. die.prism).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes (restarts).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="Maximum number of steps per episode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for simulator.",
    )

    args = parser.parse_args()
    random.seed(args.seed)
    simulate_mc(args.prism_path, args.episodes, args.max_steps, args.seed)


if __name__ == "__main__":
    main()
