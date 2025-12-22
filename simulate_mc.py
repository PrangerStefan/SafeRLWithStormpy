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

from util.model_builder import build_mc_with_valuations, create_simulator
from util.cli_args import build_simulate_mc_parser




def simulate_mc(prism_path: str, episodes: int, max_steps: int, seed: int):
    bm = build_mc_with_valuations(prism_path)
    model = bm.model
    simulator = create_simulator(model, seed=seed)
    labeling = model.labeling

    for ep in range(episodes):
        print(f"\n=== Episode {ep+1}/{episodes} ===")
        state, reward_vec, _ = simulator.restart()

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

            next_state, reward_vec, _ = simulator.step()
            state = next_state


def main():
    parser = build_simulate_mc_parser()
    args = parser.parse_args()
    random.seed(args.seed)
    simulate_mc(args.prism_path, args.episodes, args.max_steps, args.seed)

if __name__ == "__main__":
    main()

