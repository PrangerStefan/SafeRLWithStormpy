#!/usr/bin/env python3
"""
simulate_mdp.py

Simple simulator for PRISM MDPs (e.g. maze models) using stormpy.

- Builds the model with:
    - state valuations (for x,y or other variables)
    - choice labels (to see PRISM action labels: [E],[W],[N],[S],...)
- Runs a given number of episodes.
- At each step, prints:
    step, state id, (x,y) if present, labels, reward, and available actions.

Supports:
    --policy random      (default, choose actions at random)
    --policy interactive (ask user to choose action index each step)

Usage:

    python simulate_mdp.py --prism-path simple_maze.prism --episodes 3 --max-steps 50
    python simulate_mdp.py --prism-path simple_maze_sticky_slippery.prism --policy interactive
"""

from __future__ import annotations

import argparse
import random
from typing import List, Optional, Tuple

import stormpy
import stormpy.simulator

import re


def build_mdp_with_valuations_and_choice_labels(prism_path: str):
    prism_program = stormpy.parse_prism_program(prism_path)

    options = stormpy.BuilderOptions()
    options.set_build_state_valuations(True)
    options.set_build_choice_labels(True)
    options.set_build_all_reward_models(True)
    options.set_build_all_labels(True)

    model = stormpy.build_sparse_model_with_options(prism_program, options)

    print("Model type:", model.model_type)
    print("Number of states:", model.nr_states)
    print("Reward models:", list(model.reward_models.keys()))
    print("Labels:", sorted(model.labeling.get_labels()))

    return prism_program, model


def find_xy_vars(prism_program) -> Tuple[Optional[object], Optional[object]]:
    """Try to find variables named 'x' and 'y' in the PRISM program."""
    x_var = None
    y_var = None

    for var in prism_program.variables:
        if var.name == "x":
            x_var = var
        elif var.name == "y":
            y_var = var
    return x_var, y_var


def get_state_info(prism_program, model, state_id: int) -> str:
    """Return a human-readable description of the state valuation."""
    vals = model.state_valuations
    v = vals.get_valuation(state_id)

    parts: List[str] = []
    for var in prism_program.variables:
        try:
            val = v.get_value(var)
        except Exception:
            val = v[var]
        parts.append(f"{var.name}={val}")
    return ", ".join(parts)


def get_state_valuation(prism_program, model, state_id: int) -> Optional[Tuple[int, int]]:
    vals = model.state_valuations
    v = vals.get_string(state_id)

    ints = dict(re.findall(r'([a-zA-Z][_a-zA-Z0-9]*)=(-?[a-zA-Z0-9]+)', v))
    booleans = re.findall(r'(\!?)([a-zA-Z][_a-zA-Z0-9]*)[\s\t]+', v)
    booleans = {b[1]: False if b[0] == "!" else True for b in booleans}


    return (ints, booleans)


def get_choice_label(model, state_id: int, action_idx: int) -> Optional[str]:
    """
    Return the PRISM action label of local action index `action_idx` in `state_id`,
    or None if unavailable.
    """
    if not hasattr(model, "choice_labeling") or model.choice_labeling is None:
        return None

    try:
        choice_index = model.get_choice_index(state_id, action_idx)
    except AttributeError:
        return None
    labels = model.choice_labeling.get_labels_of_choice(choice_index)
    if not labels:
        return None

    # Usually a singleton set, e.g. {'E'}
    return next(iter(labels))


def simulate_mdp(
    prism_path: str,
    episodes: int,
    max_steps: int,
    seed: int,
    policy: str = "random",
):
    prism_program, model = build_mdp_with_valuations_and_choice_labels(prism_path)
    simulator = stormpy.simulator.create_simulator(model, seed=seed)

    labeling = model.labeling

    for ep in range(episodes):
        print(f"\n=== Episode {ep+1}/{episodes} ===")
        state, reward_vec, labels = simulator.restart()

        for t in range(max_steps):
            ints, booleans = get_state_valuation(prism_program, model, state)

            label_names = labeling.get_labels_of_state(state)

            print(
                f"step {t:2d}: state={state:3d}: [{ints}, {booleans}] "
                f"labels={sorted(label_names)}, "
                f"reward={list(reward_vec) if reward_vec else []}"
            )

            if simulator.is_done():
                print("  -> simulator is done (absorbing state).")
                break

            available = list(simulator.available_actions())
            num_actions = len(available)
            if num_actions == 0:
                print("  -> no available actions (deadlock).")
                break

            # Show available actions with their labels
            print("  available actions:")
            for a_idx, action in enumerate(available):
                label = get_choice_label(model, state, a_idx)
                if label is None:
                    label = "?"
                print(f"    {a_idx}: label={label}, action={action}")

            # Choose action according to policy
            if policy == "interactive":
                while True:
                    user_input = input(f"  choose action [0..{num_actions-1}] (empty=random, 'q' to quit): ").strip()
                    if user_input == "q":
                        print("  -> quitting simulation.")
                        return
                    if user_input == "":
                        a_idx = random.randrange(num_actions)
                        break
                    try:
                        a_idx = int(user_input)
                        if 0 <= a_idx < num_actions:
                            break
                    except ValueError:
                        pass
                    print("    invalid choice, try again.")
            else:
                # random policy
                a_idx = random.randrange(num_actions)
                print(f"  -> random policy chose action index {a_idx}")

            action = available[a_idx]
            next_state, reward_vec, labels = simulator.step(action)
            state = next_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prism-path",
        type=str,
        required=True,
        help="Path to PRISM MDP model (e.g. simple_maze.prism).",
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
        default=50,
        help="Maximum number of steps per episode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for simulator.",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="random",
        choices=["random", "interactive"],
        help="Action selection policy.",
    )

    args = parser.parse_args()
    random.seed(args.seed)

    simulate_mdp(
        prism_path=args.prism_path,
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        policy=args.policy,
    )


if __name__ == "__main__":
    main()
