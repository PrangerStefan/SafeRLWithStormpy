#!/usr/bin/env python3
"""
shielded_q_learning_maze.py

Single-file script for:
    - Building a PRISM MDP maze with stormpy (with valuations and labels)
    - Computing a bounded Pmin(F<=H lava) shield manually via value iteration
    - Tabular Q-learning with epsilon-greedy over shielded actions
    - Optional curses visualization using x,y from state valuations

Assumptions on the PRISM model:
    - mdp

    - Integer variables:
        x : [..] init ...
        y : [..] init ...
      used as coordinates.

    - Label "goal" marks goal states.
    - Label "lava" marks hazard states (optional; can be absent).
    - A reward structure is NOT required; we compute reward from labels:
        reward = 1 if next_state has label "goal", else 0.

Example usage:

    # Unshielded training (no shield), no visualization
    python shielded_q_learning_maze.py \
        --prism-path simple_maze.prism \
        --episodes 200 \
        --output Q_unshielded.npy

    # Shielded training with horizon 3, no visualization
    python shielded_q_learning_maze.py \
        --prism-path simple_maze.prism \
        --horizon 3 \
        --episodes 200 \
        --output Q_shielded_h3.npy

    # Shielded training with visualization every 20 episodes
    python shielded_q_learning_maze.py \
        --prism-path simple_maze.prism \
        --horizon 3 \
        --episodes 200 \
        --visualize-every 20 \
        --output Q_shielded_h3.npy
"""

from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import dataclass
from typing import Dict, Tuple, Set, Optional, List

import numpy as np
import stormpy
import stormpy.simulator
import curses

from util.maze_drawer import MazeDrawer, MazeLayout
from util.model_builder import build_model_with_valuations


# =========================================================
#  Model building with valuations
# =========================================================


def create_simulator(model, seed: int = 0):
    """
    Create a Storm simulator instance for sampling from the MDP.
    """
    return stormpy.simulator.create_simulator(model, seed=seed)


# =========================================================
#  Maze layout and visualization (using valuations + labels)
# =========================================================



def _get_xy_values(prism_program, model) -> Tuple[Dict[int, Tuple[int, int]],
                                                 Dict[Tuple[int, int], int],
                                                 int, int]:
    """
    Use state valuations to obtain x,y for each state.

    Requires the model to be built with state valuations:
        options = stormpy.BuilderOptions(...)
        options.set_build_state_valuations(True)
        model = stormpy.build_sparse_model_with_options(prism_program, options)
    """
    x_var = None
    y_var = None
    for var in prism_program.variables:
        if var.name == "x":
            x_var = var
        elif var.name == "y":
            y_var = var

    if x_var is None or y_var is None:
        raise ValueError("Could not find variables 'x' and 'y' in the PRISM program.")

    vals = model.state_valuations

    xs = vals.get_values_states(x_var)
    ys = vals.get_values_states(y_var)

    state_to_xy: Dict[int, Tuple[int, int]] = {}
    xy_to_state: Dict[Tuple[int, int], int] = {}

    max_x = 0
    max_y = 0

    for s_id in range(model.nr_states):
        x = int(xs[s_id])
        y = int(ys[s_id])
        state_to_xy[s_id] = (x, y)
        xy_to_state[(x, y)] = s_id
        max_x = max(max_x, x)
        max_y = max(max_y, y)

    width = max_x + 1
    height = max_y + 1
    return state_to_xy, xy_to_state, width, height


def _get_labeled_states(model, label_name: str) -> Set[int]:
    """
    Return all state ids that have a given label (e.g. "goal" or "lava").

    Uses get_labels_of_state, which returns a set of label names
    per state.
    """
    labeling = model.labeling
    states: Set[int] = set()
    for s_id in range(model.nr_states):
        names = labeling.get_labels_of_state(s_id)  # set[str]
        if label_name in names:
            states.add(s_id)
    return states


def build_maze_layout(prism_program, model) -> MazeLayout:
    """
    Construct MazeLayout from valuations and labels.

    - Uses variables x,y for coordinates.
    - Uses labels "goal" and "lava" if present.
    """
    state_to_xy, xy_to_state, width, height = _get_xy_values(prism_program, model)

    goal_states = _get_labeled_states(model, "goal")
    lava_states = _get_labeled_states(model, "lava")

    print("Goal states:", goal_states)
    print("Lava states:", lava_states)

    return MazeLayout(
        width=width,
        height=height,
        state_to_xy=state_to_xy,
        xy_to_state=xy_to_state,
        goal_states=goal_states,
        lava_states=lava_states,
    )


# =========================================================
#  Shield data structure
# =========================================================

class Shield:
    """
    safe_actions_by_state[state_id] = list of *available-action indices*
    that are considered safe in that state.

    Indices are with respect to simulator.available_actions() list
    and state.actions enumeration.
    """

    def __init__(self, safe_actions_by_state: Dict[int, List[int]]):
        self.safe_actions_by_state = safe_actions_by_state

    def safe_indices(self, state_id: int, num_available: int) -> List[int]:
        """
        Return safe action indices in [0, num_available).
        If state_id not in dict → treat as "no restriction" and
        return all indices 0..num_available-1.
        """
        if state_id not in self.safe_actions_by_state:
            return list(range(num_available))
        raw = self.safe_actions_by_state[state_id]
        return [a for a in raw if 0 <= a < num_available]

    def _action_direction(
        self,
        model,
        layout: MazeLayout,
        state_id: int,
        action_idx: int,
    ) -> str:
        """
        Infer a human-readable direction for an action in the maze,
        based on the change in (x,y) for one of its successors.

        Returns one of: 'E', 'W', 'N', 'S', '•' (no move), or '?' (unknown).
        """
        state = model.states[state_id]
        actions = list(state.actions)
        if action_idx < 0 or action_idx >= len(actions):
            return "?"

        action = actions[action_idx]
        (x, y) = layout.state_to_xy[state_id]

        # Look at the first successor that exists
        succ_xy = None
        for tr in action.transitions:
            succ_xy = layout.state_to_xy[tr.column]
            break

        if succ_xy is None:
            return "?"  # no successors?

        sx, sy = succ_xy
        dx = sx - x
        dy = sy - y

        if dx == 1 and dy == 0:
            return "E"
        if dx == -1 and dy == 0:
            return "W"
        if dx == 0 and dy == -1:
            return "N"
        if dx == 0 and dy == 1:
            return "S"
        if dx == 0 and dy == 0:
            return "_"
        return "?"       # currently not covered


    def debug_allowed_actions_xy(
        self,
        model,
        layout: MazeLayout,
        only_restricted: bool = False,
    ) -> None:
        """
        Print a debug mapping (x,y) -> allowed actions.

        For each (x,y), shows both:
          - safe action indices
          - labels/directions from choice labeling
        """
        print("=== Shield debug: (x,y) -> safe action indices and labels ===")

        for y in range(layout.height):
            for x in range(layout.width):
                state_id = layout.xy_to_state.get((x, y), None)
                if state_id is None:
                    continue

                state = model.states[state_id]
                actions = list(state.actions)
                num_actions = len(actions)
                if num_actions == 0:
                    continue

                safe_idxs = self.safe_actions_by_state.get(
                    state_id, list(range(num_actions))
                )

                if only_restricted and len(safe_idxs) == num_actions:
                    continue

                # Prefer choice labels, fall back to direction inference
                labels_or_dirs = []
                for a_idx in safe_idxs:
                    label = self._choice_label(model, state_id, a_idx)
                    if label is None:
                        label = self._action_direction(model, layout, state_id, a_idx)
                    labels_or_dirs.append(label)

                print(
                    f"[state {state_id:3d}] ({x},{y}) "
                    #f"-> safe idxs {safe_idxs},
                    f"labels {labels_or_dirs}"
                )

    def _action_direction(
        self,
        model,
        layout: MazeLayout,
        state_id: int,
        action_idx: int,
    ) -> str:
        """
        Infer a human-readable direction for an action in the maze.

        Prefer PRISM action labels via choice labeling (E,W,N,S,...),
        and fall back to geometry if they are missing.

        Returns one of: 'E', 'W', 'N', 'S', '_', or '?'.
        """
        state = model.states[state_id]
        actions = list(state.actions)
        if action_idx < 0 or action_idx >= len(actions):
            return "?"

        # 1) Try to use choice labels (PRISM action labels)
        label = self._choice_label(model, state_id, action_idx)
        if label is not None:
            # Map known labels to our shorthand
            if label in ("E", "W", "N", "S"):
                return label
            if label.lower() in ("stay", "noop", "tau"):
                return "_"
            # Unknown label but still useful to print
            return label

        # 2) Fallback: infer from a successor’s (x,y) difference
        action = actions[action_idx]
        (x, y) = layout.state_to_xy[state_id]

        succ_xy = None
        for tr in action.transitions:
            succ_xy = layout.state_to_xy[tr.column]
            break

        if succ_xy is None:
            return "?"

        sx, sy = succ_xy
        dx = sx - x
        dy = sy - y

        if dx == 1 and dy == 0:
            return "E"
        if dx == -1 and dy == 0:
            return "W"
        if dx == 0 and dy == -1:
            return "N"
        if dx == 0 and dy == 1:
            return "S"
        if dx == 0 and dy == 0:
            return "_"
        return "?"

    def _choice_label(self, model, state_id: int, action_idx: int) -> str | None:
        """
        Return the PRISM action label for (state, local action index),
        e.g. 'E', 'W', 'N', 'S', or None if not available.
        """
        # If choice labeling is not present, bail out
        if not hasattr(model, "choice_labeling") or model.choice_labeling is None:
            return None

        try:
            choice_index = model.get_choice_index(state_id, action_idx)
        except AttributeError:
            # very old stormpy version without get_choice_index
            return None

        labels = model.choice_labeling.get_labels_of_choice(choice_index)
        if not labels:
            return None

        # Typically for PRISM action labels this is a singleton set
        # (e.g. {'E'}, {'W'}, ...)
        return next(iter(labels))

# =========================================================
#  Manual bounded Pmin(F<=H lava) via value iteration
# =========================================================

def compute_pmin_lava_bounded_vi(
    model,
    lava_states: Set[int],
    horizon: int,
) -> List[List[float]]:
    """
    Compute v_k(s) = minimal probability to reach a lava state within <= k steps,
    for k = 0..horizon, using dynamic programming.

    Returns: vs where vs[k][s_id] = v_k(s_id).
    """

    n = model.nr_states

    # v_0(s): 1 if s is lava, else 0
    v0 = [1.0 if s in lava_states else 0.0 for s in range(n)]
    vs: List[List[float]] = [v0]

    v_prev = v0
    for k in range(1, horizon + 1):
        v_k = [0.0] * n

        for state in model.states:
            s = state.id

            if s in lava_states:
                v_k[s] = 1.0
                continue

            actions = list(state.actions)
            if not actions:
                # no choices -> keep previous probability
                v_k[s] = v_prev[s]
                continue

            # Pmin: min over actions of expected v_{k-1}(t)
            min_val = math.inf
            for action in actions:
                prob = 0.0
                for tr in action.transitions:
                    prob += float(tr.value()) * v_prev[tr.column]
                if prob < min_val:
                    min_val = prob

            v_k[s] = min_val

        vs.append(v_k)
        v_prev = v_k

    return vs


def compute_vi_based_lava_shield(
    model,
    lava_states: Set[int],
    horizon: int,
    risk_threshold: float,
    epsilon: float = 1e-12,
) -> tuple[Shield, List[float]]:
    """
    Compute a shield using bounded Pmin value iteration:

    1. Compute vs[k][s] = Pmin (reach lava in <= k steps) for k = 0..horizon.
    2. For each state s and each action a:
         q_H(s,a) = sum_t P(s,a,t) * vs[H-1][t]

       We interpret q_H(s,a) as "risk" of hitting lava within <= H steps
       if we take action a now and behave minimally risky afterward.

       Action a is SAFE iff q_H(s,a) <= risk_threshold (+ epsilon).

    Returns (shield, v_H), where v_H is the final Pmin vector for horizon H.
    """

    if horizon <= 0:
        # horizon 0 means: no lookahead; allow everything
        print("Horizon <= 0, not computing any shield (all actions allowed).")
        return Shield({}), [0.0] * model.nr_states

    vs = compute_pmin_lava_bounded_vi(model, lava_states, horizon)
    v_H_minus_1 = vs[horizon - 1]
    v_H = vs[horizon]  # not used for the threshold, but still useful to inspect

    safe_actions_by_state: Dict[int, List[int]] = {}

    for state in model.states:
        s = state.id
        actions = list(state.actions)

        if not actions:
            safe_actions_by_state[s] = []
            continue

        if s in lava_states:
            # Already in lava: no safe actions (or allow all; we choose "none")
            safe_actions_by_state[s] = []
            continue

        q_vals = []
        for action in actions:
            prob = 0.0
            for tr in action.transitions:
                prob += float(tr.value()) * v_H_minus_1[tr.column]
            q_vals.append(prob)

        # FIXED threshold: safe if risk <= risk_threshold
        safe_idxs = [
            a_idx for a_idx, q in enumerate(q_vals)
            if q <= risk_threshold + epsilon
        ]

        safe_actions_by_state[s] = safe_idxs

    shield = Shield(safe_actions_by_state)
    return shield, v_H


# =========================================================
#  Q-learning (shielded or unshielded)
# =========================================================

def epsilon_greedy_subset(
    Q: np.ndarray,
    state: int,
    candidate_indices: List[int],
    epsilon: float,
) -> int:
    """
    Epsilon-greedy over the provided subset of action indices.
    Returns an index from candidate_indices (must be non-empty).
    """
    if not candidate_indices:
        raise RuntimeError(f"No candidate actions in state {state}.")

    if random.random() < epsilon:
        return random.choice(candidate_indices)

    sub_qs = Q[state, candidate_indices]
    best_local = int(np.argmax(sub_qs))
    return candidate_indices[best_local]


def visualize_greedy_episode(
    stdscr,
    layout: MazeLayout,
    model,
    Q: np.ndarray,
    max_steps: int = 100,
    delay: float = 0.05,
) -> bool:
    """
    Run a single greedy rollout with Q, drawing it using curses.
    Returns True if the user pressed 'q' (request to quit), False otherwise.
    """
    drawer = MazeDrawer(stdscr, layout)
    simulator = create_simulator(model, seed=1234)
    labeling = model.labeling

    state, reward_vec, labels = simulator.restart()
    total_reward = 0.0

    stdscr.nodelay(True)

    for step in range(max_steps):
        drawer.draw(
            agent_state=state,
            extra_info=f"[viz] greedy rollout, step={step}, return={total_reward:.2f} (q=quit)",
        )

        ch = stdscr.getch()
        if ch == ord('q'):
            return True

        available = list(simulator.available_actions())
        if len(available) == 0:
            break

        num_avail = len(available)
        a_idx = int(np.argmax(Q[state, :num_avail]))
        action = available[a_idx]

        next_state, reward_vec, labels = simulator.step(action)
        labels_of_next_state = labeling.get_labels_of_state(next_state)
        reward = 1.0 if "goal" in labels_of_next_state else 0.0
        total_reward += reward
        state = next_state

        if simulator.is_done():
            break

        time.sleep(delay)

    drawer.draw(
        agent_state=state,
        extra_info=f"[viz] rollout finished, total return={total_reward:.2f} (q=quit)",
    )
    time.sleep(0.5)
    return False


def q_learning(
    model,
    simulator,
    shield: Optional[Shield],
    num_episodes: int = 500,
    max_steps_per_episode: int = 200,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon_start: float = 0.3,
    epsilon_end: float = 0.01,
    stdscr=None,
    layout: Optional[MazeLayout] = None,
    visualize_every: int = 0,
    viz_delay: float = 0.05,
) -> np.ndarray:
    """
    Tabular Q-learning with optional shield and optional visualization.

    - Reward is computed from labels: 1 when entering a "goal" state, else 0.
    - If shield is None → UN-SHIELDED:
        candidate actions = all available actions
      else:
        candidate actions = shield.safe_indices(...)
    """

    num_states = model.nr_states
    max_num_actions = max(len(s.actions) for s in model.states)
    if max_num_actions == 0:
        raise RuntimeError("Model seems to have no actions.")

    Q = np.zeros((num_states, max_num_actions), dtype=float)
    labeling = model.labeling

    for episode in range(num_episodes):
        frac = episode / max(1, num_episodes - 1)
        epsilon = epsilon_start + frac * (epsilon_end - epsilon_start)

        state, reward_vec, labels = simulator.restart()
        total_reward = 0.0

        for t in range(max_steps_per_episode):
            available = list(simulator.available_actions())
            num_avail = len(available)
            if num_avail == 0:
                break

            # candidate actions depending on shield
            if shield is None:
                candidate = list(range(num_avail))
            else:
                candidate = shield.safe_indices(state_id=state, num_available=num_avail)

            if not candidate:
                # shield forbids everything → treat as terminal
                break

            a_idx = epsilon_greedy_subset(Q, state, candidate, epsilon)
            action = available[a_idx]

            next_state, reward_vec, labels = simulator.step(action)
            labels_of_next_state = labeling.get_labels_of_state(next_state)
            reward = 1.0 if "goal" in labels_of_next_state else 0.0
            total_reward += reward

            # Target
            next_available = list(simulator.available_actions())
            num_next_avail = len(next_available)

            if num_next_avail == 0 or simulator.is_done() or ("goal" in labels_of_next_state):
                target = reward
            else:
                if shield is None:
                    next_candidate = list(range(num_next_avail))
                else:
                    next_candidate = shield.safe_indices(
                        state_id=next_state,
                        num_available=num_next_avail,
                    )

                if next_candidate:
                    max_next_q = np.max(Q[next_state, next_candidate])
                else:
                    max_next_q = 0.0

                target = reward + gamma * max_next_q

            td_err = target - Q[state, a_idx]
            Q[state, a_idx] += alpha * td_err

            state = next_state
            if simulator.is_done():
                break

        mode = "UNSHIELDED" if shield is None else "SHIELDED"
        print(f"[{mode}] Episode {episode+1}/{num_episodes}, return = {total_reward:.3f}")

        # Optional visualization
        if stdscr is not None and layout is not None and visualize_every > 0:
            if (episode + 1) % visualize_every == 0:
                want_quit = visualize_greedy_episode(
                    stdscr, layout, model, Q, delay=viz_delay, max_steps=max_steps_per_episode
                )
                if want_quit:
                    print("User requested quit during visualization.")
                    return Q

    return Q


# =========================================================
#  Top-level runners
# =========================================================

def train_no_curses(args):
    prism_program, model = build_model_with_valuations(args.prism_path)
    layout = build_maze_layout(prism_program, model)
    simulator = create_simulator(model)

    # Shield computation (manual Pmin VI) if horizon > 0 and lava exists
    print(args.horizon)
    if args.horizon > 0 and layout.lava_states:
        print(f"Computing VI-based lava shield with horizon={args.horizon}")
        shield, v_H = compute_vi_based_lava_shield(
            model,
            layout.lava_states,
            horizon=args.horizon,
            risk_threshold=args.risk_threshold,
        )
        print("Example safe actions in state 0:", shield.safe_actions_by_state.get(0, []))

        layout.print_ascii()
        shield.debug_allowed_actions_xy(model, layout)
    else:
        shield = None
        if args.horizon > 0 and not layout.lava_states:
            print("Horizon > 0 but no lava states found; running unshielded.")
        else:
            print("Running UN-SHIELDED Q-learning (no horizon or no lava).")
    input("")

    Q = q_learning(
        model=model,
        simulator=simulator,
        shield=shield,
        num_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_start=args.eps_start,
        epsilon_end=args.eps_end,
        stdscr=None,
        layout=None,
        visualize_every=0,
    )

    np.save(args.output, Q)
    print(f"Saved Q-table to {args.output}")


def train_with_curses(stdscr, args):
    curses.curs_set(0)

    prism_program, model = build_model_with_valuations(args.prism_path)
    layout = build_maze_layout(prism_program, model)
    simulator = create_simulator(model)

    if args.horizon > 0 and layout.lava_states:
        header = f"Running SHIELDED Q-learning with horizon={args.horizon}.\n"
        shield, v_H = compute_vi_based_lava_shield(
            model,
            layout.lava_states,
            horizon=args.horizon,
            risk_threshold=args.risk_threshold,
        )
    else:
        shield = None
        header = "Running UN-SHIELDED Q-learning.\n"

    stdscr.clear()
    stdscr.addstr(0, 0, header + "Training... (visualizing periodically)")
    stdscr.refresh()
    time.sleep(1.0)

    Q = q_learning(
        model=model,
        simulator=simulator,
        shield=shield,
        num_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_start=args.eps_start,
        epsilon_end=args.eps_end,
        stdscr=stdscr,
        layout=layout,
        visualize_every=args.visualize_every,
        viz_delay=args.visual_delay,
    )

    np.save(args.output, Q)
    stdscr.addstr(5, 0, f"Saved Q-table to {args.output}. Press any key to exit.")
    stdscr.refresh()
    stdscr.nodelay(False)
    stdscr.getch()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prism-path",
        type=str,
        required=True,
        help="Path to PRISM model (e.g. simple_maze.prism).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=0,
        help="Shield horizon H for Pmin(F<=H lava). If 0, do not compute a shield.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=200,
        help="Number of training episodes.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Max steps per episode.",
    )
    parser.add_argument(
        "--risk-threshold",
        type=float,
        default=0.0,
        help="Max allowed probability of hitting lava within <= horizon steps. "
             "0.0 means 'no risk allowed'.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Learning rate (alpha).",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor (gamma).",
    )
    parser.add_argument(
        "--eps-start",
        type=float,
        default=0.3,
        help="Initial epsilon.",
    )
    parser.add_argument(
        "--eps-end",
        type=float,
        default=0.01,
        help="Final epsilon.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="Q_unshielded.npy",
        help="Output .npy file for Q-table.",
    )
    parser.add_argument(
        "--visualize-every",
        type=int,
        default=0,
        help="If >0, visualize greedy rollout every N episodes using curses.",
    )
    parser.add_argument(
        "--visual-delay",
        type=float,
        default=0.05,
        help="Delay between visualization steps (seconds).",
    )

    args = parser.parse_args()
    print(args.horizon)

    if args.visualize_every > 0:
        curses.wrapper(train_with_curses, args)
    else:
        train_no_curses(args)


if __name__ == "__main__":
    main()
