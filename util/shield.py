from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

# Optional imports only for debug utilities
try:
    from util.maze_drawer import MazeLayout
except Exception:  # pragma: no cover
    MazeLayout = object  # type: ignore

@dataclass(frozen=True)
class Shield:
    """
    safe_actions_by_state[state_id] = list of *available-action indices*
    considered safe in that state.

    Indices are with respect to simulator.available_actions() list
    and state.actions enumeration.
    """
    safe_actions_by_state: Dict[int, List[int]]

    def safe_indices(self, state_id: int, num_available: int) -> List[int]:
        """
        Return safe action indices in [0, num_available).
        If state_id not in dict -> treat as "no restriction" and return all.
        """
        if state_id not in self.safe_actions_by_state:
            return list(range(num_available))
        raw = self.safe_actions_by_state[state_id]
        return [a for a in raw if 0 <= a < num_available]


    def debug_allowed_actions_xy(
        self,
        model,
        layout: MazeLayout,
        only_restricted: bool = False,
    ) -> None:
        """
        Print a debug mapping (x,y) -> allowed actions.
        Uses choice labels if present; otherwise infers directions from geometry.
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

                labels_or_dirs: List[str] = []
                for a_idx in safe_idxs:
                    label = self._choice_label(model, state_id, a_idx)
                    if label is None:
                        label = self._action_direction(model, layout, state_id, a_idx)
                    labels_or_dirs.append(label)

                print(
                    f"[state {state_id:3d}] ({x},{y}) labels {labels_or_dirs}"
                )

    def _action_direction(
        self,
        model,
        layout: MazeLayout,
        state_id: int,
        action_idx: int,
    ) -> str:
        state = model.states[state_id]
        actions = list(state.actions)
        if action_idx < 0 or action_idx >= len(actions):
            return "?"

        # 1) Try choice labels (PRISM action labels)
        label = self._choice_label(model, state_id, action_idx)
        if label is not None:
            if label in ("E", "W", "N", "S"):
                return label
            if label.lower() in ("stay", "noop", "tau"):
                return "_"
            return label

        # 2) Infer from successor geometry
        action = actions[action_idx]
        (x, y) = layout.state_to_xy[state_id]

        succ_xy = None
        for tr in action.transitions:
            succ_xy = layout.state_to_xy.get(tr.column)
            break

        if succ_xy is None:
            return "?"

        sx, sy = succ_xy
        dx, dy = sx - x, sy - y

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

    def _choice_label(self, model, state_id: int, action_idx: int) -> Optional[str]:
        if not hasattr(model, "choice_labeling") or model.choice_labeling is None:
            return None
        try:
            choice_index = model.get_choice_index(state_id, action_idx)
        except AttributeError:
            return None

        labels = model.choice_labeling.get_labels_of_choice(choice_index)
        if not labels:
            return None
        return next(iter(labels))


def compute_pmin_lava_bounded_vi(
    model,
    lava_states: Set[int],
    horizon: int,
) -> List[List[float]]:
    """
    vs[k][s] = minimal probability to reach lava within <= k steps.
    k = 0..horizon.
    """
    n = model.nr_states

    v0 = [1.0 if s in lava_states else 0.0 for s in range(n)]
    vs: List[List[float]] = [v0]

    v_prev = v0
    for _ in range(1, horizon + 1):
        v_k = [0.0] * n

        for state in model.states:
            s = state.id

            if s in lava_states:
                v_k[s] = 1.0
                continue

            actions = list(state.actions)
            if not actions:
                v_k[s] = v_prev[s]
                continue

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
    prism_program,
    lava_states: Set[int],
    horizon: int,
    risk_threshold: float,
    epsilon: float = 1e-12,
    disable_fallback: bool = True
) -> Tuple[Shield, Dict[int, List[float]]]:
    """
    Safe iff q_H(s,a) = sum_t P(s,a,t) * vs[H-1][t] <= risk_threshold.
    Returns (shield, q_vals_by_state).
    """
    if horizon <= 0:
        return Shield({}), [0.0] * model.nr_states

    vs = compute_pmin_lava_bounded_vi(model, lava_states, horizon)
    v_H_minus_1 = vs[horizon - 1]

    safe_actions_by_state: Dict[int, List[int]] = {}
    q_vals_by_state: Dict[int, List[float]] = {}


    for state in model.states:
        s = state.id
        actions = list(state.actions)

        if not actions:
            safe_actions_by_state[s] = []
            continue

        if s in lava_states:
            safe_actions_by_state[s] = []
            continue

        q_vals: List[float] = []
        for action in actions:
            prob = 0.0
            for tr in action.transitions:
                prob += float(tr.value()) * v_H_minus_1[tr.column]
            q_vals.append(prob)

        q_vals_by_state[s] = q_vals
        safe_idxs = [
            a_idx for a_idx, q in enumerate(q_vals)
            if q <= risk_threshold + epsilon
        ]
        if not disable_fallback:
            if len(safe_idxs) == 0:
                pass
                # TODO
        safe_actions_by_state[s] = safe_idxs

    return Shield(safe_actions_by_state), q_vals_by_state
