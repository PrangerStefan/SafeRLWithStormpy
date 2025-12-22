from __future__ import annotations

import random
import time
from dataclasses import dataclass
from collections import deque
from typing import List, Optional, Callable, Deque

import numpy as np

from loguru import logger

from util.shield import Shield
from util.maze_drawer import _dashboard_content
from util import get_choice_label, get_xy


DashboardFn = Callable[..., List[str]]

PER_STEP_COST = -0.01
LAVA_COST = -1.0
GOAL_REWARD = 1.0

try:
    import curses
    from util.maze_drawer import MazeDrawer, MazeLayout
except Exception:  # pragma: no cover
    curses = None  # type: ignore
    MazeDrawer = None  # type: ignore
    MazeLayout = None  # type: ignore

@dataclass
class RolloutStats:
    goals: int = 0
    lava: int = 0

def log_progress(
    mode: str,
    ep: int,
    num_episodes: int,
    step: int,
    max_steps: int,
    ret: float,
    ma50: float,
    goals_total: int,
    lava_total: int,
) -> None:
    print(
        f"[{mode:<9}] "
        f"ep {ep:>4d}/{num_episodes:<4d} "
        f"t {step:>3d}/{max_steps:<3d} "
        f"R {ret:>7.3f}  mov_avg50 {ma50:>7.3f}  "
        f"sum(goal={goals_total:>5d}, lava={lava_total:>5d})  "
        f"avg(goal={goals_total/ep:>5f}, lava={lava_total/ep:>5f})"
    )

def visualize_greedy_episode(
    stdscr,
    layout,
    model,
    shield,
    create_simulator_fn,
    Q: np.ndarray,
    max_steps: int = 100,
    delay: float = 0.05,
    stats: RolloutStats | None = None,
    history_len: int = 12,
) -> bool:
    """
    Greedy rollout with curses drawing + dashboard.
    Returns True if user pressed 'q'.

    `stats` is cumulative across calls if the caller reuses it.
    """
    if stats is None:
        stats = RolloutStats()

    drawer = MazeDrawer(stdscr, layout)
    simulator = create_simulator_fn(model)
    labeling = model.labeling

    action_hist: Deque[str] = deque(maxlen=history_len)

    state, _, _ = simulator.restart()
    total_reward = PER_STEP_COST

    stdscr.nodelay(True)

    def fmt_xy(s: int) -> str:
        xy = layout.state_to_xy.get(s, None)
        return f"{xy}" if xy is not None else "(?,?)"

    def fmt_action(s: int, a_idx: int) -> str:
        lbl = get_choice_label(model, s, a_idx)
        return f"a={a_idx}" if lbl is None else f"a={a_idx}({lbl})"

    available = list(simulator.available_actions())
    dash = _dashboard_content(
        t=0,
        model=model,
        max_steps=max_steps,
        state=state,
        fmt_xy=lambda s: fmt_xy(s),
        total_reward=0,
        action_id=None,
        available_action_ids=available,
        action_hist=list(action_hist),
    )
    drawer.draw(agent_state=state, side_panel_lines=dash, panel_pad=4)
    ep_reward = 0.0

    for step in range(max_steps):
        available = list(simulator.available_actions())
        if not available:
            break

        num_avail = len(available)

        if shield is None:
            candidate_ids = [a for a in available if 0 <= a < num_avail]
        else:
            safe_ids = set(shield.safe_indices(state_id=state, num_available=num_avail))
            candidate_ids = [a for a in available if a in safe_ids and 0 <= a < num_avail]

        if not candidate_ids:
            break

        chosen_idx = int(max(candidate_ids, key=lambda a: Q[state, a]))

        ch = stdscr.getch()
        if ch == ord("q"):
            return True
        action = available[chosen_idx]
        next_state, _, _ = simulator.step(action)

        action_hist.append(f"{fmt_action(state, chosen_idx)} -> s'={next_state}{fmt_xy(next_state)}")

        labels_next = labeling.get_labels_of_state(next_state)

        if "goal" in labels_next:
            total_reward += GOAL_REWARD
            stats.goals += 1
        if "lava" in labels_next:
            total_reward -= LAVA_COST
            stats.lava += 1

        state = next_state
        ep_reward += total_reward

        dash = _dashboard_content(
            t=step,
            model=model,
            max_steps=max_steps,
            state=state,
            fmt_xy=lambda s: fmt_xy(s),
            total_reward=ep_reward,
            action_id=chosen_idx,
            available_action_ids=available,
            action_hist=list(action_hist),
            sum_goals=stats.goals,
            sum_lavas=stats.lava,
        )
        drawer.draw(agent_state=state, side_panel_lines=dash, panel_pad=4)

        time.sleep(delay)

        if simulator.is_done():
            action_hist.append("DONE")
            break

        available = list(simulator.available_actions())
    time.sleep(0.3)
    return False

def q_learning(
    model,
    simulator,
    shield: Optional["Shield"],
    num_episodes: int = 500,
    max_steps_per_episode: int = 200,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon_start: float = 0.3,
    epsilon_end: float = 0.01,
    log_every: int = 10,
    stdscr=None,
    layout=None,
    visualize_every: int = 0,
    viz_delay: float = 0.05,
    create_simulator_fn=None,
    prism_program=None,
) -> np.ndarray:
    num_states = model.nr_states

    max_num_actions = max(len(s.actions) for s in model.states)
    if max_num_actions <= 0:
        raise RuntimeError("Model seems to have no actions.")

    Q = np.zeros((num_states, max_num_actions), dtype=float)
    labeling = model.labeling

    rollout_stats = RolloutStats() if "RolloutStats" in globals() else None

    returns: List[float] = []
    goals_total = 0
    lava_total = 0


    for episode in range(num_episodes):
        ep_reward = 0.0
        frac = episode / max(1, num_episodes - 1)
        epsilon = epsilon_start + frac * (epsilon_end - epsilon_start)

        state, _, _ = simulator.restart()

        for _t in range(max_steps_per_episode):
            available = list(simulator.available_actions())
            if not available:
                break

            if shield is None:
                candidate_ids = [a for a in available if 0 <= a < max_num_actions]
            else:
                safe_ids = set(shield.safe_indices(state_id=state, num_available=max_num_actions))
                candidate_ids = [a for a in available if a in safe_ids and 0 <= a < max_num_actions]

            if not candidate_ids:
                break

            if random.random() < epsilon:
                action_id = random.choice(candidate_ids)
            else:
                best = max(candidate_ids, key=lambda a: Q[state, a])
                action_id = int(best)
            next_state, _, _ = simulator.step(action_id)
            labels_of_next_state = labeling.get_labels_of_state(next_state)

            reward = 0.0
            if "goal" in labels_of_next_state:
                reward = GOAL_REWARD
            elif "lava" in labels_of_next_state:
                reward = LAVA_COST
            step_reward = PER_STEP_COST + reward

            terminal = simulator.is_done() or ("goal" in labels_of_next_state) or ("lava" in labels_of_next_state)

            if terminal:
                target = step_reward
            else:
                next_available = list(simulator.available_actions())
                if not next_available:
                    target = step_reward
                else:
                    if shield is None:
                        next_candidate_ids = [a for a in next_available if 0 <= a < max_num_actions]
                    else:
                        next_safe_ids = set(shield.safe_indices(state_id=next_state, num_available=max_num_actions))
                        next_candidate_ids = [a for a in next_available if a in next_safe_ids and 0 <= a < max_num_actions]

                    max_next_q = max((Q[next_state, a] for a in next_candidate_ids), default=0.0)
                    target = step_reward + gamma * float(max_next_q)

            Q[state, action_id] += alpha * (target - Q[state, action_id])
            ep_reward += step_reward

            state = next_state
            if "goal" in labels_of_next_state:
                goals_total += 1
            elif "lava" in labels_of_next_state:
                lava_total += 1
            if terminal:
                break
        returns.append(ep_reward)

        if log_every > 0 and (episode + 1) % log_every == 0 and visualize_every == 0:
            log_progress(
                mode="SHIELDED" if shield else "UNSHIELDED",
                ep=episode + 1,
                num_episodes=num_episodes,
                step=_t + 1,
                max_steps=max_steps_per_episode,
                ret=ep_reward,
                ma50=float(np.mean(returns[-min(len(returns), 50):])),
                goals_total=goals_total,
                lava_total=lava_total,
            )

        if (
            stdscr is not None
            and layout is not None
            and visualize_every > 0
            and (episode + 1) % visualize_every == 0
        ):
            if create_simulator_fn is None:
                raise RuntimeError("create_simulator_fn must be provided when using visualization from util.rl_q_learning")

            if "visualize_greedy_episode" in globals():
                if rollout_stats is not None:
                    want_quit = visualize_greedy_episode(
                        stdscr=stdscr,
                        layout=layout,
                        model=model,
                        shield=shield,
                        create_simulator_fn=create_simulator_fn,
                        Q=Q,
                        delay=viz_delay,
                        max_steps=max_steps_per_episode,
                        stats=rollout_stats,
                    )
                else:
                    want_quit = visualize_greedy_episode(
                        stdscr=stdscr,
                        layout=layout,
                        model=model,
                        shield=shield,
                        create_simulator_fn=create_simulator_fn,
                        Q=Q,
                        delay=viz_delay,
                        max_steps=max_steps_per_episode,
                    )
            else:
                want_quit = False

            if want_quit:
                print("User requested quit during visualization.")
                break

    return Q

