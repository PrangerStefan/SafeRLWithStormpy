#!/usr/bin/env python3
from __future__ import annotations

import curses
import random
import time
from collections import deque
from typing import Deque, List, Optional, Tuple

import numpy as np

from util import get_choice_label
from util.cli_args import build_eval_maze_parser
from util.rl_q_learning import PER_STEP_COST, LAVA_COST, GOAL_REWARD
from util.maze_drawer import MazeDrawer, build_maze_layout, _dashboard_content
from util.model_builder import (
    build_mdp_with_valuations_and_choice_labels,
    create_simulator,
)


def _fmt_xy(layout, s: int) -> str:
    xy = layout.state_to_xy.get(s, None)
    return f"{xy}" if xy is not None else "(?,?)"


def _choose_action_id(
    state: int,
    available_action_ids: List[int],
    Q: Optional[np.ndarray],
) -> int:
    if not available_action_ids:
        raise RuntimeError("No available actions.")

    if Q is None:
        return random.choice(available_action_ids)

    q_width = Q.shape[1]
    candidates = [a for a in available_action_ids if 0 <= a < q_width]
    if not candidates:
        return random.choice(available_action_ids)

    return max(candidates, key=lambda a: float(Q[state, a]))


def _episode_loop(
    model,
    layout,
    Q: Optional[np.ndarray],
    max_steps: int,
    seed: int,
    viz: bool,
    viz_delay: float,
    stdscr=None,
    totals: Optional[dict] = None,
) -> Tuple[float, int, dict]:
    """
    Run one episode and return (total_reward, final_state, episode_stats).
    If viz=True, uses curses and MazeDrawer with side-panel dashboard.
    """
    simulator = create_simulator(model, seed=seed)
    labeling = model.labeling

    state, _, _ = simulator.restart()
    total_reward = 0.0

    ep_stats = {"goal": 0, "lava": 0, "steps": 0}

    action_hist: Deque[str] = deque(maxlen=12)
    drawer = MazeDrawer(stdscr, layout) if viz else None

    def fmt_xy(s: int) -> str:
        xy = layout.state_to_xy.get(s, None)
        return f"{xy}" if xy is not None else "(?,?)"

    available = list(simulator.available_actions())
    dash = _dashboard_content(
        model=model,
        t=0,
        max_steps=max_steps,
        state=state,
        total_reward=total_reward,
        action_id=None,
        available_action_ids=available,
        action_hist=list(action_hist),
        fmt_xy=lambda s: fmt_xy(s),
    )
    drawer.draw(agent_state=state, side_panel_lines=dash, dashboard_lines=["Controls: q=quit"])

    ep_reward = 0.0
    for step in range(max_steps):
        total_reward = PER_STEP_COST
        ep_stats["steps"] = step

        available = list(simulator.available_actions())
        if not available:
            if viz and drawer is not None:
                drawer.draw(
                    agent_state=state,
                    side_panel_lines=[
                        f"state={state} xy={_fmt_xy(layout, state)}",
                        "DEADLOCK: no available actions",
                        f"Episode return={total_reward:.2f}",
                        "",
                        "Last actions:",
                        *list(action_hist),
                    ],
                )
            break

        action_id = _choose_action_id(state, available, Q)
        action_lbl = get_choice_label(model, state, action_id) or "?"


        next_state, _, _ = simulator.step(action_id)

        total_reward = PER_STEP_COST
        labels_of_next_state = labeling.get_labels_of_state(next_state)
        if "goal" in labels_of_next_state:
            reward = GOAL_REWARD
            ep_stats["goal"] += 1
            if totals is not None:
                totals["goal"] = totals.get("goal", 0) + 1
        if "lava" in labels_of_next_state:
            reward = LAVA_COST
            ep_stats["lava"] += 1
            if totals is not None:
                totals["lava"] = totals.get("lava", 0) + 1
        else:
            reward = 0.0
        total_reward += reward

        action_hist.append(
            f"{state}{_fmt_xy(layout, state)} --{action_id}:{action_lbl}--> {next_state}{_fmt_xy(layout, next_state)}"
        )

        ep_reward += total_reward
        if viz and drawer is not None:
            shown = []
            for a in available[:10]:
                lab = get_choice_label(model, state, a) or "?"
                shown.append(f"{a}:{lab}")
            avail_str = " ".join(shown)
            if len(available) > 10:
                avail_str += f" ... (+{len(available)-10})"

            goals_total = totals.get("goal", 0) if totals else 0
            lava_total = totals.get("lava", 0) if totals else 0

            dash = _dashboard_content(
                model=model,
                t=0,
                max_steps=max_steps,
                state=state,
                sum_goals=goals_total,
                sum_lavas=lava_total,
                total_reward=ep_reward,
                action_id=action_id,
                available_action_ids=available,
                action_hist=list(action_hist),
                fmt_xy=lambda s: fmt_xy(s),
            )
            drawer.draw(agent_state=state, side_panel_lines=dash)

            stdscr.nodelay(True)
            ch = stdscr.getch()
            if ch == ord("q"):
                raise KeyboardInterrupt

        state = next_state

        if simulator.is_done():
            break

        if viz:
            time.sleep(viz_delay)

    return ep_reward, state, ep_stats


def _eval_curses(stdscr, args) -> None:
    curses.curs_set(0)

    bm = build_mdp_with_valuations_and_choice_labels(args.prism_path)
    prism_program, model = bm.prism_program, bm.model
    layout = build_maze_layout(prism_program, model)

    Q = np.load(args.Q_path) if args.Q_path else None
    Q[Q == 0.0] = -np.inf

    totals = {"goal": 0, "lava": 0}
    returns: List[float] = []

    try:
        for ep in range(args.episodes):
            seed = int(args.seed + ep)

            ret, final_state, ep_stats = _episode_loop(
                model=model,
                layout=layout,
                Q=Q,
                max_steps=args.max_steps,
                seed=seed,
                viz=True,
                viz_delay=args.viz_delay,
                stdscr=stdscr,
                totals=totals,
            )
            returns.append(ret)

            drawer = MazeDrawer(stdscr, layout)
            drawer.draw(
                agent_state=final_state,
                side_panel_lines=[
                    f"Episode {ep+1}/{args.episodes} finished",
                    f"Return={ret:.2f}",
                    f"Totals: goal={totals['goal']} lava={totals['lava']}",
                    "",
                    "Press any key for next episode, or q to quit.",
                ],
            )
            stdscr.nodelay(False)
            ch = stdscr.getch()
            if ch == ord("q"):
                break
            stdscr.nodelay(True)

    except KeyboardInterrupt:
        pass

    avg_ret = float(np.mean(returns)) if returns else 0.0
    stdscr.clear()
    stdscr.addstr(0, 0, f"Evaluation complete. Episodes={len(returns)} avg_return={avg_ret:.3f}\n")
    stdscr.addstr(1, 0, f"Totals: goal={totals.get('goal',0)} lava={totals.get('lava',0)}\n")
    stdscr.addstr(3, 0, "Press any key to exit.")
    stdscr.refresh()
    stdscr.getch()


def _eval_no_curses(args) -> None:
    bm = build_mdp_with_valuations_and_choice_labels(args.prism_path)
    prism_program, model = bm.prism_program, bm.model
    layout = build_maze_layout(prism_program, model)

    Q = np.load(args.Q_path) if args.Q_path else None
    Q[Q == 0.0] = -np.inf

    for s in range(Q.shape[0]):
        x, y = layout.state_to_xy[s]
        a_star = int(np.argmax(Q[s]))
        print(f"{s}\t({x},{y})\t{a_star}\t{Q[s]}")

    totals = {"goal": 0, "lava": 0}
    returns: List[float] = []

    for ep in range(args.episodes):
        seed = int(args.seed + ep)
        ret, final_state, ep_stats = _episode_loop(
            model=model,
            layout=layout,
            Q=Q,
            max_steps=args.max_steps,
            seed=seed,
            viz=False,
            viz_delay=args.viz_delay,
            stdscr=None,
            totals=totals,
        )
        returns.append(ret)
        print(
            f"ep {ep+1}/{args.episodes} return={ret:.3f} "
            f"goal_hits={ep_stats['goal']} lava_hits={ep_stats['lava']} final_state={final_state}{_fmt_xy(layout, final_state)}"
        )

    avg_ret = float(np.mean(returns)) if returns else 0.0
    print(f"\nSummary: episodes={args.episodes} avg_return={avg_ret:.3f} totals(goal={totals['goal']}, lava={totals['lava']})")


def main() -> None:
    parser = build_eval_maze_parser()
    args = parser.parse_args()

    if args.visualize:
        curses.wrapper(_eval_curses, args)
    else:
        _eval_no_curses(args)

if __name__ == "__main__":
    main()
