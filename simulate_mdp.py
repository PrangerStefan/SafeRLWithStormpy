#!/usr/bin/env python3
"""
simulate_mdp.py

Stormpy-based simulator for PRISM MDPs (e.g. maze models).

Two modes, controlled via --visualize:

  --visualize false (default):
      * plain text output to stdout
      * shows state valuations, labels, and available actions

  --visualize true:
      * uses MazeLayout + MazeDrawer (from util.maze_drawer)
      * curses-based visualization of the maze
      * supports random / interactive policies

Usage examples:

  # Text mode, random policy
  python simulate_mdp.py --prism-path simple_maze.prism

  # Text mode, interactive policy
  python simulate_mdp.py --prism-path simple_maze.prism --policy interactive

  # Maze visualization, random policy
  python simulate_mdp.py --prism-path simple_maze.prism --visualize

  # Maze visualization, interactive
  python simulate_mdp.py --prism-path simple_maze_sticky_slippery.prism \
      --visualize --policy interactive
"""

from __future__ import annotations

import random
import time
from collections import deque
from typing import Deque

from util.cli_args import build_simulate_mdp_parser
from util.rl_q_learning import PER_STEP_COST, LAVA_COST, GOAL_REWARD
from util.model_builder import (
    build_mdp_with_valuations_and_choice_labels,
    create_simulator,
)
from util.maze_drawer import MazeLayout, MazeDrawer, build_maze_layout, _dashboard_content
from util import get_xy, get_choice_label
from util.shield import Shield, compute_vi_based_lava_shield

def simulate_mdp_text(
    prism_path: str,
    episodes: int,
    max_steps: int,
    seed: int,
    policy: str = "random",
    shield_horizon: int = 0,
    shield_risk_threshold: float = 0.0,
    shield_mode: str = "dashboard",
):
    bm = build_mdp_with_valuations_and_choice_labels(prism_path)
    model = bm.model
    prism_program = bm.prism_program

    print("Model type:", model.model_type)
    print("Number of states:", model.nr_states)
    print("Reward models:", list(model.reward_models.keys()))
    print("Labels:", sorted(model.labeling.get_labels()))

    simulator = create_simulator(model, seed=seed)
    labeling = model.labeling

    layout = build_maze_layout(prism_program, model)
    layout.print_ascii(show_coords=True)

    shield: Shield | None = None
    if shield_horizon and shield_horizon > 0:
        shield, _ = compute_vi_based_lava_shield(
            model=model,
            prism_program=prism_program,
            lava_states=layout.lava_states,
            horizon=shield_horizon,
            risk_threshold=shield_risk_threshold,
        )

    for ep in range(episodes):
        print(f"\n=== Episode {ep+1}/{episodes} ===")
        state, _, _ = simulator.restart()

        available = list(simulator.available_actions())
        num_actions = len(available)

        safe_idxs = None
        if shield is not None and num_actions > 0:
            safe_idxs = shield.safe_indices(state, num_actions)

        for t in range(max_steps):
            xy = get_xy(prism_program, model, state)
            if xy is not None:
                xy_str = f"(x={xy[0]}, y={xy[1]})"
            else:
                xy_str = "no x,y"

            label_names = labeling.get_labels_of_state(state)
            print(
                f"step {t:2d}: state={state:3d}, {xy_str}, "
                f"labels={sorted(label_names)}, "
            )

            if simulator.is_done():
                print("  -> simulator is done (absorbing state).")
                break

            available = list(simulator.available_actions())
            num_actions = len(available)
            if num_actions == 0:
                print("  -> no available actions (deadlock).")
                break

            print("  available actions:")
            for a_idx, action in enumerate(available):
                lab = get_choice_label(model, state, a_idx)
                if lab is None:
                    lab = "?"
                print(f"    {a_idx}: label={lab}, action={action}")

            if policy == "interactive":
                while True:
                    user_input = input(
                        f"  choose action [0..{num_actions-1}] "
                        "(empty=random, 'q' to quit): "
                    ).strip()

                    if user_input == "q":
                        print("  -> quitting simulation.")
                        return

                    if user_input == "":
                        if shield_mode == "enforce" and safe_idxs:
                            a_idx = random.choice(safe_idxs)   # indices into `available`
                        else:
                            a_idx = random.randrange(num_actions)
                        break

                    try:
                        cand = int(user_input)
                        if 0 <= cand < num_actions:
                            if shield_mode == "enforce" and cand not in safe_idxs:
                                continue
                            a_idx = cand
                            break
                    except ValueError:
                        pass

                    print("    invalid choice, try again.")
            else:
                if shield_mode == "enforce" and safe_idxs:
                    a_idx = random.choice(safe_idxs)           # indices into `available`
                else:
                    a_idx = random.randrange(num_actions)
                print(f"  -> random policy chose action index {a_idx}")

            action = available[a_idx]
            next_state, _, _ = simulator.step(action)
            state = next_state

            available = list(simulator.available_actions())
            num_actions = len(available)
            if shield is not None and num_actions > 0:
                safe_idxs = shield.safe_indices(state, num_actions)



def simulate_mdp_maze_curses(
    stdscr,
    prism_path: str,
    episodes: int,
    max_steps: int,
    seed: int,
    policy: str = "random",
    viz_delay: float = 0.05,
    shield_horizon: int = 0,
    shield_risk_threshold: float = 0.0,
    shield_mode: str = "dashboard",
):
    import curses  # local import for clarity

    curses.curs_set(0)

    bm = build_mdp_with_valuations_and_choice_labels(prism_path)
    model = bm.model
    layout: MazeLayout = build_maze_layout(bm.prism_program, model)

    shield: Shield | None = None
    if shield_horizon and shield_horizon > 0:
        shield, q_vals_by_state = compute_vi_based_lava_shield(
            model=model,
            prism_program=bm.prism_program,
            lava_states=layout.lava_states,
            horizon=shield_horizon,
            risk_threshold=shield_risk_threshold,
        )

    simulator = create_simulator(model, seed=seed)
    labeling = model.labeling

    drawer = MazeDrawer(stdscr, layout)
    goals_reached_total = 0
    lava_hits_total = 0

    history_len = 12

    def fmt_xy(s: int) -> str:
        xy = layout.state_to_xy.get(s, None)
        return f"{xy}" if xy is not None else "(?,?)"

    def fmt_action(s: int, a_idx: int) -> str:
        lab = get_choice_label(model, s, a_idx)
        return f"a={a_idx}" if lab is None else f"a={a_idx}({lab})"

    stdscr.nodelay(False)

    for ep in range(episodes):
        state, _,_ = simulator.restart()
        action_hist: Deque[str] = deque(maxlen=history_len)
        total_reward = 0.0

        available = list(simulator.available_actions())
        num_actions = len(available)

        safe_idxs = None
        if shield is not None and num_actions > 0:
            safe_idxs = shield.safe_indices(state, num_actions)
        dash = _dashboard_content(
            model=model,
            t=0,
            max_steps=max_steps,
            state=state,
            sum_goals=goals_reached_total,
            sum_lavas=lava_hits_total,
            total_reward=total_reward,
            action_id=None,
            available_action_ids=available,
            action_hist=list(action_hist),
            fmt_xy=lambda s: fmt_xy(s),
            shield=shield,
            safe_idxs=safe_idxs,
            shield_horizon=shield_horizon,
            shield_risk_threshold=shield_risk_threshold,
            shield_mode=shield_mode,
            q_vals_by_state=q_vals_by_state
        )
        drawer.draw(agent_state=state, side_panel_lines=dash, dashboard_lines=["Controls: q=quit"])

        ep_reward = 0.0
        for t in range(max_steps):
            num_actions = len(available)

            if num_actions == 0:
                dash = _dashboard_content(
                    model=model,
                    t=t,
                    max_steps=max_steps,
                    state=state,
                    total_reward=ep_reward,
                    sum_goals=goals_reached_total,
                    sum_lavas=lava_hits_total,
                    action_id=None,
                    available_action_ids=available,
                    action_hist=list(action_hist),
                    fmt_xy=lambda s: fmt_xy(s),
                    shield=shield,
                    safe_idxs=safe_idxs,
                    shield_horizon=shield_horizon,
                    shield_risk_threshold=shield_risk_threshold,
                    shield_mode=shield_mode,
                    q_vals_by_state=q_vals_by_state
                )
                drawer.draw(agent_state=state, side_panel_lines=dash, extra_info= " No Shield Action available... Press any key.")
                stdscr.nodelay(False)
                stdscr.getch()
                break

            if policy == "interactive":
                while True:
                    ch = stdscr.getch()

                    if ch == ord('q'):
                        return

                    # ENTER / space / 'r' => random action
                    if ch in (10, 13, ord(' '), ord('r')):
                        if shield_mode == "enforce" and safe_idxs:
                            a_idx = random.choice(safe_idxs)
                        else:
                            a_idx = random.randrange(num_actions)
                        break

                    # digits for action index
                    if ord('0') <= ch <= ord('9'):
                        cand = ch - ord('0')
                        if 0 <= cand < num_actions:
                            a_idx = cand
                            if shield_mode == "enforce" and a_idx not in safe_idxs:
                                ch = stdscr.getch()
                                continue
                            break
            else:
                if shield_mode == "enforce":
                    if safe_idxs:
                        a_idx = random.choice(safe_idxs)
                    else:
                        #drawer.draw(agent_state=state, side_panel_lines=dash, extra_info= " No Shield Action available... Press any key.")
                        stdscr.getch()
                        break
                else:
                    a_idx = random.randrange(num_actions)
                curses.napms(int(viz_delay * 1000))
                stdscr.nodelay(True)
                ch = stdscr.getch()
                stdscr.nodelay(False)
                if ch == ord('q'):
                    return

            action = available[a_idx]
            next_state, _, _ = simulator.step(action)
            labels_of_next_state = labeling.get_labels_of_state(next_state)

            total_reward = PER_STEP_COST
            if "goal" in labels_of_next_state:
                reward = GOAL_REWARD
            elif "lava" in labels_of_next_state:
                reward = LAVA_COST
            else:
                reward = 0.0
            total_reward += reward

            action_hist.append(f"{fmt_action(state, a_idx)} -> s'={next_state}{fmt_xy(next_state)}")

            if "goal" in labels_of_next_state:
                goals_reached_total += 1
            if "lava" in labels_of_next_state:
                lava_hits_total += 1

            state = next_state
            safe_idxs = shield.safe_indices(state, num_actions) if shield is not None else None

            available = list(simulator.available_actions())

            ep_reward += total_reward
            dash = _dashboard_content(
                model=model,
                t=t,
                max_steps=max_steps,
                state=state,
                total_reward=ep_reward,
                sum_goals=goals_reached_total,
                sum_lavas=lava_hits_total,
                action_id=a_idx,
                available_action_ids=available,
                action_hist=list(action_hist),
                fmt_xy=lambda s: fmt_xy(s),
                shield=shield,
                safe_idxs=safe_idxs,
                shield_horizon=shield_horizon,
                shield_risk_threshold=shield_risk_threshold,
                shield_mode=shield_mode,
                q_vals_by_state=q_vals_by_state
            )
            drawer.draw(agent_state=state, side_panel_lines=dash, dashboard_lines=["Controls: q=quit"])

            if simulator.is_done():
                curses.napms(int(viz_delay * 1000))
                stdscr.getch()
                break

    drawer.draw(agent_state=state, extra_info="Simulation finished. Press any key.")
    stdscr.nodelay(False)
    stdscr.getch()


def main():
    parser = build_simulate_mdp_parser()
    args = parser.parse_args()
    random.seed(args.seed)

    if args.visualize:
        import curses
        curses.wrapper(
            simulate_mdp_maze_curses,
            prism_path=args.prism_path,
            episodes=args.episodes,
            max_steps=args.max_steps,
            seed=args.seed,
            policy=args.policy,
            viz_delay=args.viz_delay,
            shield_horizon=args.shield_horizon,
            shield_risk_threshold=args.shield_risk_threshold,
            shield_mode=args.shield_mode,
        )
    else:
        simulate_mdp_text(
            prism_path=args.prism_path,
            episodes=args.episodes,
            max_steps=args.max_steps,
            seed=args.seed,
            policy=args.policy,
            shield_horizon=args.shield_horizon,
            shield_risk_threshold=args.shield_risk_threshold,
            shield_mode=args.shield_mode,
        )

if __name__ == "__main__":
    main()
