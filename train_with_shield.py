from __future__ import annotations

import curses
import numpy as np

from util.cli_args import build_train_maze_parser
from util.model_builder import build_mdp_with_valuations, create_simulator
from util.maze_drawer import build_maze_layout
from util.shield import compute_vi_based_lava_shield
from util.rl_q_learning import q_learning


def train_no_curses(args):
    bm = build_mdp_with_valuations(args.prism_path)
    prism_program, model = bm.prism_program, bm.model
    layout = build_maze_layout(prism_program, model)

    simulator = create_simulator(model, seed=args.seed)

    shield = None
    if args.horizon > 0 and layout.lava_states:
        print(f"Computing VI-based lava shield with horizon={args.horizon}")
        shield, _ = compute_vi_based_lava_shield(
            model=model,
            prism_program=prism_program,
            lava_states=layout.lava_states,
            horizon=args.horizon,
            risk_threshold=args.risk_threshold,
        )

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
        log_every=args.log_every,
        stdscr=None,
        layout=None,
        visualize_every=0,
        viz_delay=args.viz_delay,
        prism_program=prism_program
    )

    np.save(args.output, Q)
    print(f"Saved Q-table to {args.output}")


def train_with_curses(stdscr, args):
    curses.curs_set(0)

    bm = build_mdp_with_valuations(args.prism_path)
    prism_program, model = bm.prism_program, bm.model
    layout = build_maze_layout(prism_program, model)

    simulator = create_simulator(model, seed=args.seed)

    shield = None
    header = "Running UN-SHIELDED Q-learning.\n"
    if args.horizon > 0 and layout.lava_states:
        header = f"Running SHIELDED Q-learning with horizon={args.horizon}.\n"
        shield, _ = compute_vi_based_lava_shield(
            model=model,
            prism_program=prism_program,
            lava_states=layout.lava_states,
            horizon=args.horizon,
            risk_threshold=args.risk_threshold,
        )

    stdscr.clear()
    stdscr.addstr(0, 0, header + "Training... (visualizing periodically)")
    stdscr.refresh()

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
        log_every=args.log_every,
        stdscr=stdscr,
        layout=layout,
        visualize_every=args.visualize_every,
        viz_delay=args.viz_delay,
        create_simulator_fn=lambda m: create_simulator(m, seed=args.seed),
        prism_program=prism_program
    )

    np.save(args.output, Q)
    stdscr.addstr(5, 0, f"Saved Q-table to {args.output}. Press any key to exit.")
    stdscr.refresh()
    stdscr.nodelay(False)
    stdscr.getch()


def main():
    parser = build_train_maze_parser()
    args = parser.parse_args()

    if args.visualize_every > 0:
        curses.wrapper(train_with_curses, args)
    else:
        train_no_curses(args)


if __name__ == "__main__":
    main()
