from __future__ import annotations

import argparse

# ---------------------------------------------------------------------------
# Common building blocks
# ---------------------------------------------------------------------------

def add_prism_path_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--prism-path",
        type=str,
        required=True,
        help="Path to PRISM model file (e.g. simple_maze.prism, die.prism).",
    )


def add_seed_arg(parser: argparse.ArgumentParser, default: int = 0) -> None:
    parser.add_argument(
        "--seed",
        type=int,
        default=default,
        help=f"Random seed (default: {default}).",
    )


def add_common_simulation_args(
    parser: argparse.ArgumentParser,
    default_episodes: int = 3,
    default_max_steps: int = 50,
) -> None:
    parser.add_argument(
        "--episodes",
        type=int,
        default=default_episodes,
        help=f"Number of episodes / restarts (default: {default_episodes}).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=default_max_steps,
        help=f"Maximum number of steps per episode (default: {default_max_steps}).",
    )


def add_policy_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--policy",
        type=str,
        default="random",
        choices=["random", "interactive"],
        help="Policy for simulation: 'random' or 'interactive' (default: random).",
    )

def add_visualization_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--visualize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, use curses + MazeDrawer visualization (default: true).",
    )
    parser.add_argument(
        "--viz-delay",
        type=float,
        default=0.05,
        help="Delay between steps in visual mode, in seconds (default: 0.05).",
    )

# ---------------------------------------------------------------------------
# Parsers for the concrete scripts
# ---------------------------------------------------------------------------

def build_simulate_mc_parser() -> argparse.ArgumentParser:
    """
    Parser for simulate_mc.py (Markov chains, e.g. die.prism).
    """
    parser = argparse.ArgumentParser(
        prog="simulate_mc.py",
        description="Simulate a PRISM Markov chain (DTMC/CTMC) using stormpy.",
    )
    add_prism_path_arg(parser)
    add_seed_arg(parser, default=0)
    add_common_simulation_args(parser, default_episodes=3, default_max_steps=20)
    return parser


def build_simulate_mdp_parser() -> argparse.ArgumentParser:
    """
    Parser for simulate_mdp.py (MDPs, e.g. maze models),
    with optional curses-based visualization.
    """
    parser = argparse.ArgumentParser(
        prog="simulate_mdp.py",
        description="Simulate a PRISM MDP (maze) using stormpy.",
    )
    add_prism_path_arg(parser)
    add_seed_arg(parser, default=0)
    add_common_simulation_args(parser, default_episodes=3, default_max_steps=50)
    add_policy_arg(parser)
    add_visualization_args(parser)

    parser.add_argument(
        "--shield-horizon",
        type=int,
        default=0,
        help="If >0, compute a lava shield with horizon H (default: 0 = off).",
    )
    parser.add_argument(
        "--shield-risk-threshold",
        type=float,
        default=0.0,
        help="Risk threshold for lava within <=H steps (default: 0.0).",
    )
    parser.add_argument(
        "--shield-mode",
        type=str,
        default="dashboard",
        choices=["dashboard", "enforce"],
        help=(
            "dashboard: only display safe actions; "
            "enforce: restrict action selection to safe actions."
        ),
    )
    return parser


def build_train_maze_parser() -> argparse.ArgumentParser:
    """
    Parser for train_q_learning_maze.py:
      - training episodes
      - Q-learning hyperparameters
      - shield parameters (horizon, risk-threshold)
      - optional periodic visualization
    """
    parser = argparse.ArgumentParser(
        prog="train_q_learning_maze.py",
        description="Train a tabular Q-learning agent on a PRISM maze MDP.",
    )
    add_prism_path_arg(parser)
    add_seed_arg(parser, default=42)

    # Q-learning core
    parser.add_argument(
        "--episodes",
        type=int,
        default=200,
        help="Number of training episodes (default: 200).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Maximum number of steps per episode (default: 200).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Learning rate alpha (default: 0.1).",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor gamma (default: 0.99).",
    )
    parser.add_argument(
        "--eps-start",
        type=float,
        default=0.3,
        help="Initial epsilon for epsilon-greedy (default: 0.3).",
    )
    parser.add_argument(
        "--eps-end",
        type=float,
        default=0.01,
        help="Final epsilon for epsilon-greedy (default: 0.01).",
    )

    # Shield parameters
    parser.add_argument(
        "--horizon",
        type=int,
        default=0,
        help=(
            "Shield horizon H for Pmin(F<=H lava). "
            "If 0, no shield is computed (unshielded training)."
        ),
    )
    parser.add_argument(
        "--risk-threshold",
        type=float,
        default=0.0,
        help=(
            "Max allowed probability of hitting lava within <= horizon steps "
            "(0.0 means 'no risk allowed')."
        ),
    )

    # Logging & visualization
    parser.add_argument(
        "--output",
        type=str,
        default="Q_unshielded.npy",
        help="Output .npy file for the learned Q-table.",
    )
    parser.add_argument(
        "--visualize-every",
        type=int,
        default=0,
        help=(
            "If >0, run a greedy rollout with curses visualization every N episodes. "
            "0 disables visualization during training."
        ),
    )
    add_visualization_args(parser)
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Print training stats every N episodes (default: 10).",
    )
    parser.add_argument(
        "--ma-window",
        type=int,
        default=50,
        help="Moving average window for printed returns (default: 50).",
    )

    return parser


def build_eval_maze_parser() -> argparse.ArgumentParser:
    """
    Parser for evaluate_policy_maze.py:
      - load a Q-table
      - run greedy episodes (with or without visualization)
    """
    parser = argparse.ArgumentParser(
        prog="evaluate_policy_maze.py",
        description="Evaluate a learned Q-policy on a PRISM maze MDP.",
    )
    add_prism_path_arg(parser)
    add_seed_arg(parser, default=123)

    parser.add_argument(
        "--Q-path",
        type=str,
        required=True,
        help="Path to .npy file with learned Q-table.",
    )
    add_common_simulation_args(parser, default_episodes=20, default_max_steps=200)
    add_visualization_args(parser)

    return parser
