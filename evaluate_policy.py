import time
import curses
import random
import numpy as np
import stormpy
import stormpy.simulator

from util.maze_drawer import build_maze_layout, MazeDrawer
from util.model_builder import build_model_with_valuations  # from step 1


def run_episode_simulator(simulator, max_steps: int, policy_Q=None):
    """
    policy_Q: optional Q-table; if None, choose actions uniformly at random.
    Returns total reward.
    """
    state, reward_vec, labels = simulator.restart()
    total_reward = 0.0

    for _ in range(max_steps):
        available = list(simulator.available_actions())
        if not available:
            break

        if policy_Q is None:
            # purely random
            a_idx = random.randrange(len(available))
        else:
            num_avail = len(available)
            a_idx = int(np.argmax(policy_Q[state, :num_avail]))

        action = available[a_idx]
        next_state, reward_vec, labels = simulator.step(action)
        reward = float(reward_vec[0]) if reward_vec else 0.0
        total_reward += reward
        state = next_state

        if simulator.is_done():
            break

    return total_reward, state


def eval_loop(stdscr, prism_path: str, Q_path: str | None,
              episodes: int = 10, max_steps: int = 100):
    curses.curs_set(0)

    prism_program, model = build_model_with_valuations(prism_path)
    layout = build_maze_layout(prism_program, model)
    drawer = MazeDrawer(stdscr, layout)


    Q = None
    if Q_path is not None:
        Q = np.load(Q_path)

    for ep in range(episodes):
        simulator = stormpy.simulator.create_simulator(model)
        state, reward_vec, labels = simulator.restart()
        total_reward = 0.0

        for step in range(max_steps):
            drawer.draw(
                agent_state=state,
                extra_info=f"Episode {ep+1}/{episodes}, step {step}, return={total_reward:.2f}"
            )
            time.sleep(0.05)

            available = list(simulator.available_actions())
            if not available:
                break

            if Q is None:
                a_idx = random.randrange(len(available))
            else:
                num_avail = len(available)
                a_idx = int(np.argmax(Q[state, :num_avail]))

            action = available[a_idx]
            next_state, reward_vec, labels = simulator.step(action)
            reward = float(reward_vec[0]) if reward_vec else 0.0
            total_reward += reward
            state = next_state

            if simulator.is_done():
                break

        drawer.draw(
            agent_state=state,
            extra_info=f"Episode {ep+1}/{episodes} finished, return={total_reward:.2f} (press any key)"
        )
        stdscr.nodelay(False)
        stdscr.getch()
        stdscr.nodelay(True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--prism-path", type=str, required=True)
    parser.add_argument("--Q", type=str, default=None,
                        help="Optional path to Q-table .npy; if omitted, random policy is used.")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=100)
    args = parser.parse_args()

    curses.wrapper(
        eval_loop,
        prism_path=args.prism_path,
        Q_path=args.Q,
        episodes=args.episodes,
        max_steps=args.max_steps,
    )
