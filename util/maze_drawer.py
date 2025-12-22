from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Set, Optional, List

import sys

from util import get_choice_label

import curses

def _pad(label: str, value: str, w: int = 12) -> str:
    return f"{label:<{w}} {value}"


def _dashboard_sim_section(
    t: int,
    max_steps: int,
    state: int,
    total_reward: float,
    fmt_xy: Callable[[int], str],
    sum_goals: Optional[int] = None,
    sum_lavas: Optional[int] = None,
) -> List[str]:
    return [
        "SIMULATION",
        "-" * 30,
        _pad("step:", f"{t}/{max_steps}"),
        _pad("state:", f"{state}  xy={fmt_xy(state)}"),
        _pad("goals", f"{sum_goals}") + "\t" + _pad("lava", f"{sum_lavas}"),
        _pad("return:", f"{total_reward:+.2f}"),
        "",
    ]


def _dashboard_actions_section(
    model,
    state: int,
    action_id: Optional[int],
    available_action_ids: Optional[List[int]],
    max_avail_labels: int = 10,
) -> List[str]:
    if available_action_ids is None:
        avail_block = ["avail labels:", "  (pending)", ""]
    else:
        shown = []
        for a in available_action_ids[:max_avail_labels]:
            lab = get_choice_label(model, state, a) or "?"
            shown.append(f"{a}:{lab}")
        avail_str = " ".join(shown)
        if len(available_action_ids) > max_avail_labels:
            avail_str += f" ... (+{len(available_action_ids) - max_avail_labels})"
        avail_block = [
            "avail labels:",
            f"  {avail_str}" if avail_str else "  (none)",
            "",
        ]

    if action_id is None:
        chosen_line = _pad("chosen:", "(pending)")
    else:
        chosen_lbl = get_choice_label(model, state, action_id) or "?"
        chosen_line = _pad("chosen:", f"{action_id}:{chosen_lbl}")


    return [
        "ACTIONS",
        "-" * 30,
        *avail_block,
        chosen_line,
    ]


def _dashboard_shield_section(
    model,
    state: int,
    shield,
    safe_idxs: Optional[List[int]],
    shield_horizon: int,
    shield_risk_threshold: float,
    shield_mode: str,
    q_vals_by_state: Dict[int, List[float]],
    max_safe_labels: int = 12,
) -> List[str]:
    if shield is None:
        return [
            "SHIELD",
            "-" * 30,
            "Shield: OFF",
            "",
        ]

    if safe_idxs is None:
        return [
            "SHIELD",
            "-" * 30,
            f"Shield: ON  H={shield_horizon}  thr={shield_risk_threshold:g}  mode={shield_mode}",
            "Safe actions:",
            "  (pending)",
            "",
        ]

    safe_labs = []
    for a_idx in safe_idxs[:max_safe_labels]:
        lab = get_choice_label(model, state, a_idx) or "?"
        safe_labs.append(f"{a_idx}:{lab}")
    safe_str = " ".join(safe_labs)
    if len(safe_idxs) > max_safe_labels:
        safe_str += f" ... (+{len(safe_idxs) - max_safe_labels})"

    q_vals_line = []
    if q_vals_by_state is not None:
        q_vals_line.append("q_H(s,a):")
        try:
            for a_idx, q in enumerate(q_vals_by_state[state]):
                q_vals_line.append(f"  a={a_idx}:{q:.8f}")
        except KeyError:
            q_vals_line = ["No safe actions available"]

    return [
        "SHIELD",
        "-" * 30,
        f"Shield: ON  H={shield_horizon}  thr={shield_risk_threshold:g}  mode={shield_mode}",
        f"{' '.join(q_vals_line)}",
        "Safe actions:",
        f"  {safe_str}" if safe_str else "  (none)",
        "",
    ]


def _dashboard_history_section(action_hist: List[str], max_lines: int = 12) -> List[str]:
    hist = action_hist[-max_lines:] if len(action_hist) > max_lines else action_hist
    return [
        "HISTORY (last 12)",
        "-" * 30,
        *[f"  {h}" for h in hist],
        "",
        "Controls: q=quit",
    ]


def _dashboard_content(
    model,
    t: int,
    max_steps: int,
    state: int,
    total_reward: float,
    fmt_xy: Callable[[int], str],
    sum_goals: Optional[int] = None,
    sum_lavas: Optional[int] = None,
    *,
    action_id: Optional[int] = None,
    available_action_ids: Optional[List[int]] = None,
    action_hist: Optional[List[str]] = None,
    shield=None,
    safe_idxs: Optional[List[int]] = None,
    shield_horizon: int = 0,
    shield_risk_threshold: float = 0.0,
    shield_mode: str = "dashboard",
    q_vals_by_state: Dict[int, List[float]] = {}
) -> List[str]:
    dash: List[str] = []
    dash += _dashboard_sim_section(t, max_steps, state, total_reward, fmt_xy, sum_goals=sum_goals, sum_lavas=sum_lavas)
    dash += _dashboard_actions_section(model, state, action_id, available_action_ids) + [" "]
    if shield is not None:
        dash += _dashboard_shield_section(
            model,
            state,
            shield=shield,
            safe_idxs=safe_idxs,
            shield_horizon=shield_horizon,
            shield_risk_threshold=shield_risk_threshold,
            shield_mode=shield_mode,
            q_vals_by_state=q_vals_by_state
        )
    dash += _dashboard_history_section(action_hist or [])
    return dash


@dataclass
class MazeLayout:
    width: int
    height: int
    state_to_xy: Dict[int, Tuple[int, int]]
    xy_to_state: Dict[Tuple[int, int], int]
    goal_states: Set[int]
    lava_states: Set[int]
    slippery_states: Set[int]

    def print_ascii(
        self,
        file=sys.stdout,
        show_coords: bool = True,
    ) -> None:
        """
        Print an ASCII representation of the maze to stdout (or another file-like).

        Legend:
            G = goal
            L = lava
            s = slippery
            . = empty cell
            # = unmapped / missing state / wall (i.e. unreachable states)

        If show_coords is True, prints a simple coordinate header.
        """
        f = file
        print(f"{self.height} - {self.width}")

        print("=== Maze layout (ASCII) ===", file=f)

        if show_coords:
            header = "    " + " ".join(f"{x}" for x in range(1, self.width + 1))
            if self.width < 10:
                print(header, file=f)
            print("    " + "--" * self.width, file=f)

        for y in range(1, self.height + 1):
            row_chars = []
            for x in range(1, self.width + 1):
                s_id = self.xy_to_state.get((x, y), None)
                if s_id is None:
                    ch = "#"
                else:
                    if s_id in self.lava_states:
                        ch = "L"
                    elif s_id in self.goal_states:
                        ch = "G"
                    elif s_id in self.slippery_states:
                        ch = "s"
                    else:
                        ch = "."
                row_chars.append(ch)


            if show_coords:
                print(f"{y:2d} | " + " ".join(row_chars), file=f)
            else:
                print(" ".join(row_chars), file=f)

        print("Legend: G=goal, L=lava, .=empty, #=unmapped\n", file=f)

def _get_xy_values(prism_program, model) -> Tuple[Dict[int, Tuple[int, int]],
                                                 Dict[Tuple[int, int], int],
                                                 int, int]:
    x_var = next(v for v in prism_program.variables if v.name == "x")
    y_var = next(v for v in prism_program.variables if v.name == "y")

    xs = model.state_valuations.get_integer_values_states(x_var)
    ys = model.state_valuations.get_integer_values_states(y_var)

    state_to_xy: Dict[int, Tuple[int, int]] = {}
    xy_to_state: Dict[Tuple[int, int], int] = {}

    for s_id in range(model.nr_states):
        x = int(xs[s_id])
        y = int(ys[s_id])
        state_to_xy[s_id] = (x, y)
        xy_to_state[(x, y)] = s_id

    width  = next(c for c in prism_program.constants if c.name == "WIDTH")
    height = next(c for c in prism_program.constants if c.name == "HEIGHT")
    return state_to_xy, xy_to_state, int(str(width.definition)), int(str(height.definition))

def _get_labeled_states(model, label_name: str) -> set[int]:
    labeling = model.labeling
    states: set[int] = set()
    for s_id in range(model.nr_states):
        names = labeling.get_labels_of_state(s_id)  # set of str
        if label_name in names:
            states.add(s_id)
    return states

def build_maze_layout(prism_program, model) -> MazeLayout:
    state_to_xy, xy_to_state, width, height = _get_xy_values(prism_program, model)

    goal_states = _get_labeled_states(model, "goal")
    lava_states = _get_labeled_states(model, "lava")
    slippery_states = _get_labeled_states(model, "slippery")

    return MazeLayout(
        width=width,
        height=height,
        state_to_xy=state_to_xy,
        xy_to_state=xy_to_state,
        goal_states=goal_states,
        lava_states=lava_states,
        slippery_states=slippery_states,
    )


class MazeDrawer:
    PAIR_BG   = 1
    PAIR_SLIP = 2
    PAIR_LAVA = 3
    PAIR_GOAL = 4
    PAIR_WALL = 5

    def __init__(self, stdscr, layout):
        self.stdscr = stdscr
        self.layout = layout

        # Conservative default
        self._has_colors = False

        try:
            self._has_colors = curses.has_colors()
        except Exception:
            self._has_colors = False

        if self._has_colors:
            try:
                curses.start_color()
            except curses.error:
                self._has_colors = False

        if self._has_colors:
            # Optional; may fail on some terminals
            try:
                curses.use_default_colors()
            except curses.error:
                pass

            # Global background pair (choose your default background)
            # If you want a true "terminal background", use bg = -1 (default) where supported.
            bg = -1
            fg = curses.COLOR_WHITE

            try:
                curses.init_pair(self.PAIR_BG, fg, bg)
                curses.init_pair(self.PAIR_SLIP, curses.COLOR_WHITE, curses.COLOR_BLUE)
                curses.init_pair(self.PAIR_LAVA, curses.COLOR_WHITE, curses.COLOR_RED)
                curses.init_pair(self.PAIR_GOAL, curses.COLOR_BLACK, curses.COLOR_GREEN)
                curses.init_pair(self.PAIR_WALL, curses.COLOR_WHITE, curses.COLOR_BLACK)
            except curses.error:
                # If init_pair fails, disable colors cleanly
                self._has_colors = False


    def _tile_attr(self, s_id):
        if not self._has_colors:
            return 0

        # walls = "no state"
        if s_id is None:
            return curses.color_pair(self.PAIR_WALL) | curses.A_DIM

        if s_id in self.layout.lava_states:
            return curses.color_pair(self.PAIR_LAVA)

        if s_id in self.layout.goal_states:
            return curses.color_pair(self.PAIR_GOAL)

        if s_id in self.layout.slippery_states:
            return curses.color_pair(self.PAIR_SLIP)

        return 0

    def draw(
        self,
        agent_state: Optional[int],
        extra_info: str = "",
        dashboard_lines: Optional[List[str]] = None,
        side_panel_lines: Optional[List[str]] = None,
        panel_pad: int = 4,
    ) -> None:
        """
        Draw maze grid and optional dashboards.

        - dashboard_lines: multi-line block *below* the grid
        - side_panel_lines: multi-line block *to the right* of the grid
        """
        self.stdscr.clear()

        max_y, max_x = self.stdscr.getmaxyx()

        agent_xy = None
        if agent_state is not None:
            agent_xy = self.layout.state_to_xy.get(agent_state, None)

        for y in range(1, self.layout.height + 1):
            if y >= max_y:
                break
            for x in range(1, self.layout.width + 1):
                col = x * 2
                if col >= max_x:
                    break

                s_id = self.layout.xy_to_state.get((x, y), None)
                is_robot = (agent_xy is not None and agent_xy == (x, y))

                if s_id is None:
                    ch = "#"
                else:
                    if s_id in self.layout.lava_states:
                        ch = "L"
                    elif s_id in self.layout.goal_states:
                        ch = "G"
                    elif s_id in self.layout.slippery_states:
                        ch = "s"
                    else:
                        ch = "."
                    if is_robot:
                        ch = "R"
                attr = self._tile_attr(s_id)

                # If the robot is here, make it pop without losing background.
                if is_robot:
                    ch = "R"
                    attr = attr | curses.A_BOLD

                self.stdscr.addstr(y, col, ch, attr)
                self.stdscr.addstr(y, col + 1, " ", attr)

        if side_panel_lines:
            grid_width_chars = self.layout.width * 2
            panel_x = grid_width_chars + panel_pad
            if panel_x < max_x - 1:
                for i, line in enumerate(side_panel_lines):
                    y = i
                    if y >= max_y:
                        break
                    avail = max_x - panel_x - 1
                    if avail <= 0:
                        break
                    self.stdscr.addstr(y, panel_x, line[:avail])

        info_y = self.layout.height + 1
        lines: List[str]
        if dashboard_lines is not None:
            lines = dashboard_lines
        else:
            lines = [extra_info] if extra_info else []

        for i, line in enumerate(lines):
            y = info_y + i
            if y >= max_y:
                break
            self.stdscr.addstr(y, 0, line[: max_x - 1])

        self.stdscr.refresh()
