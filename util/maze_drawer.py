from dataclasses import dataclass
from typing import Dict, Tuple, Set, Optional

import stormpy
import curses
import sys


@dataclass
class MazeLayout:
    width: int
    height: int
    # state_id -> (x,y)
    state_to_xy: Dict[int, Tuple[int, int]]
    # (x,y) -> state_id
    xy_to_state: Dict[Tuple[int, int], int]
    goal_states: Set[int]
    lava_states: Set[int]

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
            . = empty cell
            # = unmapped / missing state

        If show_coords is True, prints a simple coordinate header.
        """
        f = file

        print("=== Maze layout (ASCII) ===", file=f)

        if show_coords:
            # X-axis header
            header = "    " + " ".join(f"{x}" for x in range(self.width))
            if self.width < 10:
                print(header, file=f)
            print("    " + "--" * self.width, file=f)

        for y in range(self.height):
            row_chars = []
            for x in range(self.width):
                s_id = self.xy_to_state.get((x, y), None)
                if s_id is None:
                    ch = "#"
                else:
                    if s_id in self.lava_states:
                        ch = "L"
                    elif s_id in self.goal_states:
                        ch = "G"
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
    """
    Use state valuations to obtain x,y for each state.
    Requires model to be built with state_valuations.
    """

    # Get the Variable objects for x,y from the PRISM program.
    # (prism_program.variables is iterable, elements have .name)
    x_var = next(v for v in prism_program.variables if v.name == "x")
    y_var = next(v for v in prism_program.variables if v.name == "y")

    # Convenience function provided by storm to get integer values per state.
    xs = model.state_valuations.get_integer_values_states(x_var)
    ys = model.state_valuations.get_integer_values_states(y_var)

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

def _get_label_indices(model) -> Dict[str, int]:
    """
    Map label name -> index in BitVector.
    """
    labeling = model.labeling
    labels = sorted(labeling.get_labels())  # names
    name_to_idx = {name: i for i, name in enumerate(labels)}
    return name_to_idx

def _get_labeled_states(model, label_name: str) -> set[int]:
    """
    Return all state ids that have a given label (e.g. "goal" or "lava").

    Uses get_labels_of_state_by_name, which returns a set of label *names*
    for each state. This works regardless of whether get_labels_of_state()
    returns a set or bitvector internally.
    """
    labeling = model.labeling
    states: set[int] = set()
    for s_id in range(model.nr_states):
        names = labeling.get_labels_of_state(s_id)  # set of str
        if label_name in names:
            states.add(s_id)
    return states

def build_maze_layout(prism_program, model) -> MazeLayout:
    state_to_xy, xy_to_state, width, height = _get_xy_values(prism_program, model)

    # no name_to_idx needed anymore
    goal_states = _get_labeled_states(model, "goal")
    lava_states = _get_labeled_states(model, "lava")  # will be empty if no lava yet

    return MazeLayout(
        width=width,
        height=height,
        state_to_xy=state_to_xy,
        xy_to_state=xy_to_state,
        goal_states=goal_states,
        lava_states=lava_states,
    )


class MazeDrawer:
    """
    Curses-based drawer that understands MazeLayout.
    """

    def __init__(self, stdscr, layout: MazeLayout):
        self.stdscr = stdscr
        self.layout = layout

    def draw(self,
             agent_state: Optional[int],
             extra_info: str = "") -> None:
        """
        Draw entire maze.

        agent_state: current state id of the robot (R).
        extra_info: optional status line below the maze.
        """
        self.stdscr.clear()

        agent_xy = None
        if agent_state is not None:
            agent_xy = self.layout.state_to_xy.get(agent_state, None)

        # Draw grid
        for y in range(self.layout.height):
            for x in range(self.layout.width):
                s_id = self.layout.xy_to_state.get((x, y), None)
                is_robot = (agent_xy is not None and agent_xy == (x, y))
                ch = "?"

                if s_id is None:
                    # Should not happen in your grid model;
                    # treat as wall or unknown.
                    ch = "#"
                else:
                    if s_id in self.layout.lava_states:
                        ch = "L"
                    elif s_id in self.layout.goal_states:
                        ch = "G"
                    else:
                        ch = "."

                    if is_robot:
                        ch = "R"

                # simple spacing so it looks like a grid
                self.stdscr.addstr(y, x * 2, ch)

        # Info line
        info_y = self.layout.height + 1
        self.stdscr.addstr(info_y, 0, extra_info)
        self.stdscr.refresh()
