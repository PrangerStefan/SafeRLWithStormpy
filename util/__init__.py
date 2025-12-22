import json
from typing import Optional, Tuple

def get_xy(prism_program, model, state_id: int) -> Optional[Tuple[int, int]]:
    """
    Return (x,y) if variables x,y exist and are numeric, else None.
    """
    vals = model.state_valuations

    # find x,y variables by name
    x_var = None
    y_var = None
    for var in prism_program.variables:
        if var.name == "x":
            x_var = var
        elif var.name == "y":
            y_var = var

    if x_var is None or y_var is None:
        return None

    xs = vals.get_values_states(x_var)
    ys = vals.get_values_states(y_var)

    try:
        x_val = int(xs[state_id])
        y_val = int(ys[state_id])
    except (IndexError, ValueError, TypeError):
        return None

    return (x_val, y_val)

def get_choice_label(model, state_id: int, action_id: int) -> Optional[str]:
    """
    Return a single PRISM choice label for the given (state, action-index),
    or None if choice labeling is not available.

    Stormpy may expose 'choice_labeling' as an optional-like object that
    raises RuntimeError('bad optional access') when not present.
    """
    try:
        cl = model.choice_labeling  # may raise RuntimeError
    except RuntimeError:
        return None

    if cl is None:
        return None

    try:
        choice_index = model.get_choice_index(state_id, action_id)
    except (AttributeError, RuntimeError):
        return None

    try:
        labels = cl.get_labels_of_choice(choice_index)
    except RuntimeError:
        return None

    if not labels:
        return None
    return next(iter(labels))
