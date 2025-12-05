import stormpy

def build_model_with_valuations(prism_path: str):
    prism_program = stormpy.parse_prism_program(prism_path)
    # Build with valuations + all labels so we can read x,y, goal, lava, ...
    options = stormpy.BuilderOptions(True, True)
    options.set_build_state_valuations()
    options.set_build_choice_labels(True)
    model = stormpy.build_sparse_model_with_options(prism_program, options)

    return prism_program, model
