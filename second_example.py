import stormpy
## Module/Directory that comes shipped with stormpy containing example model files
from stormpy.examples import files

import sys


prism_program = stormpy.parse_prism_program(f"./die.prism")

formula_str = "P=? [F (\"two\" | \"four\" | \"six\")];"

options = stormpy.BuilderOptions(False, False)
options.set_build_all_labels()

properties = stormpy.parse_properties(formula_str, prism_program)

model = stormpy.build_sparse_model_with_options(prism_program, options)

print(f"\n{model}")
result = stormpy.model_checking(model, properties[0])

for s in model.initial_states:
    print(f"Prob F \"even\" result initial state: {result.at(s)}")

