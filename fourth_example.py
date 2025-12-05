import stormpy
## Module/Directory that comes shipped with stormpy containing example model files
from stormpy.examples import files

import sys


prism_program = stormpy.parse_prism_program(f"./die.prism")
print(f"\n{prism_program}")

formula_str = """
P=? [F (\"two\" | \"four\" | \"six\")];
R{\"coin_flips\"}=? [F \"done\"];
"""
#
options = stormpy.BuilderOptions(False, False)
options.set_build_all_labels()
options.set_build_state_valuations()
options.set_build_all_reward_models()

properties = stormpy.parse_properties(formula_str, prism_program)

model = stormpy.build_sparse_model_with_options(prism_program, options)

print(f"\n{model}")
result = stormpy.model_checking(model, properties[0])

for s in model.states:
    print(f"Prob F \"even\" result for {s.valuations} = {result.at(s)}")

result = stormpy.model_checking(model, properties[1])

for s in model.states:
    res = result.at(s)
    if result.at(s) <= 0.0:
        continue
    print(f"Expected number of throws until \"done\" for {s.valuations} = {result.at(s)}")



