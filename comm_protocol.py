# pip install stormpy numpy
import stormpy as sp
import numpy as np


prop_str = 'P=? [ F "delivered" ]'
program = sp.parse_prism_program("./msg_delivery_faulty.prism")
props = sp.parse_properties_for_prism_program(prop_str, program)

options = sp.BuilderOptions(False, False)
options.set_build_all_labels()
options.set_build_state_valuations()
options.set_build_all_reward_models()

model = sp.build_sparse_model_with_options(program, options)
res = sp.model_checking(model, props[0])

initial_state = model.initial_states[0]
lab = model.labeling
num_states = model.nr_states

S_one  = set(lab.get_states("delivered"))
S_zero = set(lab.get_states("gone"))
S_maybe = set(range(num_states)) - S_one - S_zero

row_it = model.transition_matrix

idxT = {s:i for i, s in enumerate(sorted(S_maybe))}
num_maybe_states = len(S_maybe)
A_maybe = np.zeros((num_maybe_states, num_maybe_states))
b = np.zeros(num_maybe_states)

for s in S_maybe:
    r = idxT[s]
    for entry in row_it.get_row(s):
        t = entry.column
        prob = float(entry.value())
        if t in S_maybe:
            A_maybe[r, idxT[t]] += prob
        elif t in S_one:
            b[r] += prob

print(f"\n{A_maybe=}")
print(f"\n{b=}")
# --- Solve (I - P_TT) x_T = b
A = np.eye(num_maybe_states) - A_maybe
print(A)
x_T = np.linalg.solve(A, b)

x = np.zeros(num_states)
for s in S_one: x[s] = 1.0
for s in S_zero: x[s] = 0.0
for s in S_maybe: x[s] = x_T[idxT[s]]

print(f"{x=}")

print("P(F delivered=1) from init (linear equation system) =", x[initial_state])


val_init = res.at(initial_state) if hasattr(res, "at") else res.get_values()[initial_state]
print("P(F delivered=1) from init (stormpy MC)    =", val_init)

print("Difference =", abs(x[initial_state] - val_init))
