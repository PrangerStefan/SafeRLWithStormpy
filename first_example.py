import stormpy

import sys


prism_program = stormpy.parse_prism_program(f"./die_in_class.prism")
print("="*80)
print(f"Input")
print("="*80)
print(f"\n{prism_program}")
model = stormpy.build_sparse_model(prism_program)

print("="*80)
print(f"Created Model")
print("="*80)
print(f"\n{model}")
print("="*80)
