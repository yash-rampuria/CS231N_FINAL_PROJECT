"""Verify that all new modules import cleanly inside the container."""

import nav_policy.evaluate.offline
import nav_policy.evaluate.closed_loop
import nav_policy.dagger.run_dagger
from nav_policy.data.build_dataset import write_cache

print("imports OK")
print("  evaluate.offline:    ", nav_policy.evaluate.offline.__file__)
print("  evaluate.closed_loop:", nav_policy.evaluate.closed_loop.__file__)
print("  dagger.run_dagger:   ", nav_policy.dagger.run_dagger.__file__)
print("  build_dataset.write_cache:", write_cache)
