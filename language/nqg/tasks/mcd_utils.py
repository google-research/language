# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities for measuring and maximizing compound divergence across splits.

The definition of compound divergence is from:
https://arxiv.org/abs/1912.09713
"""

import collections


def _compute_divergence(compound_counts_1, compound_counts_2, coef=0.1):
  """Compute compound divergence using Chernoff coefficient."""
  sum_1 = sum(compound_counts_1.values())
  sum_2 = sum(compound_counts_2.values())
  frequencies_1 = {
      key: float(count) / sum_1 for key, count in compound_counts_1.items()
  }
  frequencies_2 = {
      key: float(count) / sum_2 for key, count in compound_counts_2.items()
  }

  similarity = 0.0
  for compound, frequency_1 in frequencies_1.items():
    if compound not in frequencies_2:
      # Contribution will be 0.
      continue
    frequency_2 = frequencies_2[compound]
    similarity += frequency_1**coef * frequency_2**(1.0 - coef)

  return 1.0 - similarity


def _get_all_compounds(examples, get_compounds_fn):
  compounds_to_count = collections.Counter()
  for example in examples:
    compounds_to_count.update(get_compounds_fn(example))
  return compounds_to_count


def measure_example_divergence(examples_1, examples_2, get_compounds_fn):
  compounds_1 = _get_all_compounds(examples_1, get_compounds_fn)
  compounds_2 = _get_all_compounds(examples_2, get_compounds_fn)
  return _compute_divergence(compounds_1, compounds_2)


def _get_mcd_idx_1(divergence, examples_1, compounds_1, compounds_2, atoms,
                   get_compounds_fn, get_atoms_fn):
  """Return index of example to swap from examples_1 to examples_2."""
  for example_idx, example in enumerate(examples_1):
    # Ensure example does not contain any atom that appears only once in
    # examples_1. Otherwise, we would violate the atom constraint.
    if _contains_atom(example, atoms, get_atoms_fn):
      continue

    # Compute the new compound divergence if we move the example from examples_1
    # to examples_2, ignoring the effect of moving some other example in
    # examples_2 to examples_1 for now.
    # TODO(petershaw): This could potentially be computed more effeciently
    # for larger numbers of compounds by incrementally computing the change
    # in compound divergence over affected compound counts only, and using
    # this as an estimate for the overall change in compound divergence.
    compounds_example = get_compounds_fn(example)
    compounds_1_copy = compounds_1.copy()
    compounds_1_copy.subtract(compounds_example)
    compounds_2_copy = compounds_2.copy()
    compounds_2_copy.update(compounds_example)

    new_divergence = _compute_divergence(compounds_1_copy, compounds_2_copy)

    # Return the first example that we find that would increase compound
    # divergence.
    if new_divergence > divergence:
      return example_idx, example
  return None, None


def _get_mcd_idx_2(divergence, examples_2, compounds_1, compounds_2,
                   get_compounds_fn):
  """Return index of example to swap from examples_2 to examples_1."""
  for example_idx, example in enumerate(examples_2):
    # Compute the change in compound divergence from moving the example from
    # examples_2 to examples_1.
    # TODO(petershaw): This could potentially be computed more effeciently
    # for larger numbers of compounds by incrementally computing the change
    # in compound divergence over affected compound counts only, and using
    # this as an estimate for the overall change in compound divergence.
    compounds_example = get_compounds_fn(example)
    compounds_1_copy = compounds_1.copy()
    compounds_1_copy.update(compounds_example)
    compounds_2_copy = compounds_2.copy()
    compounds_2_copy.subtract(compounds_example)

    # Return the first example that we find that would increase compuond
    # divergence.
    new_divergence = _compute_divergence(compounds_1_copy, compounds_2_copy)
    if new_divergence > divergence:
      return example_idx, example
  return None, None


def maximize_divergence(examples_1, examples_2, get_compounds_fn, get_atoms_fn,
                        max_iterations, max_divergence):
  """Approx. maximizes compound divergence by iteratively swapping examples."""
  for iteration_num in range(max_iterations):
    atoms_1_single = _get_atoms_below_count(examples_1, get_atoms_fn)

    # Compute the compound divergence for the current split of examples.
    compounds_1 = _get_all_compounds(
        examples_1, get_compounds_fn=get_compounds_fn)
    compounds_2 = _get_all_compounds(
        examples_2, get_compounds_fn=get_compounds_fn)
    divergence = _compute_divergence(compounds_1, compounds_2)
    print("Iteration %s divergence: %s" % (iteration_num, divergence))

    if max_divergence and divergence >= max_divergence:
      print("Reached divergence target.")
      break

    # Find a new pair of examples to swap to increase compound divergence.
    # First, we find an example in examples_1 that would increase compound
    # divergence if moved to examples_2, and would not violate the atom
    # constraint.
    example_1_idx, example_1 = _get_mcd_idx_1(
        divergence,
        examples_1,
        compounds_1,
        compounds_2,
        atoms_1_single,
        get_compounds_fn=get_compounds_fn,
        get_atoms_fn=get_atoms_fn)

    if not example_1:
      print("Cannot find example_1 idx to swap.")
      break

    compounds_example_1 = get_compounds_fn(example_1)
    compounds_1.subtract(compounds_example_1)
    compounds_2.update(compounds_example_1)

    # Second, we find an example in examples_2 that would increase compound
    # divergence if moved to examples_1, taking into account the effect of
    # moving the example selected above to examples_2 first.
    example_2_idx, example_2 = _get_mcd_idx_2(
        divergence,
        examples_2,
        compounds_1,
        compounds_2,
        get_compounds_fn=get_compounds_fn)

    if not example_2:
      print("Cannot find example_2 idx to swap.")
      break

    # Swap the examples.
    print("Swapping %s and %s." % (example_1, example_2))
    del examples_1[example_1_idx]
    examples_1.append(example_2)
    del examples_2[example_2_idx]
    examples_2.append(example_1)

  print("Max iterations reached.")

  return examples_1, examples_2


def get_all_atoms(examples, get_atoms_fn):
  atoms = set()
  for example in examples:
    atoms |= get_atoms_fn(example)
  return atoms


def _get_swap_idx(examples, atoms, get_atoms_fn, contains=True):
  """Returns an example based on a constraint over atoms.

  If `contains` is True, returns an example in `examples` that contains any
  atom in `atoms`.

  If `contains` is False, returns an example in `examples` that does not contain
  any atom in `atoms`.

  Args:
    examples: List of examples.
    atoms: Set of atoms.
    get_atoms_fn: Function from an example to set of atoms.
    contains: Bool (see function docstring for usage).

  Returns:
    (example_idx, example) for example meeting criteria in docstring.
  """
  for example_idx, example in enumerate(examples):
    example_contains_atom = _contains_atom(example, atoms, get_atoms_fn)
    if example_contains_atom == contains:
      return example_idx, example
  if contains:
    raise ValueError("Could not find example that contains any atoms in: %s" %
                     atoms)
  else:
    raise ValueError(
        "Could not find example that doesn't contain any atoms in: %s" % atoms)


def balance_atoms(examples_1, examples_2, get_atoms_fn, max_iterations):
  """Attempts to ensure every atom is represented in the first set."""
  for iteration_num in range(max_iterations):
    atoms_1 = get_all_atoms(examples_1, get_atoms_fn=get_atoms_fn)
    atoms_2 = get_all_atoms(examples_2, get_atoms_fn=get_atoms_fn)

    # Find atoms in examples_2 not in examples_1.
    atoms_2_m_1 = atoms_2 - atoms_1
    # If there are no atoms in examples_2 not in examples_1, then we have
    # reached our goal state.
    if not atoms_2_m_1:
      print("Atoms are balanced after %s iterations." % iteration_num)
      return examples_1, examples_2

    # Find atoms that appear only once in examples_1.
    atoms_1_single = _get_atoms_below_count(examples_1, get_atoms_fn)

    # Find candidates to swap.
    # First, find an example in examples_1 that does not contain any atoms
    # that appear only once in examples_1. Otherwise, moving the examples to
    # examples_2 can take us farther from our goal state.
    example_1_idx, example_1 = _get_swap_idx(
        examples_1, atoms_1_single, get_atoms_fn=get_atoms_fn, contains=False)
    # Second, find an example in examples_2 that contains one of the atoms
    # that is currently missing from examples_1.
    example_2_idx, example_2 = _get_swap_idx(
        examples_2, atoms_2_m_1, get_atoms_fn=get_atoms_fn, contains=True)

    # Swap the examples.
    del examples_1[example_1_idx]
    examples_1.append(example_2)
    del examples_2[example_2_idx]
    examples_2.append(example_1)

  raise ValueError("Could not find split that balances atoms [%s] [%s]" %
                   (atoms_1_single, atoms_2_m_1))


def _contains_atom(example, atoms, get_atoms_fn):
  """Returns True if example contains any atom in atoms."""
  example_atoms = get_atoms_fn(example)
  for example_atom in example_atoms:
    if example_atom in atoms:
      return True
  return False


def _get_atoms_below_count(examples, get_atoms_fn, max_count=1):
  """Return set of atoms that appear <= max_count times across all examples."""
  # Map of atom to count of examples containing atom.
  atoms_to_count = collections.defaultdict(int)
  for example in examples:
    atoms = get_atoms_fn(example)
    for atom in atoms:
      atoms_to_count[atom] += 1

  single_atoms = set(
      [atom for atom, count in atoms_to_count.items() if count <= max_count])
  return single_atoms


def print_compound_frequencies(examples_1, examples_2, get_compounds_fn):
  """Prints compound frequencies for debugging."""
  compound_counts_1 = _get_all_compounds(
      examples_1, get_compounds_fn=get_compounds_fn)
  compound_counts_2 = _get_all_compounds(
      examples_2, get_compounds_fn=get_compounds_fn)
  sum_1 = sum(compound_counts_1.values())
  sum_2 = sum(compound_counts_2.values())
  frequencies_1 = {
      key: float(count) / sum_1 for key, count in compound_counts_1.items()
  }
  frequencies_2 = {
      key: float(count) / sum_2 for key, count in compound_counts_2.items()
  }
  for key in set(compound_counts_1.keys()).union(set(compound_counts_2.keys())):
    frequency_1 = frequencies_1.get(key, 0.0)
    frequency_2 = frequencies_2.get(key, 0.0)
    print("%s: %s - %s" % (key, frequency_1, frequency_2))


def swap_examples(examples_1,
                  examples_2,
                  get_compounds_fn,
                  get_atoms_fn,
                  max_iterations=1000,
                  max_divergence=None):
  """Swaps examples between examples_1 and examples_2 to maximize divergence.

  This approach first balances atoms to ensure that every atom that appears
  in examples_2 appears in examples_1. Then, the algorithm
  identifies a swap between each collection of examples that does not violate
  the atom constraint, but increases compound divergence.
  The procedure breaks when no swap that increases compound divergence can
  be found, or max_iterations or max_divergence is reached.

  To generate different splits, a different initial random split into examples_1
  and examples_2 can be used before calling this function.

  Args:
    examples_1: A list of examples of type E.
    examples_2: A list of examples of type E.
    get_compounds_fn: A function from E to a collections.Counter of strings
      representing compounds.
    get_atoms_fn: A function from E to a set of strings representing atoms.
    max_iterations: A maximum number of iterations (i.e. swap) to run for.
    max_divergence: If not None, will break if compound divergence exceeds this
      value.

  Returns:
    (examples_1, examples_2) where each list is the same length and type as the
    corresponding input, but examples have been swapped per the method described
    above.
  """
  examples_1, examples_2 = balance_atoms(examples_1, examples_2, get_atoms_fn,
                                         max_iterations)
  examples_1, examples_2 = maximize_divergence(examples_1, examples_2,
                                               get_compounds_fn, get_atoms_fn,
                                               max_iterations, max_divergence)
  print_compound_frequencies(examples_1, examples_2, get_compounds_fn)
  return examples_1, examples_2
