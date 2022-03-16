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
  if not sum_1 or not sum_2:
    return 1.0

  # For efficiency, we avoid normalizing compound counts into relative
  # frequencies until later.
  # Note that: (a / b)**coef = (a**coef) / (b**coef)
  numerator = 0.0
  for compound, count_1 in compound_counts_1.items():
    if compound not in compound_counts_2:
      # Contribution will be 0.
      continue
    count_2 = compound_counts_2[compound]
    numerator += count_1**coef * count_2**(1.0 - coef)

  similarity = numerator / (sum_1**coef * sum_2**(1.0 - coef))
  return 1.0 - similarity


def _compute_new_divergence_1(compound_counts_1,
                              compound_counts_2,
                              compounds_to_move,
                              original_divergence,
                              coef=0.1):
  """Returns the updated compound divergence if compounds are moved.

  This function calculates the new compound divergence if the specified
  compounds are moved from compound_counts_1 to compound_counts_2.

  Args:
    compound_counts_1: Compound counter for examples_1.
    compound_counts_2: Compound counter for examples_2.
    compounds_to_move: The set of compounds to move from examples_1 to
      examples_2.
    original_divergence: The original compound divergence.
    coef: The coefficient used in _compute_divergence.
  """
  sum_1 = sum(compound_counts_1.values())
  sum_2 = sum(compound_counts_2.values())
  original_denominator = sum_1**coef * sum_2**(1.0 - coef)

  original_similarity = 1.0 - original_divergence
  numerator = original_denominator * original_similarity
  # For each compound moved, we update the numerator of the similarity
  # expression, which is a sum over all compounds.
  for compound in compounds_to_move:
    count_1 = compound_counts_1[compound]
    count_2 = compound_counts_2[compound]

    original_numerator_for_compound = count_1**coef * count_2**(1.0 - coef)
    new_numerator_for_compound = ((count_1 - 1)**coef *
                                  (count_2 + 1)**(1.0 - coef))

    numerator += new_numerator_for_compound - original_numerator_for_compound

  new_denominator = ((sum_1 - len(compounds_to_move))**coef *
                     (sum_2 + len(compounds_to_move))**(1.0 - coef))
  new_similarity = numerator / new_denominator
  return 1.0 - new_similarity


def _compute_new_divergence_2(compound_counts_1,
                              compound_counts_2,
                              compounds_to_move,
                              original_divergence,
                              coef=0.1):
  """Returns the updated compound divergence if compounds are moved.

  This function calculates the new compound divergence if the specified
  compounds are moved from compound_counts_2 to compound_counts_1.

  Args:
    compound_counts_1: Compound counter for examples_1.
    compound_counts_2: Compound counter for examples_2.
    compounds_to_move: The set of compounds to move from examples_2 to
      examples_1.
    original_divergence: The original compound divergence.
    coef: The coefficient used in _compute_divergence.
  """
  sum_1 = sum(compound_counts_1.values())
  sum_2 = sum(compound_counts_2.values())
  original_denominator = sum_1**coef * sum_2**(1.0 - coef)

  original_similarity = 1.0 - original_divergence
  numerator = original_denominator * original_similarity
  # For each compound moved, we update the numerator of the similarity
  # expression, which is a sum over all compounds.
  for compound in compounds_to_move:
    count_1 = compound_counts_1[compound]
    count_2 = compound_counts_2[compound]

    original_numerator_for_compound = count_1**coef * count_2**(1.0 - coef)
    new_numerator_for_compound = ((count_1 + 1)**coef *
                                  (count_2 - 1)**(1.0 - coef))

    numerator += new_numerator_for_compound - original_numerator_for_compound

  new_denominator = ((sum_1 + len(compounds_to_move))**coef *
                     (sum_2 - len(compounds_to_move))**(1.0 - coef))
  new_similarity = numerator / new_denominator
  return 1.0 - new_similarity


def _get_all_compounds(examples, get_compounds_fn):
  compounds_to_count = collections.Counter()
  for example in examples:
    compounds_to_count.update(get_compounds_fn(example))
  return compounds_to_count


def measure_example_divergence(examples_1, examples_2, get_compounds_fn):
  compounds_1 = _get_all_compounds(examples_1, get_compounds_fn)
  compounds_2 = _get_all_compounds(examples_2, get_compounds_fn)
  return _compute_divergence(compounds_1, compounds_2)


def _shifted_enumerate(items, start_idx):
  """Yields (index, item) pairs starting from start_idx.

  This function yields the same elements as `enumerate`, but starting from the
  specified start_idx.

  Examples:
    With items = ['a', 'b', 'c'], if start_idx is 0, then this function yields
    (0, 'a'), (1, 'b'), (2, 'c') (the same behavior as `enumerate`), but if
    start_idx is 1, then this function yields (1, 'b'), (2, 'c'), (0, 'a').

  Args:
    items: The list of items to enumerate over.
    start_idx: The index to start yielding at.
  """
  for idx in range(start_idx, start_idx + len(items)):
    shifted_idx = idx % len(items)
    item = items[shifted_idx]
    yield shifted_idx, item


def _get_mcd_idx_1(divergence, examples_1, compounds_1, compounds_2, atoms,
                   get_compounds_fn, get_atoms_fn, start_idx):
  """Return index of example to swap from examples_1 to examples_2."""
  for example_idx, example in _shifted_enumerate(examples_1, start_idx):
    # Ensure example does not contain any atom that appears only once in
    # examples_1. Otherwise, we would violate the atom constraint.
    if _contains_atom(example, atoms, get_atoms_fn):
      continue

    # Compute the new compound divergence if we move the example from examples_1
    # to examples_2, ignoring the effect of moving some other example in
    # examples_2 to examples_1 for now.
    compounds_example = get_compounds_fn(example)
    new_divergence = _compute_new_divergence_1(compounds_1, compounds_2,
                                               compounds_example, divergence)

    # Return the first example that we find that would increase compound
    # divergence.
    if new_divergence > divergence:
      return example_idx, example
  return None, None


def _get_mcd_idx_2(divergence, examples_2, compounds_1, compounds_2,
                   get_compounds_fn, start_idx):
  """Return index of example to swap from examples_2 to examples_1."""
  for example_idx, example in _shifted_enumerate(examples_2, start_idx):
    # Compute the change in compound divergence from moving the example from
    # examples_2 to examples_1.
    compounds_example = get_compounds_fn(example)
    new_divergence = _compute_new_divergence_2(compounds_1, compounds_2,
                                               compounds_example, divergence)

    # Return the first example that we find that would increase compuond
    # divergence.
    if new_divergence > divergence:
      return example_idx, example
  return None, None


def maximize_divergence(examples_1, examples_2, get_compounds_fn, get_atoms_fn,
                        max_iterations, max_divergence, min_atom_count):
  """Approx. maximizes compound divergence by iteratively swapping examples."""
  start_idx_1 = 0
  start_idx_2 = 0
  compounds_1 = _get_all_compounds(
      examples_1, get_compounds_fn=get_compounds_fn)
  compounds_2 = _get_all_compounds(
      examples_2, get_compounds_fn=get_compounds_fn)
  for iteration_num in range(max_iterations):
    atoms_1_single = _get_atoms_below_count(
        examples_1, get_atoms_fn, atom_count=min_atom_count)

    # Compute the compound divergence for the current split of examples.
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
        get_atoms_fn=get_atoms_fn,
        start_idx=start_idx_1)

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
        get_compounds_fn=get_compounds_fn,
        start_idx=start_idx_2)

    if not example_2:
      print("Cannot find example_2 idx to swap.")
      break

    compounds_example_2 = get_compounds_fn(example_2)
    compounds_2.subtract(compounds_example_2)
    compounds_1.update(compounds_example_2)

    # Swap the examples.
    print("Swapping %s and %s." % (example_1, example_2))
    examples_1[example_1_idx] = example_2
    examples_2[example_2_idx] = example_1

    # If a swap happens, we continue the search from those indices in the next
    # iteration, since the previously skipped examples might be less likely to
    # increase divergence.
    start_idx_1 = (example_1_idx + 1) % len(examples_1)
    start_idx_2 = (example_2_idx + 1) % len(examples_2)

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


def balance_atoms(examples_1, examples_2, get_atoms_fn, max_iterations,
                  min_atom_count):
  """Attempts to ensure every atom is represented in the first set."""
  for iteration_num in range(max_iterations):
    # Find atoms that appear >= min_atom_count in `examples_1`.
    atoms_1_above = _get_atoms_above_count(examples_1, get_atoms_fn,
                                           min_atom_count)

    # Find atoms in `examples_2`.
    atoms_2 = get_all_atoms(examples_2, get_atoms_fn=get_atoms_fn)

    # Find atoms in examples_2 not in examples_1 at least `min_atom_count`.
    atoms_2_m_1 = atoms_2 - atoms_1_above

    # If there are no atoms in `atoms_2` not in  in examples_1, then we have
    # reached our goal state.
    if not atoms_2_m_1:
      print("Atoms are balanced after %s iterations." % iteration_num)
      return examples_1, examples_2

    # Find atoms that appear <= min_atom_count in `examples_1`.
    atoms_1_below = _get_atoms_below_count(examples_1, get_atoms_fn,
                                           min_atom_count)

    # Find candidates to swap.
    # First, find an example in examples_1 that does not contain any atoms
    # that appear only once in examples_1. Otherwise, moving the examples to
    # examples_2 can take us farther from our goal state.
    example_1_idx, example_1 = _get_swap_idx(
        examples_1, atoms_1_below, get_atoms_fn=get_atoms_fn, contains=False)
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
                   (atoms_1_below, atoms_2_m_1))


def _contains_atom(example, atoms, get_atoms_fn):
  """Returns True if example contains any atom in atoms."""
  example_atoms = get_atoms_fn(example)
  for example_atom in example_atoms:
    if example_atom in atoms:
      return True
  return False


def _get_atoms_to_count(examples, get_atoms_fn):
  """Return map of atom to count of examples containing atom."""
  atoms_to_count = collections.defaultdict(int)
  for example in examples:
    atoms = get_atoms_fn(example)
    for atom in atoms:
      atoms_to_count[atom] += 1
  return atoms_to_count


def _get_atoms_above_count(examples, get_atoms_fn, atom_count):
  """Return set of atoms that appear >= atom_count times across all examples."""
  atoms_to_count = _get_atoms_to_count(examples, get_atoms_fn)
  atoms = set(
      [atom for atom, count in atoms_to_count.items() if count >= atom_count])
  return atoms


def _get_atoms_below_count(examples, get_atoms_fn, atom_count):
  """Return set of atoms that appear <= atom_count times across all examples."""
  atoms_to_count = _get_atoms_to_count(examples, get_atoms_fn)
  atoms = set(
      [atom for atom, count in atoms_to_count.items() if count <= atom_count])
  return atoms


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
                  max_divergence=None,
                  min_atom_count=1,
                  print_frequencies=True):
  """Swaps examples between examples_1 and examples_2 to maximize divergence.

  This approach first balances atoms to ensure that every atom that appears
  in examples_2 appears in examples_1 at least `min_atom_count` times.
  Then, the algorithm identifies a swap between each collection of examples
  that does not violate the atom constraint, but increases compound divergence.
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
    min_atom_count: Minimum amount of times an atom in examples_2 should appear
      in examples_1.
    print_frequencies: Whether to print compound frequencies at the end of
      swapping.

  Returns:
    (examples_1, examples_2) where each list is the same length and type as the
    corresponding input, but examples have been swapped per the method described
    above.
  """
  examples_1, examples_2 = balance_atoms(examples_1, examples_2, get_atoms_fn,
                                         max_iterations, min_atom_count)
  examples_1, examples_2 = maximize_divergence(examples_1, examples_2,
                                               get_compounds_fn, get_atoms_fn,
                                               max_iterations, max_divergence,
                                               min_atom_count)
  if print_frequencies:
    print_compound_frequencies(examples_1, examples_2, get_compounds_fn)
  return examples_1, examples_2
