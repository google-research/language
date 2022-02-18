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
"""Implements CKY parsing using a Trie data structure to index rules.

This implementation supports non-binarized grammars with rules containing
up to 2 non-terminals.

For each span, rather than enumerating every possible sub-span for up to
2 non-terminals, the algorithm iterates across the span left-to-right and
attempts to match rules stored in a Trie.
"""

import collections

from language.compgen.nqg.common.cky import cfg_rule


class TrieNode(object):
  """Represents a node in a generic Trie data structure."""

  def __init__(self, symbol=None):
    # The individual symbol associated with this node.
    self.symbol = symbol  # Can only be None for root.
    # Map from symbol to TrieNode.
    self.symbol_to_child = {}
    # A set of arbitrarily-typed values associated with this node.
    self.values = []

  def maybe_add_child(self, symbol):
    """Adds a new node for a given child symbol if not already in Trie."""
    if symbol in self.symbol_to_child:
      return self.symbol_to_child[symbol]
    else:
      node = TrieNode(symbol)
      self.symbol_to_child[symbol] = node
      return node

  def maybe_get_child(self, symbol):
    return self.symbol_to_child.get(symbol)

  def __str__(self):
    return "%s %s" % (self.symbol, set(self.symbol_to_child.keys()))

  def __repr__(self):
    return str(self)


def print_trie(trie_node, indent=0):
  """Recursively prints Trie for debugging purposes."""
  print("%s %s" % ("-" * indent, trie_node.symbol))
  for value in trie_node.values:
    print("%s value: %s" % ("-" * indent, value))
  for child in trie_node.symbol_to_child.values():
    print_trie(child, indent=indent + 1)


def add_rule_to_trie(trie_root, rule):
  current_node = trie_root
  for symbol in rule.rhs:
    current_node = current_node.maybe_add_child(symbol)
  current_node.values.append(rule)


class Chart(object):
  """Represents parse chart state."""

  def __init__(self, populate_fn, postprocess_fn):
    # The key_map stores chart entries (of type T) indexed by:
    # (span_begin, span_end, nonterminal)
    self.key_map = collections.defaultdict(list)
    # For optimization purposes, we also index chart entries by their
    # span_begin index only in start_map.
    # Key is span_begin and value is List of (span_end, nonterminal).
    self.start_map = collections.defaultdict(set)
    # See `cfg_parser.py` for definitions of populate_fn and postprocess_fn.
    self.populate_fn = populate_fn
    self.postprocess_fn = postprocess_fn

  def add(self, span_begin, span_end, rule, children):
    """Add an entry to the chart."""
    entry = self.populate_fn(span_begin, span_end, rule, children)
    nonterminal = rule.lhs
    self.key_map[(span_begin, span_end, nonterminal)].append(entry)
    self.start_map[span_begin].add((span_end, nonterminal))

  def get_from_key(self, span_begin, span_end, nonterminal):
    """Get entries based on full key."""
    return self.key_map[(span_begin, span_end, nonterminal)]

  def get_from_start(self, span_begin):
    """Get entries based on start index only."""
    return self.start_map[span_begin]

  def postprocess(self, span_begin, span_end, nonterminal):
    """Apply postpostprocess_fn to a chart cell."""
    if self.postprocess_fn:
      self.key_map[(span_begin, span_end, nonterminal)] = self.postprocess_fn(
          self.key_map[(span_begin, span_end, nonterminal)])


# For a given span, SearchState represents a potential match with a ParserRule.
SearchState = collections.namedtuple(
    "SearchState",
    [
        "anchored_nonterminals",  # List of (span_begin, span_end, nonterminal).
        "trie_node",  # TrieNode.
    ])

# The maximum number of RHS non-terminals in ParserRules that are supported.
MAX_NONTERMINALS = 2


def parse(input_ids,
          rules,
          nonterminals,
          start_idx,
          populate_fn,
          postprocess_fn,
          verbose=False):
  """Run bottom up parser using Trie-based implementation."""
  input_len = len(input_ids)
  input_symbols = tuple(
      [cfg_rule.CFGSymbol(idx, cfg_rule.TERMINAL) for idx in input_ids])

  # Initialize the empty chart.
  chart = Chart(populate_fn, postprocess_fn)

  # Initialize Trie of rules.
  trie_root = TrieNode()
  for rule in rules:
    add_rule_to_trie(trie_root, rule)

  # Populate the chart.
  for span_end in range(1, input_len + 1):
    for span_begin in range(span_end - 1, -1, -1):

      # Map of span_begin to List of SearchState.
      search_map = collections.defaultdict(list)
      search_map[span_begin].append(SearchState([], trie_root))

      # Iterate across every input token in the span range to find rule matches.
      for idx in range(span_begin, span_end):

        # End early if there are no remaining candidate matches.
        if not search_map[idx]:
          continue

        terminal_symbol = input_symbols[idx]

        # Iterate through partial matches.
        while search_map[idx]:
          search_state = search_map[idx].pop()

          # Consider matching terminal.
          new_trie_node = search_state.trie_node.maybe_get_child(
              terminal_symbol)
          if new_trie_node:
            # Found a match for the terminal in the Trie.
            # Add a partial match to search_map with idx incremented by 1 token.
            new_search_state = SearchState(search_state.anchored_nonterminals,
                                           new_trie_node)
            search_map[idx + 1].append(new_search_state)

          # Consider matching non-terminal.
          nonterminal_tuples = chart.get_from_start(idx)
          if len(search_state.anchored_nonterminals) < MAX_NONTERMINALS:
            # Iterate through lower chart entries with a completed sub-tree
            # that starts at the current index.
            for nt_end, nonterminal in nonterminal_tuples:
              nonterminal_symbol = cfg_rule.CFGSymbol(nonterminal,
                                                      cfg_rule.NON_TERMINAL)
              new_trie_node = search_state.trie_node.maybe_get_child(
                  nonterminal_symbol)
              if new_trie_node:
                # Found a match for the non-terminal in the Trie.
                # Add a partial match to search_map with idx set to the end
                # of the sub-tree span.
                new_anchored_nonterminals = search_state.anchored_nonterminals[:]
                new_anchored_nonterminals.append((idx, nt_end, nonterminal))
                search_map[nt_end].append(
                    SearchState(new_anchored_nonterminals, new_trie_node))

      # Loop through search_map for completed matches at span_end.
      for search_state in search_map[span_end]:
        # Get the ParserRule(s) associated with the particular Trie path.
        matched_rules = search_state.trie_node.values
        if not matched_rules:
          continue

        for rule in matched_rules:
          # Given the ParserRule and anchored nonterminal positions, generate
          # new chart entries and add chart.
          if len(search_state.anchored_nonterminals) == 1:
            # Matched rule contains 1 non-terminal.
            for child in chart.get_from_key(
                *search_state.anchored_nonterminals[0]):
              chart.add(span_begin, span_end, rule, [child])
          elif len(search_state.anchored_nonterminals) == 2:
            # Matched rule contains 2 non-terminals.
            for child_0 in chart.get_from_key(
                *search_state.anchored_nonterminals[0]):
              for child_1 in chart.get_from_key(
                  *search_state.anchored_nonterminals[1]):
                chart.add(span_begin, span_end, rule, [child_0, child_1])
          elif len(search_state.anchored_nonterminals) > 2:
            raise ValueError
          else:
            # Matched rule contains 0 non-terminals.
            chart.add(span_begin, span_end, rule, [])

      for nt in nonterminals:
        chart.postprocess(span_begin, span_end, nt)

      if verbose:
        for nt in nonterminals:
          cell = chart.get_from_key(span_begin, span_end, nt)
          if cell:
            print("Populated (%s,%s): %s - %s" %
                  (span_begin, span_end, nt, cell))

  # Return completed parses.
  return chart.get_from_key(0, input_len, start_idx)
