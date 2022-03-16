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
"""CFG parser for non-binarized grammars.

The parser uses callbacks so that it can be flexibly extended to various
use cases, such as QCFG parsing.

Implements CFG parsing using a Trie data structure to index rules. This
implementation supports non-binarized grammars  with rules containing
any number of non-terminals. For each span, rather than enumerating every
possible sub-span for up to the maximum number of non-terminals, the algorithm
iterates across the span left-to-right and attempts to match rules stored in a
Trie.
"""

import collections
import itertools

from language.compgen.csl.cky import cfg_rule


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
    self.key_map[(span_begin, span_end, nonterminal)].extend(entry)
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


def parse_symbols(input_symbols,
                  rules,
                  nonterminals,
                  start_idx_set,
                  populate_fn,
                  postprocess_fn,
                  max_single_nt_applications=1,
                  verbose=False):
  """Run bottom up parser.

  Let T be an arbitrary type for chart entries, specified by the return type
  of populate_fn. Examples for T are simple types that simply indicate presenece
  of a parse for a given span, or more complex structures that represent
  parse forests.

  Args:
    input_symbols: List of CFGSymbols in rules.
    rules: A list of CFGRule instances.
    nonterminals: Collection of CFGSymbol objects for possible non-terminals.
    start_idx_set: A set of index of non-terminal that is start symbol.
    populate_fn: A function that takes: (span_begin (Interger), span_end
      (Integer), parser_rule (CFGRule), substitutions (List of T)) and returns a
      list of objects of type T, which can be any type. These objects are added
      to the chart. Depending on what information is desired about completed
      parses, T can be anything from a simple count to a complex parse forest
      object.
    postprocess_fn: A function that takes and returns a list of T. This function
      post-processes each cell after it has been populated. This function is
      useful for pruning the chart, or merging equivalent entries. Ignored if
      None.
    max_single_nt_applications: The maximum number of times a rule where the
      RHS is a single nonterminal symbol can be applied consecutively.
    verbose: Print debug logging if True.

  Returns:
    A list of T.
  """
  input_len = len(input_symbols)

  # Initialize the empty chart.
  chart = Chart(populate_fn, postprocess_fn)

  # Initialize Trie of rules.
  trie_root = TrieNode()
  max_num_nts = 0
  for rule in rules:
    add_rule_to_trie(trie_root, rule)
    max_num_nts = max(max_num_nts, cfg_rule.get_num_nts(rule.rhs))

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
          if len(search_state.anchored_nonterminals) < max_num_nts:
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
          children_list = []
          for anchored_nt in search_state.anchored_nonterminals:
            children = chart.get_from_key(*anchored_nt)
            children_list.append(children)
          for children in itertools.product(*children_list):
            chart.add(span_begin, span_end, rule, children)

      for nt in nonterminals:
        chart.postprocess(span_begin, span_end, nt)

      # Optionally apply rule where RHS is a single NT.
      for _ in range(max_single_nt_applications):
        for nt in nonterminals:
          # Copy cell since we are mutating it during iteration below.
          cell = chart.get_from_key(span_begin, span_end, nt).copy()
          nt_symbol = cfg_rule.CFGSymbol(nt, cfg_rule.NON_TERMINAL)
          child = trie_root.maybe_get_child(nt_symbol)
          if child:
            single_nt_rules = child.values
            for rule in single_nt_rules:
              for node in cell:
                chart.add(span_begin, span_end, rule, [node])
          chart.postprocess(span_begin, span_end, nt)

      if verbose:
        for nt in nonterminals:
          cell = chart.get_from_key(span_begin, span_end, nt)
          if cell:
            print("Populated (%s,%s): %s - %s" %
                  (span_begin, span_end, nt, cell))

  # Return completed parses.
  parses = []
  for start_idx in start_idx_set:
    parses.extend(chart.get_from_key(0, input_len, start_idx))
  return parses


def parse(input_ids,
          rules,
          nonterminals,
          start_idx_set,
          populate_fn,
          postprocess_fn,
          max_single_nt_applications=1,
          verbose=False):
  """Run bottom up parser where all inputs are terminals."""
  input_symbols = tuple(
      [cfg_rule.CFGSymbol(idx, cfg_rule.TERMINAL) for idx in input_ids])
  return parse_symbols(
      input_symbols,
      rules,
      nonterminals,
      start_idx_set,
      populate_fn,
      postprocess_fn,
      max_single_nt_applications=max_single_nt_applications,
      verbose=verbose)
