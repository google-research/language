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

There are two equivalent implementations, with the Trie variant being a bit
more complicated but faster for most applications, especially for longer inputs.
"""

from language.compgen.nqg.common.cky import cky_utils
from language.compgen.nqg.common.cky import trie_utils


def parse(input_ids,
          rules,
          nonterminals,
          start_idx,
          populate_fn,
          postprocess_fn,
          use_trie=True,
          verbose=False):
  """Run bottom up parser.

  Let T be an arbitrary type for chart entries, specified by the return type
  of populate_fn. Examples for T are simple types that simply indicate presenece
  of a parse for a given span, or more complex structures that represent
  parse forests.

  Args:
    input_ids: List of integers corresponding to idx of terminal CFGSymbols in
      rules.
    rules: A list of CFGRule instances.
    nonterminals: Collection of CFGSymbol objects for possible non-terminals.
    start_idx: Index of non-terminal that is start symbol.
    populate_fn: A function that takes: (span_begin (Interger), span_end
      (Integer), parser_rule (CFGRule), substitutions (List of T)) and returns
      an object of type T, which can be any type. This object is added to the
      chart. Depending on what information is desired about completed parses, T
      can be anything from a simple count to a complex parse forest object.
    postprocess_fn: A function that takes and returns a list of T. This function
      post-processes each cell after it has been populated. This function is
      useful for pruning the chart, or merging equivalent entries. Ignored if
      None.
    use_trie: Whether to use the Trie-based parsing algorithm.
    verbose: Print debug logging if True.

  Returns:
    A list of T.
  """
  if use_trie:
    return trie_utils.parse(input_ids, rules, nonterminals, start_idx,
                            populate_fn, postprocess_fn, verbose)
  else:
    return cky_utils.parse(input_ids, rules, nonterminals, start_idx,
                           populate_fn, postprocess_fn, verbose)
