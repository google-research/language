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
"""Data classes used by the search agent."""

from typing import Any, Dict, List, Optional, Sequence, Tuple

from absl import logging
import json
import nltk
import re
import unicodedata

from language.search_agents import environment_pb2
from language.search_agents.muzero import state_tree
from language.search_agents.muzero import utils

from google.protobuf import text_format
from muzero import core as mzcore


class HistoryEntry:
  """History kept for a single step."""

  def __init__(self, query: str,
               original_query: environment_pb2.GetQueryResponse,
               documents: List[environment_pb2.Document]):
    self.query = query
    self.original_query = original_query

    if not documents:
      logging.info('Couldn\'t retrieve any documents for query: %s .',
                   self.query)
    self.documents = documents

  @classmethod
  def from_json(cls, json_dict: Dict[str, Any]):
    return cls(
        query=json_dict['query'],
        original_query=environment_pb2.GetQueryResponse.FromString(
            json_dict['original_query'].encode('latin-1')),
        documents=list(
            environment_pb2.GetDocumentsResponse.FromString(
                json_dict['document_response'].encode('latin-1')).documents))

  def json_repr(self) -> Dict[str, Any]:
    return {
        'query':
            self.query,
        'documents': [
            text_format.MessageToString(document) for document in self.documents
        ],
    }

  @staticmethod
  def remove_diacritics(s: str) -> str:
    """Removes accents and such."""
    normalized = unicodedata.normalize('NFKD', s)
    res = u''.join([c for c in normalized if not unicodedata.combining(c)])
    res = unicodedata.normalize('NFC', res)
    if len(res) != len(s):
      raise ValueError('Error removing diacritics: length changed.')
    return res

  @staticmethod
  def find_substr(text: str, substr: str) -> Tuple[str, int]:
    """Locates the substr in the text independent of surface variations.

    Args:
      text: A string that might contain the substr.
      substr: The substr we try to find in the text. If the exact string can't
        be found in the text, the search is repeated ignoring diacritics in the
        text. If the substr contains question marks, these can match any
        character.

    Returns:
      A tuple (found_substr, substr_begin):
      found_substr: A string containing the surface from of the substr as
        found in the text (i.e. possibly with added diacritics, resolved '?'.
      substr_begin: Integer indicating the start position of the substr in the
        text.

    """
    substr_begin = text.find(substr)
    if substr_begin < 0:
      # Try finding the substr in the text without diacritics.
      text_without_diacritics = text
      try:
        text_without_diacritics = HistoryEntry.remove_diacritics(text)
      except ValueError:
        # length changed
        pass
      substr_begin = text_without_diacritics.find(substr)
    if substr_begin < 0:
      regexp_string = substr.replace('?', 'UNKNOWNCHARACTER')
      regexp_string = re.escape(regexp_string)
      regexp_string = regexp_string.replace('UNKNOWNCHARACTER', '.')
      for m in re.finditer(regexp_string, text):
        substr_begin = m.start()
        break
      if substr_begin < 0:
        # match the ? but also remove diacritics
        for m in re.finditer(regexp_string, text_without_diacritics):
          substr_begin = m.start()
          break
    if substr_begin >= 0:  # recover the substr as in the text
      substr = text[substr_begin:substr_begin + len(substr)]
    return substr, substr_begin

  @staticmethod
  def get_window_around_substr(text: str, substr: str,
                               number_of_words: int) -> Tuple[str, str]:
    """Produces a span of limited length from `text` centered around `substr`.

    Args:
      text: The document context containing the substr.
      substr: The string to be found in the text.
      number_of_words: Maximum number of tokens for returned span.

    Returns:
      A tuple of strings (found_substr, context) where `found_substr` is the
      surface form of `substr` that was found in the text and `context` is a
      subspan of the text centered on `substr`.
    """

    def normalize_whitespace(s: str) -> str:
      return ' '.join(s.split())

    text = normalize_whitespace(text)
    substr = normalize_whitespace(substr)
    found_substr, substr_begin = HistoryEntry.find_substr(text, substr)

    if substr_begin < 0:
      return substr, ' '.join(text.split()[:number_of_words])

    before = text[:substr_begin].split()
    after = text[substr_begin + len(found_substr):].split()
    substr_tokens = found_substr.split()

    max_context_tokens = number_of_words - len(substr_tokens)
    # here comes the clever bit:
    num_before_tokens = min(
        max(max_context_tokens // 2, max_context_tokens - len(after)),
        len(before))
    num_after_tokens = min(
        max(max_context_tokens // 2, max_context_tokens - len(before)),
        len(after))

    res = []
    if num_before_tokens > 0:
      res += before[-num_before_tokens:]
    res += substr_tokens
    if num_after_tokens > 0:
      res += after[:num_after_tokens]

    return found_substr, ' '.join(res)

  def __str__(self) -> str:
    result_jsons = []
    for doc in self.documents:
      result_jsons.append(
          json.dumps({
              'title': doc.title,
              'content': doc.content,
              'score': doc.answer.mr_score,
              'answer': doc.answer.answer,
          }))
    return json.dumps({
        'query': self.query,
        'results': result_jsons,
    })

  def __repr__(self) -> str:
    return self.__str__()


class EnvState:
  """Full environment state."""

  def __init__(self,
               original_query: environment_pb2.GetQueryResponse,
               history: Optional[List[HistoryEntry]] = None,
               tree: Optional[state_tree.NQStateTree] = None,
               k: Optional[int] = None):
    self.original_query = original_query
    self.tree = tree
    self.k = k

    self.history = []

    if history:
      for history_entry in history:
        self.add_history_entry(history_entry)

    # Target documents retrieved using the gold_answer. This is only set
    # if common_flags.RELEVANCE_FEEDBACK_RESTRICT is true.
    self.target_documents = []

  def add_history_entry(self, history_entry: HistoryEntry):
    self.history.append(history_entry)

  def json_repr(self) -> Dict[str, Any]:
    return {
        'history': [step.json_repr() for step in self.history],
    }

  def sorted_unique_documents(self, step: int):
    """Unique documents seen until 'step' ordered by mr score."""
    documents = []
    seen_contents = set()

    for entry in self.history[:step]:
      for doc in entry.documents:
        if doc.content not in seen_contents:
          documents.append(doc)
          seen_contents.add(doc.content)

    # sort the documents according to the MR score of its extracted answer
    sorted_documents = sorted(
        documents,
        key=lambda doc: doc.answer.mr_score if doc.answer else -15.,
        reverse=True)
    return sorted_documents

  def retrieval_score(self,
                      k: int,
                      documents_list: List[environment_pb2.Document],
                      gold_answer: List[str],
                      score_type: str = 'dcg') -> float:
    """Return the retrieval score @k for documents_list."""
    # compute the relevance as a binary score that is 1 if any of the gold
    # answers is present at the document.
    relevances = [
        float(utils.gold_answer_present(doc.content, gold_answer))
        for doc in documents_list
    ]

    # Since we want to compute the relevance @k we need to cut the relevance
    # list at k
    relevances = relevances[:k]

    if score_type == 'dcg':
      return utils.dcg_score(relevances)
    elif score_type == 'ndcg':
      return utils.ndcg_score(relevances)
    elif score_type == 'mrr':
      return utils.mrr_score(relevances)
    else:
      raise NotImplementedError(
          f'Score type {score_type} is not yet implemented.')

  def score(self,
            identifier: str,
            documents_list: List[environment_pb2.Document],
            gold_answer: Optional[List[str]] = None,
            **kwargs):
    """Return the answer score for documents_list."""
    if gold_answer is None:
      gold_answer = list(self.original_query.gold_answer)

    if not documents_list or not gold_answer:
      return 0.

    if identifier == 'em':
      return utils.compute_em(documents_list[0].answer.answer, gold_answer)
    elif identifier == 'em_at_k':
      assert 'k' in kwargs, f'Provide "k" for computing the {identifier} score.'
      return max([
          utils.compute_em(document.answer.answer, gold_answer)
          for document in documents_list[:kwargs['k']]
      ])
    elif identifier == 'f1':
      return utils.compute_f1(documents_list[0].answer.answer, gold_answer)
    elif identifier in ('dcg', 'mrr', 'ndcg'):
      assert 'k' in kwargs, f'Provide "k" for computing the {identifier} score.'
      return self.retrieval_score(
          documents_list=documents_list,
          gold_answer=gold_answer,
          score_type=identifier,
          **kwargs)
    elif identifier == 'recall_at_k':
      assert 'k' in kwargs, f'Provide "k" for computing the {identifier} score.'
      return max([
          float(utils.gold_answer_present(doc.content, gold_answer))
          for doc in documents_list[:kwargs['k']]
      ])
    else:
      raise NotImplementedError(f'Score "{identifier}" is not yet implemented.')

  @property
  def num_completed_requests(self):
    return len(self.history)

  @property
  def total(self):
    return self.original_query.total


class VisNode:
  """TreeNode for the MCTS visualization."""

  def __init__(self,
               children=None,
               visit_count=-1,
               prior=0,
               value_sum=0,
               reward=0,
               qvalue=-10,
               stack=None,
               leaves=None):
    self.children = children or {}
    self.visit_count = visit_count
    self.prior = prior
    self.value_sum = value_sum
    self.reward = reward
    self.qvalue = qvalue
    self.stack = stack or []
    self.leaves = leaves or []

  @classmethod
  def from_core_node(cls, node: mzcore.Node, stack=None, leaves=None):
    return cls(
        children=node.children,
        visit_count=node.visit_count,
        prior=node.prior,
        value_sum=node.value_sum,
        reward=node.reward,
        qvalue=node.qvalue(),
        stack=stack,
        leaves=leaves)

  @classmethod
  def from_root_with_history(cls, node: mzcore.Node, history, grammar):
    """Make a VisNode frmo a root."""

    def updated_stack_leaves(stack, leaves, action):
      if not stack:
        stack = [grammar.start()]
      new_stack = stack[:-1].copy()
      new_leaves = leaves.copy()
      for new_node in grammar.productions()[action].rhs()[::-1]:
        if isinstance(new_node, nltk.grammar.Nonterminal):
          new_stack.append(new_node)
        else:
          new_leaves.append(new_node)
      return new_stack, new_leaves

    def reccast(node, stack, leaves):
      vis_node = VisNode.from_core_node(node, stack, leaves)
      if not stack:
        vis_node.children = {}
        return vis_node
      for action_idx, child in node.children.items():
        new_stack, new_leaves = updated_stack_leaves(stack, leaves, action_idx)
        vis_node.children[action_idx] = reccast(child, new_stack, new_leaves)
      return vis_node

    def rec_hist_cast(idx, stack, leaves, visit_count):
      node = VisNode(stack=stack, leaves=leaves, visit_count=visit_count)
      if idx == len(history):
        return node
      new_stack, new_leaves = updated_stack_leaves(stack, leaves, history[idx])
      child_node = rec_hist_cast(idx + 1, new_stack, new_leaves, visit_count)
      node.children = {history[idx]: child_node}
      return node

    def get_last_hist_node(node):
      if not node or not node.children:
        return node
      return get_last_hist_node([v for v in node.children.values()][0])

    if history:
      root = rec_hist_cast(
          idx=0,
          stack=[grammar.start()],
          leaves=[],
          visit_count=node.visit_count)
      final_node_hist = get_last_hist_node(root)
      mcts_stack = final_node_hist.stack
      mcts_leaves = final_node_hist.leaves
      mcts_root = reccast(node, mcts_stack, mcts_leaves)
      final_node_hist.children = mcts_root.children
    else:
      root = reccast(node, [grammar.start()], [])

    return root


def visualize_mcts(root: mzcore.Node, history: Sequence[HistoryEntry],
                   grammar: state_tree.NQCFG, min_visit_count: int) -> str:
  """Visualize the MCTS corresponding to `root` and `history`.

  The returned string can be visualized using the `forest` LaTex package
  (https://ctan.org/pkg/forest?lang=en).

  Args:
    root:  Root node for the Monte-Carlo Tree search.
    history:  History from the Agent's state.
    grammar:  Underlying grammar.
    min_visit_count:  Only visualize a node if it got visited at least that many
      times.  If set to 0, all children - even unexplored - will be visualized,
      which might become unwieldy.  Setting it to 1 will visualize all paths
      actually taken.

  Returns:
    A `forest`-package compatible string representing the Monte-Carlo Tree
    search tree.
  """

  edge_str_template = (
      ', edge '
      'label={{node[midway,fill=white,font=\\scriptsize,align=right]{{\\textcolor{{red}}{{{}}}'
      ' \\tiny{{\\textcolor{{blue}}{{{:.2f}}}}} \\\\\\ '
      '\\tiny{{\\textcolor{{magenta}}{{{:.2f}}}}}}}}}')

  def to_latex_tree(node: VisNode) -> str:
    edge_str = edge_str_template.format(
        node.visit_count, node.reward,
        node.qvalue) if node.visit_count > 0 else ''
    label = '{} : {}'.format(
        ' '.join(
            [str(nonterminal).replace('_', '') for nonterminal in node.stack]),
        ' '.join([
            str(leaf).replace('[', '{[').replace(']', ']}')
            for leaf in node.leaves
        ]))
    if not node.children or all([
        child.visit_count < min_visit_count for child in node.children.values()
    ]) or not node.stack:
      return '[{}{} ]'.format(label, edge_str)
    children_strs = []
    for child in node.children.values():
      if child.visit_count >= min_visit_count:
        children_strs.append(to_latex_tree(child))
    latex_str = '[{}{} {} ]'.format(label, edge_str,
                                    ' '.join(children_strs)).replace(
                                        '##', '@@')
    return latex_str

  root = VisNode.from_root_with_history(root, history, grammar)
  tree_str = to_latex_tree(root)

  return tree_str


class InvalidActionError(Exception):
  pass
