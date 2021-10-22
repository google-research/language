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
"""Utility functions for the T5 agent."""

import dataclasses
import re
from typing import Dict, Sequence, Tuple

from language.search_agents import environment_pb2
from language.search_agents.muzero import types
from language.search_agents.muzero import utils


@dataclasses.dataclass(frozen=True, eq=True)
class Term:
  field: str
  term: str
  term_type: str


@dataclasses.dataclass
class EpisodeStep:
  query: str
  results: Sequence[environment_pb2.Document] = dataclasses.field(
      default_factory=list)
  reward: float = 0.0
  target: str = ""


def add_to_dict(documents: Sequence[environment_pb2.Document],
                all_docs: Dict[str, environment_pb2.Document]) -> None:
  for doc in documents:
    all_docs[doc.content] = doc


def get_aggregated_documents(all_docs: Dict[str, environment_pb2.Document],
                             max_num_aggregated_documents: int):
  ranked_docs = sorted(
      list(all_docs.values()),
      key=lambda doc: doc.answer.mr_score,
      reverse=True)
  return ranked_docs[:max_num_aggregated_documents]


def _term_to_str(term: Term, operator: str) -> str:
  if term.field and term.term:
    return f'{operator}({term.field}:"{utils.escape_for_lucene(term.term)}")'
  if term.term:
    return f"{utils.escape_for_lucene(term.term)}"
  return ""


def make_query(base: str, addition_terms: Sequence[Term],
               subtraction_terms: Sequence[Term],
               or_terms: Sequence[Term]) -> str:
  """Modifies a `base`-query with addition and subtraction terms.

  Args:
    base:  Base query, as is, i.e. without manual escaping nor operators.
    addition_terms:  Sequence of terms which will be added to `base` with the
      "+" operator.
    subtraction_terms:  Sequence of terms which will be added to `base` with the
      "-" operator.
    or_terms: Sequence of terms which will be added to `base` directly without
      any operator.

  Returns:
    The properly modified base-query which can be issued to lucene.
  """
  escaped_base = utils.escape_for_lucene(base)

  modifications = []
  for subtraction_term, addition_term, or_term in zip(subtraction_terms,
                                                      addition_terms, or_terms):
    modifications.append(" ".join(
        list(
            filter(None, (_term_to_str(subtraction_term, operator="-"),
                          _term_to_str(addition_term, operator="+"),
                          _term_to_str(or_term, operator=""))))))

  full_query = f'{escaped_base} {" ".join(modifications)}'
  return full_query.strip()


def state_from_documents(results: Sequence[environment_pb2.Document],
                         max_title_tokens: int, max_context_tokens: int) -> str:
  """Produces a string representation of the search results.

  Args:
    results: A number of documents to be included in the state
    max_title_tokens: Maximum number of tokens from the title per document.
    max_context_tokens: Maximum number of tokens from the context per document.

  Returns:
    state: A flat string representation of the documents.
  """
  state = []
  for result in results:
    answer, answer_with_context = types.HistoryEntry.get_window_around_substr(
        result.content, result.answer.answer, max_context_tokens)
    state.append(f"Answer: '{answer}'.")
    if max_title_tokens > 0:
      title = " ".join(result.title.split()[:max_title_tokens])
      state.append(f"Title: '{title}'.")
    state.append(f"Result: '{answer_with_context}'.")
  return " ".join(state)


def query_to_prompt(query: str, addition_terms: Sequence[Term],
                    subtraction_terms: Sequence[Term],
                    or_terms: Sequence[Term]) -> str:
  """Produces a plaintext form of the current query with all modifiers.

  This is not the form to be used with Lucence but part of the T5 input.

  Args:
    query: The original query.
    addition_terms: Forced terms.
    subtraction_terms: Prohibited terms.
    or_terms: Optional terms.

  Returns:
    query_with_modifiers as a flat string representation.
  """
  res = []
  if query.strip():
    res = [f"Query: '{query}'."]

  def has_terms(terms):
    for operator in terms:
      if operator.term:
        return True
    return False

  # The or-terms are all joined together with commas.
  if has_terms(or_terms):
    res.append("Should contain: " + ", ".join(
        f"'{operator.term}'" for operator in or_terms if operator.term) + ".")

  if has_terms(addition_terms):
    res.append(" ".join(
        f"{operator.field.capitalize()} must contain: '{operator.term}'."
        for operator in addition_terms
        if operator.term))

  if has_terms(subtraction_terms):
    res.append(" ".join(
        f"{operator.field.capitalize()} cannot contain: '{operator.term}'."
        for operator in subtraction_terms
        if operator.term))

  return " ".join(r for r in res if r)


def target_from_terms(addition_term: Term, subtraction_term: Term,
                      or_term: Term) -> str:
  return query_to_prompt("", [addition_term], [subtraction_term], [or_term])


def parse_t5_response(response: str) -> Tuple[bool, bool, Sequence[Term]]:
  """Parse the response of the T5 agent model.

  Args:
    response: The T5 response given as a string.

  Returns:
    A triple: (success, stop, terms).
    success indicates that the response was correctly parsed.
    stop indicates the stop action
    terms is a list of length 3 containing the newly added/subtracted/or-terms.
  """
  stop = response == "Stop."

  def parse_by_regexp(regexp_string: str, response: str) -> Term:
    field = ""
    term = ""
    for match in re.findall(regexp_string, response):
      if isinstance(match, tuple):
        field, term = match
      else:
        term = match
      break
    return Term(field.lower(), term, term_type="")

  # pylint: disable=g-complex-comprehension
  terms = [
      parse_by_regexp(regexp_string, response) for regexp_string in [
          r"(Contents|Title) must contain: '(.+?)'.",
          r"(Contents|Title) cannot contain: '(.+?)'.",
          r"Should contain: '(.+?)'."
      ]
  ]
  # pylint: enable=g-complex-comprehension

  success = stop
  for term in terms:
    success = success or bool(term.term)
  return success, stop, terms


def ndcg(documents: Sequence[environment_pb2.Document],
         answers: Sequence[str]) -> float:
  """NDCG metric."""
  relevances = [
      float(utils.gold_answer_present(doc.content, answers))
      for doc in documents
  ]
  return utils.ndcg_score(relevances)
