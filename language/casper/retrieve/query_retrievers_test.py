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
"""Tests for query_retrievers."""

from absl.testing import absltest
from language.casper.retrieve import query_retrievers
import numpy as np


class NeighborFilterTest(absltest.TestCase):
  """Tests the subclasses of NeighborFilter."""

  def setUp(self):
    """Creates examples for the retrieval index."""
    super().setUp()
    self._examples = [
        {
            "hashed_id": "a0",
            "domain": "home",
            "intent": "IN:CREATE_ALARM"
        },
        {
            "hashed_id": "a1",
            "domain": "office",
            "intent": "IN:CREATE_ALARM"
        },
        {
            "hashed_id": "a2",
            "domain": "home",
            "intent": "IN:CREATE_ALARM"
        },
        {
            "hashed_id": "a3",
            "domain": "home",
            "intent": "IN:CREATE_ALARM"
        },
        {
            "hashed_id": "t4",
            "domain": "home",
            "intent": "IN:CREATE_TIMER"
        },
        {
            "hashed_id": "t5",
            "domain": "home",
            "intent": "IN:CREATE_TIMER"
        },
        {
            "hashed_id": "t6",
            "domain": "office",
            "intent": "IN:CREATE_TIMER"
        },
    ]

  def test_simple_neighbor_filter(self):
    """Tests SimpleNeighborFilter."""
    neighbor_filter = query_retrievers.SimpleNeighborFilter()
    state = None
    # Should reject when input == neighbor and accept otherwise
    for i in range(len(self._examples)):
      actual = neighbor_filter.check_neighbor(self._examples[0],
                                              self._examples[i], state)
      expected = (i != 0)
      self.assertIs(actual, expected, f"Incorrect result on example {i}")

  def test_crowded_neighbor_filter(self):
    """Tests CrowdedNeighborFilter."""
    neighbor_filter = query_retrievers.CrowdedNeighborFilter(
        example_to_crowding_key=lambda x: x["intent"], max_per_crowding_key=1)
    state = None
    # Should reject when input == neighbor and accept otherwise
    for i in range(len(self._examples)):
      actual = neighbor_filter.check_neighbor(self._examples[0],
                                              self._examples[i], state)
      expected = (i != 0)
      self.assertIs(actual, expected, f"Incorrect result on example {i}")
    # Should reject neighbors with the same intent as the accepted one.
    state = neighbor_filter.accept_neighbor(self._examples[4], state)
    for i in range(len(self._examples)):
      actual = neighbor_filter.check_neighbor(self._examples[0],
                                              self._examples[i], state)
      expected = (i not in (0, 4, 5, 6))
      self.assertIs(actual, expected, f"Incorrect result on example {i}")

  def test_property_match_neighbor_filter(self):
    """Tests PropertyMatchNeighborFilter."""
    neighbor_filter = query_retrievers.PropertyMatchNeighborFilter(
        example_to_property=lambda x: x["domain"])
    state = None
    # Should reject when input == neighbor or input.domain != neighbor.domain.
    for i in range(len(self._examples)):
      actual = neighbor_filter.check_neighbor(self._examples[0],
                                              self._examples[i], state)
      expected = (i not in (0, 1, 6))
      self.assertIs(actual, expected, f"Incorrect result on example {i}")

  def test_chained_neighbor_filter(self):
    """Tests ChainedNeighborFilter."""
    neighbor_filter = query_retrievers.ChainedNeighborFilter([
        query_retrievers.PropertyMatchNeighborFilter(
            example_to_property=lambda x: x["domain"]),
        query_retrievers.CrowdedNeighborFilter(
            example_to_crowding_key=lambda x: x["intent"],
            max_per_crowding_key=1)
    ])
    state = None
    # Should reject based on the two filters
    for i in range(len(self._examples)):
      actual = neighbor_filter.check_neighbor(self._examples[0],
                                              self._examples[i], state)
      expected = (i not in (0, 1, 6))
      self.assertIs(actual, expected, f"Incorrect result on example {i}")
    # Should reject neighbors with the same intent as the accepted one.
    state = neighbor_filter.accept_neighbor(self._examples[4], state)
    for i in range(len(self._examples)):
      actual = neighbor_filter.check_neighbor(self._examples[0],
                                              self._examples[i], state)
      expected = (i not in (0, 1, 4, 5, 6))
      self.assertIs(actual, expected, f"Incorrect result on example {i}")


class _MockEmbeddingBasedRetriever(query_retrievers.EmbeddingBasedRetriever):

  def embed_batch(self, examples):
    """Returns an embedding matrix with embedding size 51."""
    return np.random.rand(len(examples), 51)


class EmbeddingBasedRetrieverTest(absltest.TestCase):
  """Tests EmbeddingBasedRetriever."""

  def test_batching_no_remainder(self):
    """Tests if batching does not drop any example."""
    index_size, num_examples, batch_size = 32, 24, 8
    index = [{"hashed_id": str(i)} for i in range(index_size)]
    retriever = _MockEmbeddingBasedRetriever(index, batch_size=batch_size)
    examples = [{"hashed_id": str(i)} for i in range(num_examples)]
    results = list(retriever.retrieve_all(examples))
    self.assertLen(results, num_examples)
    for i, result in enumerate(results):
      self.assertEqual(result.example["hashed_id"], str(i))
      self.assertLen(result.neighbor_ids, index_size)
      self.assertLen(result.neighbor_distances, index_size)
      for j in range(1, index_size):
        # Higher-ranked neighbors should have smaller distances.
        self.assertGreaterEqual(result.neighbor_distances[j],
                                result.neighbor_distances[j - 1])

  def test_batching_has_remainder(self):
    """Tests if batching does not drop any example."""
    index_size, num_examples, batch_size = 37, 23, 8
    index = [{"hashed_id": str(i)} for i in range(index_size)]
    retriever = _MockEmbeddingBasedRetriever(index, batch_size=batch_size)
    examples = [{"hashed_id": str(i)} for i in range(num_examples)]
    results = list(retriever.retrieve_all(examples))
    self.assertLen(results, num_examples)
    for i, result in enumerate(results):
      self.assertEqual(result.example["hashed_id"], str(i))
      self.assertLen(result.neighbor_ids, index_size)
      self.assertLen(result.neighbor_distances, index_size)
      for j in range(1, index_size):
        # Higher-ranked neighbors should have smaller distances.
        self.assertGreaterEqual(result.neighbor_distances[j],
                                result.neighbor_distances[j - 1])


if __name__ == "__main__":
  absltest.main()
