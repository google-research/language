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
"""Utilities for caching query retrievals."""
import dataclasses
import json


from absl import logging
from language.casper.utils import data_types
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text  # pylint: disable=unused-import


# A generic type for the examples loaded from a JSONL file.
Example = data_types.RawExample

_BERT_PREPROCESSOR_HUB = (
    "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
_BERT_HUBS = {
    "base": "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3",
    "large": "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/3",
}
_USE_HUBS = {
    "base": "https://tfhub.dev/google/universal-sentence-encoder/4",
    "large": "https://tfhub.dev/google/universal-sentence-encoder-large/5",
}


def read_jsonl(jsonl_filenames: Union[str, Iterable[str]]) -> Iterator[Example]:
  """Yields entries from the JSONL files."""
  if isinstance(jsonl_filenames, str):
    jsonl_filenames = [jsonl_filenames]
  for filename in jsonl_filenames:
    with tf.io.gfile.GFile(filename) as fin:
      for line in fin:
        entry = json.loads(line)
        yield entry


class NeighborFilter:
  """Interface for a filter that accepts or rejects the retrieved neighbors."""

  def check_neighbor(self, input_ex: Example, neighbor_ex: Example,
                     state: Any) -> bool:
    """Returns whether the neighbor is usable.

    Optionally, a state can be maintained (e.g., to count how many neighbors so
    far belong to each intent, for crowding purposes). The state will be given
    as an argument.

    Args:
      input_ex: Input example.
      neighbor_ex: Neighbor example.
      state: The state object. Will be None for the first neighbor.

    Returns:
      True if the neighbor should be accepted and False otherwise.
    """
    raise NotImplementedError

  def accept_neighbor(self, neighbor_ex: Example, state: Any) -> Any:
    """Returns a new state after accepting the neighbor.

    The default behavior is to return the old state unchanged.

    Args:
      neighbor_ex: Neighbor example.
      state: The state object. Will be None for the first neighbor.

    Returns:
      The new state.
    """
    del neighbor_ex
    return state


class SimpleNeighborFilter(NeighborFilter):
  """A filter that only checks if the input and neighbor have different IDs."""

  def check_neighbor(self, input_ex: Example, neighbor_ex: Example,
                     state: Any) -> bool:
    return input_ex["hashed_id"] != neighbor_ex["hashed_id"]


class CrowdedNeighborFilter(NeighborFilter):
  """A filter that performs crowding (limit the max number of items per key).

  The state is a dict counting the number of neighbors so far for each key.
  """

  def __init__(self,
               example_to_crowding_key: Callable[[Example], str],
               max_per_crowding_key: int = 1):
    """Initializes the CrowdedNeighborFilter.

    Args:
      example_to_crowding_key: A function mapping Example to a crowding key.
      max_per_crowding_key: Maximum number of neighbors per crowding key.
    """
    self._example_to_crowding_key = example_to_crowding_key
    self._max_per_crowding_key = max_per_crowding_key

  def check_neighbor(self, input_ex: Example, neighbor_ex: Example,
                     state: Any) -> bool:
    """Returns whether the neighbor is usable."""
    if input_ex["hashed_id"] == neighbor_ex["hashed_id"]:
      return False
    key = self._example_to_crowding_key(neighbor_ex)
    return state is None or state.get(key, 0) < self._max_per_crowding_key

  def accept_neighbor(self, neighbor_ex: Example, state: Any) -> Any:
    """Returns the new state after accepting the neighbor."""
    # Make a copy of the state
    state = dict(state) if state else {}
    key = self._example_to_crowding_key(neighbor_ex)
    state[key] = state.get(key, 0) + 1
    return state


class PropertyMatchNeighborFilter(NeighborFilter):
  """Filters for neighbors with the same specified property as the input.

  For example, this filter can be used to do oracle retrieval (property = frame
  of the output logical form) or in-domain retrieval (property = domain).
  """

  def __init__(self, example_to_property: Callable[[Example], Any]):
    """Initializes the PropertyMatchNeighborFilter.

    Args:
      example_to_property: A function mapping Example to the property value.
    """
    self._example_to_property = example_to_property

  def check_neighbor(self, input_ex: Example, neighbor_ex: Example,
                     state: Any) -> bool:
    """Returns whether the neighbor is usable."""
    if input_ex["hashed_id"] == neighbor_ex["hashed_id"]:
      return False
    input_property = self._example_to_property(input_ex)
    neighbor_property = self._example_to_property(neighbor_ex)
    return input_property == neighbor_property


class ChainedNeighborFilter(NeighborFilter):
  """Combines multiple NeighborFilters."""

  def __init__(self, neighbor_filters: Sequence[NeighborFilter]):
    self._neighbor_filters = neighbor_filters

  def check_neighbor(self, input_ex: Example, neighbor_ex: Example,
                     state: Any) -> bool:
    if state is None:
      state = [None] * len(self._neighbor_filters)
    for i, neighbor_filter in enumerate(self._neighbor_filters):
      if not neighbor_filter.check_neighbor(input_ex, neighbor_ex, state[i]):
        return False
    return True

  def accept_neighbor(self, neighbor_ex: Example, state: Any) -> Any:
    # Make a copy of the state
    state = list(state) if state else [None] * len(self._neighbor_filters)
    for i, neighbor_filter in enumerate(self._neighbor_filters):
      state[i] = neighbor_filter.accept_neighbor(neighbor_ex, state[i])
    return state


@dataclasses.dataclass
class RetrievalResult:
  example: Example
  neighbor_ids: List[int]
  neighbor_distances: List[float]


class Retriever:
  """Interface for a retriever."""

  def __init__(self, batch_size: int = 32, log_every: int = 1000):
    """Initializes a retriever.

    Args:
      batch_size: Batch size for the batched retrieval.
      log_every: Logging frequency.
    """
    self._batch_size = batch_size
    self._log_every = log_every

  def get_index_entry(self, index_id: int) -> Example:
    """Returns the index entry for the specified id."""
    raise NotImplementedError

  def retrieve_batch(self, examples: List[Example]) -> List[RetrievalResult]:
    """Returns the neighbors and distances from a batch of examples.

    Args:
      examples: A batch of examples.

    Returns:
      a list of RetrievalResults, with the same length as `examples`.
    """
    raise NotImplementedError

  def retrieve_all(self,
                   examples: Iterable[Example]) -> Iterator[RetrievalResult]:
    """Batches the examples, retrieves the neighbors, and yields the results."""
    batch = []
    for example in examples:
      batch.append(example)
      if len(batch) == self._batch_size:
        for result in self.retrieve_batch(batch):
          yield result
        batch = []
    # For the final batch
    if batch:
      for result in self.retrieve_batch(batch):
        yield result

  def dump_neighbors(self,
                     examples: Iterable[Example],
                     outfile: str,
                     neighbor_filter: NeighborFilter,
                     max_neighbors: int = 100) -> None:
    """Dumps the neighbors to a JSONL file.

    Args:
      examples: A stream of examples
      outfile: JSONL filename.
      neighbor_filter: A NeighborFilter.
      max_neighbors: Maximum number of neighbors per example (int).
    """
    with tf.io.gfile.GFile(outfile, "w") as fout:
      count = 0
      for result in self.retrieve_all(examples):
        if count % self._log_every == 0:
          logging.info("Processing %d", count)
        input_ex = result.example
        neighbor_hashed_ids = []
        neighbor_distances = []
        filter_state = None
        for neighbor_id, neighbor_distance in zip(result.neighbor_ids,
                                                  result.neighbor_distances):
          neighbor_ex = self.get_index_entry(neighbor_id)
          is_ok = neighbor_filter.check_neighbor(input_ex, neighbor_ex,
                                                 filter_state)
          if is_ok:
            filter_state = neighbor_filter.accept_neighbor(
                neighbor_ex, filter_state)
            neighbor_hashed_ids.append(neighbor_ex["hashed_id"])
            neighbor_distances.append(round(neighbor_distance, 3))
          if len(neighbor_hashed_ids) >= max_neighbors:
            break
        output = dict(
            input_ex,
            exemplars={
                "hashed_ids": neighbor_hashed_ids,
                "distances": neighbor_distances,
            })
        print(json.dumps(output), file=fout)
        count += 1
    logging.info("Wrote %d entries to %s", count, outfile)


class EmbeddingBasedRetriever(Retriever):
  """Interface for a retriever with distance = 1 - cosine between embeddings."""

  def __init__(self,
               index_exs: List[Example],
               batch_size: int = 32,
               log_every: int = 1000):
    super().__init__(batch_size=batch_size, log_every=log_every)
    self._index_exs = index_exs
    self._embed_index()

  def embed_batch(self, examples: List[Example]) -> np.ndarray:
    """Embeds a batch of examples.

    Args:
      examples: A batch of N examples.

    Returns:
      numpy array of shape (N, D) where D is the embedding size.
    """
    raise NotImplementedError

  def _embed_index(self) -> None:
    """Embeds the entire retrieval index."""
    batch = []
    results = []
    for i, example in enumerate(self._index_exs):
      if i % self._log_every == 0:
        logging.info("Processed %d / %d examples", i, len(self._index_exs))
      batch.append(example)
      if len(batch) == self._batch_size:
        results.append(self.embed_batch(batch))
        batch = []
    # Final batch
    if batch:
      results.append(self.embed_batch(batch))
    self._index_embs = np.vstack(results)

  def get_index_entry(self, index_id: int) -> Example:
    return self._index_exs[index_id]

  def retrieve_batch(self, examples: List[Example]) -> List[RetrievalResult]:
    if self._index_embs is None:
      raise RuntimeError("embed_index must be called first.")
    input_embs = self.embed_batch(examples)
    batch_distances = 1. - np.matmul(input_embs, self._index_embs.T)
    batch_sorted_ids = np.argsort(batch_distances, axis=1)
    batch_sorted_distances = np.take_along_axis(
        batch_distances, batch_sorted_ids, axis=1)
    results = []
    for example, ex_sorted_ids, ex_sorted_distances in zip(
        examples, batch_sorted_ids, batch_sorted_distances):
      results.append(
          RetrievalResult(example, ex_sorted_ids.tolist(),
                          ex_sorted_distances.tolist()))
    return results


class BertRetriever(EmbeddingBasedRetriever):
  """A retriever that uses BERT to embed queries."""

  def __init__(self,
               index_exs: List[Example],
               bert_size: str = "base",
               embed_method: str = "cls",
               batch_size: int = 32,
               log_every: int = 1000):
    self._preprocessor = hub.load(_BERT_PREPROCESSOR_HUB)
    self._bert = hub.load(_BERT_HUBS[bert_size])
    self._embed_method = embed_method
    super().__init__(index_exs, batch_size=batch_size, log_every=log_every)

  def embed_batch(self, examples: List[Example]) -> np.ndarray:
    batch = [example["orig_query"] for example in examples]
    bert_inputs = self._preprocessor(tf.constant(batch))
    bert_outputs = self._bert(bert_inputs)
    if self._embed_method == "pooled":
      # Use the default BERT output (CLS embedding + UNTUNED projection)
      # embs: [batch_size, embed_size]
      embs = bert_outputs["pooled_output"].numpy()
    elif self._embed_method == "cls":
      # [CLS] embedding (= embedding of the token at index 0)
      seq_embs = bert_outputs["sequence_output"].numpy()
      embs = seq_embs[:, 0, :]
    elif self._embed_method == "avg":
      # average embedding (compute the sum here; we will later normalize it)
      # The embeddings of [CLS] (101) and [SEP] (102) are not included.
      seq_embs = bert_outputs["sequence_output"].numpy()
      tok_ids = bert_inputs["input_word_ids"].numpy()
      mask = (tok_ids != 101) * (tok_ids != 102) * (tok_ids != 0) * 1
      embs = np.sum(mask[:, :, np.newaxis] * seq_embs, axis=1)
    else:
      raise ValueError("Unknown embed_method " + self._embed_method)
    # Normalize
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    return embs


class USERetriever(EmbeddingBasedRetriever):
  """A retriever that uses Univeral Sentence Encoder to embed queries."""

  def __init__(self,
               index_exs: List[Example],
               use_size: str = "base",
               batch_size: int = 32,
               log_every: int = 1000):
    self._use = hub.load(_USE_HUBS[use_size])
    super().__init__(index_exs, batch_size=batch_size, log_every=log_every)

  def embed_batch(self, examples: List[Example]) -> np.ndarray:
    batch = [example["orig_query"] for example in examples]
    embs = self._use(batch).numpy()
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    return embs
