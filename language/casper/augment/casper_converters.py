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
r"""Converters that take cached retrievals and yield seq2seq examples."""
import collections
import dataclasses
import functools
import random
from typing import Any, Dict, Iterable, Iterator, Sequence

from absl import logging
from language.casper.augment import casper_formatters
from language.casper.utils import data_types
from language.casper.utils import sample_utils
from language.casper.utils import top_utils

RawExample = data_types.RawExample
AugmentedExample = data_types.AugmentedExample


def _get_frame(funcall: str, funcall_format: str) -> str:
  """Returns the frame (intent and slot labels) of the function call."""
  if funcall_format == "top":
    return top_utils.get_frame_top(funcall)
  else:
    raise ValueError(f"Unknown funcall_format: {funcall_format}")


_CONVERTER_CONFIG_ALIASES = {
    "n": "num_samples",
    "k": "max_exemplars",
    "p": "sample_prob",
}


@dataclasses.dataclass
class ConverterConfig:
  """Config for the ExampleConverters.

  Attrs:
    num_samples: The number of exemplar lists to generate.
    sampler: The sampler for sampling exemplars. Available choices are:
      - "uniform": Uniform sampling.
      - "geometric": Geometric sampling without replacement; exemplars with
        higher ranks have a higher chance of being sampled.
    max_exemplars: Maximum number of exemplars in each exemplar list.
    sample_prob: Probability for geometric sampling.
  """
  num_samples: int = 1
  sampler: str = "geometric"
  max_exemplars: int = 1
  sample_prob: float = 0.5

  @classmethod
  def from_dict(cls, converter_kwargs: Dict[str, Any]) -> "ConverterConfig":
    """Constructs a ConverterConfig from the given dict."""
    # Make a copy
    converter_kwargs = dict(converter_kwargs)
    # Resolve aliases
    for abbr_key, full_key in _CONVERTER_CONFIG_ALIASES.items():
      if abbr_key in converter_kwargs:
        converter_kwargs[full_key] = converter_kwargs[abbr_key]
        del converter_kwargs[abbr_key]
    return cls(**converter_kwargs)

  def get_sampler(self):
    """Returns the exemplar sampler based on the config."""
    if self.sampler == "uniform":
      return functools.partial(
          sample_utils.uniform_sample, max_num_items=self.max_exemplars)
    elif self.sampler == "geometric":
      return functools.partial(
          sample_utils.geometric_sample,
          max_num_items=self.max_exemplars,
          sample_prob=self.sample_prob)
    else:
      raise ValueError(f"Unknown sampler: {self.sampler}")


class BaseExampleConverter:
  """Abstract base class for example converters."""

  def __init__(self, retrieval_index: Iterable[RawExample], funcall_format: str,
               converter_config: ConverterConfig,
               formatter_config: casper_formatters.FormatterConfig):
    """Constructs a new example converter.

    Args:
      retrieval_index: The retrieval index.
      funcall_format: Format of the output function call or logical form.
      converter_config: A ConverterConfig object.
      formatter_config: A FormatterConfig object.
    """
    self._funcall_format = funcall_format
    self._converter_config = converter_config
    self._preprocess_example = functools.partial(
        casper_formatters.preprocess_example, funcall_format=funcall_format)
    self._augment_exemplars = functools.partial(
        casper_formatters.augment_exemplars,
        funcall_format=funcall_format,
        config=formatter_config)
    self._process_index(retrieval_index)
    # A Counter that can collect arbitrary statistics.
    self.stats = collections.Counter()

  def _process_index(self, retrieval_index: Iterable[RawExample]) -> None:
    """Preprocesses the retrieval index."""
    self._hashed_id_to_exemplar = {}
    self._frame_to_hashed_ids = {}
    self._hashed_id_to_frame = {}
    for example in retrieval_index:
      example = self._preprocess_example(example)
      hashed_id = example["hashed_id"]
      if hashed_id in self._hashed_id_to_exemplar:
        # Check for duplicates
        existing_entry = self._hashed_id_to_exemplar[hashed_id]
        if existing_entry["hashed_id"] != example["hashed_id"]:
          raise ValueError(f"Duplicated hashed ID: {hashed_id}")
      else:
        self._hashed_id_to_exemplar[hashed_id] = example
        frame = _get_frame(example["output_str"], self._funcall_format)
        self._hashed_id_to_frame[hashed_id] = frame
        self._frame_to_hashed_ids.setdefault(frame, []).append(hashed_id)
    logging.info("Read %d index entries with %d unique frames.",
                 len(self._hashed_id_to_exemplar),
                 len(self._frame_to_hashed_ids))
    # List of hashed IDs (for sampling)
    self._all_hashed_ids = sorted(self._hashed_id_to_exemplar)
    # List of frames (for sampling)
    self._all_frames = sorted(self._frame_to_hashed_ids)

  def verify_exemplars(self, example: RawExample) -> None:
    """Filters out an example's exemplars that are not in the index.

    Args:
      example: an Example. The "exemplars" field will be modified in-place.
    """
    if "exemplars" not in example:
      # No retrieval (for the query_only converter).
      return
    filtered_hashed_ids = []
    filtered_distances = []
    for hashed_id, distance in zip(example["exemplars"]["hashed_ids"],
                                   example["exemplars"]["distances"]):
      if hashed_id not in self._hashed_id_to_exemplar:
        logging.warn("Example %s: Exemplar hashed ID %s is not in the index.",
                     example["hashed_id"], hashed_id)
      else:
        filtered_hashed_ids.append(hashed_id)
        filtered_distances.append(distance)
    example["exemplars"]["hashed_ids"] = filtered_hashed_ids
    example["exemplars"]["distances"] = filtered_distances

  def convert(self, example: RawExample) -> Iterator[AugmentedExample]:
    """Takes the retrieval results of an example and yields seq2seq examples.

    Args:
      example: a RawExample.

    Yields:
      AugmentedExample, one for each seq2seq example.
    """
    example = self._preprocess_example(example)
    for hashed_ids in self._select_exemplars(example):
      exemplars = [
          self._hashed_id_to_exemplar[hashed_id] for hashed_id in hashed_ids
      ]
      input_str, output_str = self._augment_exemplars(example, exemplars)
      yield AugmentedExample(input_str, output_str)

  def _select_exemplars(self, example: RawExample) -> Iterator[Sequence[str]]:
    """Selects lists of exemplars to be augmented to the given example.

    This method should be overridden.

    Args:
      example: a preprocessed RawExample.

    Yields:
      Lists of hashed_ids of the selected exemplars. Each list will be used to
      create a retrieval-augmented example.
    """
    raise NotImplementedError


class QueryOnlyConverter(BaseExampleConverter):
  """Generates the example without using the retrievals."""

  def _select_exemplars(self, example: RawExample) -> Iterator[Sequence[str]]:
    """Yields a single empty list (no exemplars)."""
    for _ in range(self._converter_config.num_samples):
      yield []


class AddTopKConverter(BaseExampleConverter):
  """Adds the top K exemplars to the input query."""

  def _select_exemplars(self, example: RawExample) -> Iterator[Sequence[str]]:
    """Yields a single list containing the top `max_exemplars` exemplars."""
    exemplar_hashed_ids = example["exemplars"]["hashed_ids"]
    for _ in range(self._converter_config.num_samples):
      yield exemplar_hashed_ids[:self._converter_config.max_exemplars]


class AddSampledKConverter(BaseExampleConverter):
  """Adds K sampled exemplars to the input query."""

  def _select_exemplars(self, example: RawExample) -> Iterator[Sequence[str]]:
    """Yields `num_samples` lists with `max_exemplars` sampled exemplars."""
    sampler = self._converter_config.get_sampler()
    exemplar_hashed_ids = example["exemplars"]["hashed_ids"]
    for _ in range(self._converter_config.num_samples):
      yield sampler(exemplar_hashed_ids)


class AddOracleKConverter(AddSampledKConverter):
  """Adds K exemplars whose semantic frame matches the target output.

  Used for oracle and controllability experiments.
  """

  def _select_exemplars(self, example: RawExample) -> Iterator[Sequence[str]]:
    """Yields `num_samples` lists with `max_exemplars` oracle exemplars."""
    self.stats["num_examples"] += 1
    gold_frame = _get_frame(example["output_str"], self._funcall_format)
    candidate_hashed_ids = []

    # Find all retrieved exemplars with a matching frame
    for hashed_id in example["exemplars"]["hashed_ids"]:
      exemplar_frame = self._hashed_id_to_frame[hashed_id]
      if exemplar_frame == gold_frame:
        candidate_hashed_ids.append(hashed_id)
    if not candidate_hashed_ids:
      self.stats["no_match_in_retrieved"] += 1

    # Find all index entries with a matching frame
    extra_candidate_hashed_ids = []
    for hashed_id in self._frame_to_hashed_ids.get(gold_frame, []):
      if (hashed_id != example["hashed_id"] and
          hashed_id not in candidate_hashed_ids):
        extra_candidate_hashed_ids.append(hashed_id)
    if not extra_candidate_hashed_ids:
      self.stats["no_match_in_index"] += 1
    candidate_hashed_ids.extend(extra_candidate_hashed_ids)

    if not candidate_hashed_ids:
      return

    # Sample K exemplars
    sampler = self._converter_config.get_sampler()
    for _ in range(self._converter_config.num_samples):
      yield sampler(candidate_hashed_ids)


class AddAdversarialKConverter(AddSampledKConverter):
  """Adds K exemplars with the same frame but different from the target output.

  Used for parse guiding analysis.
  """
  # Try finding an adversarial frame this number of times before giving up.
  _MAX_TRIALS = 100

  def _select_exemplars(self, example: RawExample) -> Iterator[Sequence[str]]:
    """Yields `num_samples` lists with `max_exemplars` adversarial exemplars."""
    gold_frame = _get_frame(example["output_str"], self._funcall_format)
    sampler = self._converter_config.get_sampler()
    for _ in range(self._converter_config.num_samples):
      # Pick index entries with the same frame but different from the target.
      adversarial_frame = None
      found_adversarial_frame = False
      for _ in range(self._MAX_TRIALS):
        adversarial_frame = random.choice(self._all_frames)
        if adversarial_frame == gold_frame:
          continue
        # Ensure that there are enough exemplars.
        num_exemplars = len(self._frame_to_hashed_ids[adversarial_frame])
        if num_exemplars >= self._converter_config.max_exemplars:
          found_adversarial_frame = True
          break
      if not found_adversarial_frame:
        raise RuntimeError("An adversarial frame is not found.")
      yield sampler(self._frame_to_hashed_ids[adversarial_frame])


_CONVERTERS = {
    "query_only": QueryOnlyConverter,
    "add_top": AddTopKConverter,
    "add_samp": AddSampledKConverter,
    "add_oracle": AddOracleKConverter,
    "add_adversarial": AddAdversarialKConverter,
}


def get_converter(converter_name: str, retrieval_index: Iterable[RawExample],
                  funcall_format: str, converter_kwargs: Dict[str, Any],
                  formatter_kwargs: Dict[str, Any]) -> BaseExampleConverter:
  """Returns an example converter with the specified name.

  Args:
    converter_name: Name of the converter.
    retrieval_index: An iterable of dicts, where each dict contains information
      about an entry in the retrieval index.
    funcall_format: Format of the output function call or logical form.
    converter_kwargs: Keyword arguments for the converter's initializer. Some
      keywords have shorthands as defined in _CONVERTER_CONFIG_ALIASES.
    formatter_kwargs: Keyword arguments for the converter's initializer. Some
      keywords have shorthands as defined in _FORMATTER_CONFIG_ALIASES.

  Returns:
    A subclass of BaseExampleConverter.
  """
  converter = _CONVERTERS[converter_name]
  converter_config = ConverterConfig.from_dict(converter_kwargs)
  formatter_config = casper_formatters.FormatterConfig.from_dict(
      formatter_kwargs)
  return converter(retrieval_index, funcall_format, converter_config,
                   formatter_config)
