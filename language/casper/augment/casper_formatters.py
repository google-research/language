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
"""Formats the exemplars and how they are augmented to the input query."""
import copy
import dataclasses
import random


from language.casper.utils import data_types
from language.casper.utils import top_constants
from language.casper.utils import top_utils

RawExample = data_types.RawExample

_MAX_NUM_LABELS = 100


def _punctuation_config_preset(kwargs):
  """Populates the config to use punctuations as separators in the prompt."""
  return dict(
      kwargs,
      orig_input_prefix="",
      exemplar_input_prefix="",
      exemplar_output_prefix=" ## ",
      exemplar_separator=" @@ ")


def _verbal_config_preset(kwargs):
  """Populates the config to use the T5-style verbalized prompt."""
  return dict(
      kwargs,
      orig_input_prefix="input: ",
      exemplar_input_prefix="example {i}: ",
      exemplar_output_prefix=" ## output {i}: ",
      exemplar_separator=" @@ ")


def _platinum_config_preset(kwargs):
  """Modifies the config to add a PLATINUM token before each exemplar."""
  return dict(
      kwargs,
      exemplar_input_prefix="PLATINUM " + kwargs["exemplar_input_prefix"])


_FORMATTER_CONFIG_PRESETS = {
    "punc": _punctuation_config_preset,
    "verbal": _verbal_config_preset,
    "plat": _platinum_config_preset,
}

_FORMATTER_CONFIG_ALIASES = {
    "inv": "orig_input_at_end",
}


@dataclasses.dataclass
class FormatterConfig:
  """Config for the augment_exemplars method.

  Attrs:
    rename_labels: A dict mapping old label names to new label names. If the new
      name is an empty string, subtrees with the old label will be dropped. This
      renaming is performed before anonymization.
    anonymize: Whether to anonymize the intent/slot labels.
    anonymized_labels_type: Type of anonymized labels
    orig_input_prefix: Text to put before the original input query.
    exemplar_input_prefix: Text to put before each exemplar input. Occurrences
      of "{i}" will be replaced with the exemplar index.
    exemplar_output_prefix: Text to put before each exemplar output. Occurrences
      of "{i}" will be replaced with the exemplar index.
    exemplar_separator: The separator between exemplars and between the original
      input and the exemplars.
    orig_input_at_end: Whether to put the original input at the end instead of
      the beginning.
  """
  rename_labels: Optional[Dict[str, str]] = None
  anonymize: bool = False
  anonymized_labels_type: str = "numbers"
  orig_input_prefix: str = ""
  exemplar_input_prefix: str = ""
  exemplar_output_prefix: str = " ## "
  exemplar_separator: str = " @@ "
  orig_input_at_end: bool = False

  @classmethod
  def from_dict(cls, converter_kwargs):
    """Constructs a ConverterConfig from the given dict.

    Args:
      converter_kwargs: A dict with attributes of FormatterConfig as keys.
        Optionally, converter_kwargs["presets"] can be a list of preset names
        from _FORMATTER_CONFIG_PRESETS, which will populate the dict with preset
        config values.

    Returns:
      a FormatterConfig object.
    """
    # Make a copy
    converter_kwargs = dict(converter_kwargs)
    # Resolve presets
    if "presets" in converter_kwargs:
      for preset in converter_kwargs["presets"]:
        converter_kwargs = _FORMATTER_CONFIG_PRESETS[preset](converter_kwargs)
      del converter_kwargs["presets"]
    # Resolve aliases
    for abbr_key, full_key in _FORMATTER_CONFIG_ALIASES.items():
      if abbr_key in converter_kwargs:
        converter_kwargs[full_key] = converter_kwargs[abbr_key]
        del converter_kwargs[abbr_key]
    config = cls(**converter_kwargs)
    return config


def preprocess_example(example, funcall_format):
  """Returns a preprocessed example according to the funcall format."""
  if funcall_format == "top":
    example = copy.deepcopy(example)
    # For old versions of the JSONL files:
    # Change `input_str` to match the official tokenization.
    if "tokens" in example:
      example["input_str"] = " ".join(example["tokens"])
    # Change `output_str` to be the formatted logical form.
    example["output_str"] = top_utils.format_serialized(example["output_str"])
    return example
  else:
    raise ValueError(f"Unknown funcall_format: {funcall_format}")


def rename_top_lf_labels(lfs,
                         label_map):
  """Rename the labels of the TopLFs in place.

  Args:
    lfs: List of TopLF objects.
    label_map: Mapping from old labels to new labels.
  """
  for lf in lfs:
    if lf.name in label_map and not label_map[lf.name]:
      raise ValueError("Cannot drop the whole LF: " + str(lf))
    stack: List[top_utils.TopLF] = [lf]
    while stack:
      sub_lf = stack.pop()
      # Renaming is done here
      if sub_lf.name in label_map:
        sub_lf.name = label_map[sub_lf.name]
      new_args = []
      for child in reversed(sub_lf.args):
        if isinstance(child, top_utils.TopLF):
          # Dropping is done here
          if child.name in label_map and not label_map[child.name]:
            continue
          stack.append(child)
        new_args.append(child)
      sub_lf.args = new_args[::-1]


def anonymize_top_lf_labels(
    lfs,
    anonymized_labels = None,
    anonymized_labels_type = None):
  """Anonymizes the intent/slot labels of the TopLFs in place.

  Args:
    lfs: List of TopLF objects.
    anonymized_labels: List of anonymized labels to use. The first intent/slot
      label will be replaced with the first entry of `anonymized_labels` and so
      on. If not specified, use a list based on `anonymized_labels_type`.
    anonymized_labels_type: Type of anonymized labels when `anonymized_labels`
      is not specified.
  """
  if not anonymized_labels:
    if anonymized_labels_type == "numbers":
      # Use a random list of numbers as anonymized labels
      anonymized_labels = list(range(_MAX_NUM_LABELS))
    elif anonymized_labels_type == "mtop":
      # Use a random list of labels from the MTOP dataset
      # Warning: This exposes unseen labels in the domain boostrapping setup.
      anonymized_labels = top_constants.MTOP_LABELS[:]
    else:
      raise ValueError(
          "Unknown anonymized_label_type: {}".format(anonymized_labels_type))
    random.shuffle(anonymized_labels)
  anonymized_labels_iter = iter(anonymized_labels)
  label_map = {}
  for lf in lfs:
    stack: List[top_utils.TopLF] = [lf]
    while stack:
      sub_lf = stack.pop()
      if sub_lf.name not in label_map:
        # "IN:CREATE_ALARM" --> "IN:87"
        label_map[sub_lf.name] = (
            sub_lf.name[:3] + str(next(anonymized_labels_iter)))
      sub_lf.name = label_map[sub_lf.name]
      for child in reversed(sub_lf.args):
        if isinstance(child, top_utils.TopLF):
          stack.append(child)


def top_funcall_processor(exemplar_outputs, orig_output,
                          config):
  """Returns the processed (exemplar_outputs, orig_output)."""
  # Convert to TopLF
  top_lfs = []
  for output_str in list(exemplar_outputs) + [orig_output]:
    lf = top_utils.deserialize_top(output_str)
    if lf is None:
      raise ValueError(f"Cannot deserialize {output_str}")
    top_lfs.append(lf)
  # Process the TopLFs
  if config.rename_labels:
    rename_top_lf_labels(top_lfs, config.rename_labels)
  if config.anonymize:
    anonymize_top_lf_labels(
        top_lfs, anonymized_labels_type=config.anonymized_labels_type)
  # Convert back into strings.
  outputs = [top_utils.format_serialized(lf.serialize()) for lf in top_lfs]
  return outputs[:-1], outputs[-1]


_FUNCALL_PROCESSOR = {
    "top": top_funcall_processor,
}


def format_prompt(orig_input, exemplar_inputs,
                  exemplar_outputs,
                  config):
  """Constructs a retrieval-augmented input string.

  Args:
    orig_input: Input of the original example.
    exemplar_inputs: Inputs of the exemplars.
    exemplar_outputs: Outputs of the exemplars.
    config: a FormatterConfig object.

  Returns:
    The retrieval-augmented input string.
  """
  input_parts = []
  if not config.orig_input_at_end:
    input_parts.append(config.orig_input_prefix + orig_input)
  for i, (ex_in, ex_out) in enumerate(zip(exemplar_inputs, exemplar_outputs)):
    input_prefix = config.exemplar_input_prefix.replace("{i}", str(i + 1))
    output_prefix = config.exemplar_output_prefix.replace("{i}", str(i + 1))
    input_parts.append(input_prefix + ex_in + output_prefix + ex_out)
  if config.orig_input_at_end:
    input_parts.append(config.orig_input_prefix + orig_input)
  return config.exemplar_separator.join(input_parts)


def augment_exemplars(example, exemplars,
                      funcall_format,
                      config):
  """Generates (input, output) pair where the input is retrieval-augmented.

  Args:
    example: Original RawExample.
    exemplars: List of exemplars to augment.
    funcall_format: Format of the output function call or logical form.
    config: a FormatterConfig object.

  Returns:
    Tuple (input string, output string).
  """
  orig_input = example["input_str"]
  orig_output = example["output_str"]
  exemplar_inputs = [exemplar["input_str"] for exemplar in exemplars]
  exemplar_outputs = [exemplar["output_str"] for exemplar in exemplars]

  # Process the funcalls.
  funcall_processor = _FUNCALL_PROCESSOR[funcall_format]
  exemplar_outputs, orig_output = funcall_processor(exemplar_outputs,
                                                    orig_output, config)
  input_str = format_prompt(orig_input, exemplar_inputs, exemplar_outputs,
                            config)
  return input_str, orig_output
