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
"""Common functions for multilingual data generators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import translate
from tensor2tensor.layers import modalities
from tensor2tensor.utils import metrics

from tensor2tensor.utils import mlperf_log

import tensorflow as tf


# Length of the language tag strings.
LANG_TAG_LENGTH = 4

# List of all language tags.
LANG_TAGS = ["<de>", "<en>", "<es>", "<fr>", "<it>", "<nl>", "<ro>", "<ru>"]


def get_tag_id(tag):
  """Given the tag string, returns its index in the vocabulary."""
  index = LANG_TAGS.index(tag)
  # Adjust index to account for the tokens reserved by text_encoder.
  index += len(text_encoder.RESERVED_TOKENS)
  return index


class TranslateMultilingualProblem(translate.TranslateProblem):
  """Base class for multilingual translate problems."""

  @property
  def vocab_type(self):
    return text_problems.VocabType.SUBWORD

  @property
  def approx_vocab_size(self):
    return 2**15  # ~32k

  @property
  def additional_reserved_tokens(self):
    """Returns a list of language tags."""
    return LANG_TAGS

  def example_reading_spec(self):
    data_fields = {
        "all_tags": tf.VarLenFeature(tf.int64),
        "inputs": tf.VarLenFeature(tf.int64),
        "input_tags": tf.VarLenFeature(tf.int64),
        "targets": tf.VarLenFeature(tf.int64),
        "target_tags": tf.VarLenFeature(tf.int64),
    }
    data_items_to_decoders = None
    return data_fields, data_items_to_decoders

  def hparams(self, defaults, unused_model_hparams):
    super(TranslateMultilingualProblem, self).hparams(
        defaults, unused_model_hparams)
    p = defaults
    p.modality["all_tags"] = modalities.SymbolModality
    p.vocab_size["all_tags"] = self._encoders["inputs"].vocab_size
    p.modality["input_tags"] = modalities.SymbolModality
    p.vocab_size["input_tags"] = self._encoders["inputs"].vocab_size
    p.modality["target_tags"] = modalities.SymbolModality
    p.vocab_size["target_tags"] = self._encoders["targets"].vocab_size

    if self.packed_length:
      raise NotImplementedError("TranslateMultilingualProblem does not "
                                "support packed_length")

  def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
    if dataset_split == problem.DatasetSplit.TRAIN:
      mlperf_log.transformer_print(key=mlperf_log.PREPROC_TOKENIZE_TRAINING)
    elif dataset_split == problem.DatasetSplit.EVAL:
      mlperf_log.transformer_print(key=mlperf_log.PREPROC_TOKENIZE_EVAL)

    generator = self.generate_samples(data_dir, tmp_dir, dataset_split)
    encoder = self.get_or_create_vocab(data_dir, tmp_dir)
    return text2text_generate_encoded(generator, encoder)

  def _generate_samples(self, data_dir, tmp_dir, dataset_split,
                        auxiliary_tags=None,
                        compile_data_fn=None):
    if auxiliary_tags is None:
      auxiliary_tags = LANG_TAGS
    if compile_data_fn is None:
      compile_data_fn = translate.compile_data
    datasets = self.source_data_files(dataset_split)
    tag = "train" if dataset_split == problem.DatasetSplit.TRAIN else "dev"
    data_path = compile_data_fn(
        tmp_dir, datasets, "%s-compiled-%s" % (self.name, tag))
    return text2text_txt_iterator(
        source_txt_path=data_path + ".src",
        target_txt_path=data_path + ".tgt",
        auxiliary_tags=auxiliary_tags)

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    """Generates samples. Must be implemented by a subclass."""
    raise NotImplementedError("Abstract Method")

  def _generate_text_for_vocab(self, data_dir, tmp_dir,
                               datapath, parse_lines_fn):
    return generate_lines_for_shared_vocab(
        datasets=self.vocab_data_files(),
        datapath=datapath,
        parse_lines_fn=parse_lines_fn)

  def generate_text_for_vocab(self, data_dir, tmp_dir):
    """Generates text lines for vocab. Must be implemented by a subclass."""
    raise NotImplementedError("Abstract Method")


class TranslateMultilingualLmProblem(TranslateMultilingualProblem):
  """Base class for multilingual (auxiliary) LM problems."""

  @property
  def decode_hooks(self):
    return []

  def eval_metrics(self):
    return [
        metrics.Metrics.ACC,
        metrics.Metrics.ACC_TOP5,
        metrics.Metrics.ACC_PER_SEQ,
        metrics.Metrics.NEG_LOG_PERPLEXITY,
    ]


def parse_lines(path):
  """Parses lines from dataset file."""
  lines = []
  with tf.gfile.GFile(path) as fp:
    for line in fp:
      lines.append(line.strip())
  return lines


def generate_lines_for_shared_vocab(datasets, datapath, parse_lines_fn):
  """Generate lines for a shared vocabulary generation."""
  for d in datasets:
    src_fpath = os.path.join(datapath + d["src_fname"])
    tgt_fpath = os.path.join(datapath + d["tgt_fname"])
    src_lines = parse_lines_fn(src_fpath)
    tgt_lines = parse_lines_fn(tgt_fpath)
    for src_line, tgt_line in zip(src_lines, tgt_lines):
      yield src_line
      yield tgt_line


def txt_line_lang_iterator(txt_path):
  """Iterate through lines of file."""
  with tf.gfile.Open(txt_path) as f:
    for line in f:
      line = line.strip()
      content, lang = line[LANG_TAG_LENGTH:], line[:LANG_TAG_LENGTH]
      yield content, lang


def text2text_txt_iterator(source_txt_path, target_txt_path, auxiliary_tags):
  """Yield dicts for TranslateMultilingual.generate_samples from files."""
  for sources, targets in zip(
      txt_line_lang_iterator(source_txt_path),
      txt_line_lang_iterator(target_txt_path)):
    yield {
        "all_tags": auxiliary_tags,
        "inputs": sources[0],
        "input_tags": sources[1],
        "targets": targets[0],
        "target_tags": targets[1],
    }


def text2text_generate_encoded(sample_generator, vocab, targets_vocab=None):
  """Encode TranslateMultilingual samples from the generator with the vocab."""
  targets_vocab = targets_vocab or vocab
  for sample in sample_generator:
    # Encode sequences.
    sample["inputs"] = vocab.encode(sample["inputs"])
    sample["targets"] = targets_vocab.encode(sample["targets"])
    # Add EOS ids.
    sample["inputs"].append(text_encoder.EOS_ID)
    sample["targets"].append(text_encoder.EOS_ID)
    # Encode language tags.
    sample["input_tags"] = [get_tag_id(sample["input_tags"])]
    sample["target_tags"] = [get_tag_id(sample["target_tags"])]
    sample["all_tags"] = [get_tag_id(tag) for tag in sample["all_tags"]]
    yield sample
