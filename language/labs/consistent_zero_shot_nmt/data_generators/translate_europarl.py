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
"""Data generators for Europarl zero-shot translation task."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import flags
from absl import logging

from language.labs.consistent_zero_shot_nmt.data_generators import translate_multilingual
from tensor2tensor.data_generators import problem
from tensor2tensor.utils import registry

import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_string("europarl_orig_data_path", "",
                    "Data directory for Europarl.")
flags.DEFINE_string("europarl_overlap_data_path", "",
                    "Overlap data directory for Europarl.")

__all__ = [
    "TranslateEuroparl",
    "TranslateEuroparlNonoverlap",
]

_EPL_ALL_LANG_PAIRS = [
    # en <> {de, es, fr} (6 pairs).
    ("en", "de"),
    ("de", "en"),
    ("en", "es"),
    ("es", "en"),
    ("en", "fr"),
    ("fr", "en"),
    # {de, fr, es} <> {de, fr, es} (6 zero-shot pairs).
    ("es", "de"),
    ("de", "es"),
    ("es", "fr"),
    ("fr", "es"),
    ("de", "fr"),
    ("fr", "de"),
]

# 4 training pairs (excluding zero-shot).
_EPL_TRAIN_LANG_PAIRS = _EPL_ALL_LANG_PAIRS[:6]

# 6 testing pairs (all directions).
_EPL_TEST_LANG_PAIRS = _EPL_ALL_LANG_PAIRS[:]

_EPL_TRAIN_DATASETS = [
    {
        "src_lang": "<" + src_lang + ">",
        "tgt_lang": "<" + tgt_lang + ">",
        "src_fname": "parallel/europarl-v7.{src_lang}-{tgt_lang}.{src_lang}".format(  # pylint: disable=line-too-long
            src_lang=src_lang, tgt_lang=tgt_lang),
        "tgt_fname": "parallel/europarl-v7.{src_lang}-{tgt_lang}.{tgt_lang}".format(  # pylint: disable=line-too-long
            src_lang=src_lang, tgt_lang=tgt_lang),
    }
    for src_lang, tgt_lang in _EPL_TRAIN_LANG_PAIRS
]

_EPL_TRAIN_REMOVE_SETS = [
    {
        "src_remove": "overlap/remove.{src_lang}-{tgt_lang}.{src_lang}".format(
            src_lang=src_lang, tgt_lang=tgt_lang),
        "tgt_remove": "overlap/remove.{src_lang}-{tgt_lang}.{tgt_lang}".format(
            src_lang=src_lang, tgt_lang=tgt_lang),
    }
    for src_lang, tgt_lang in _EPL_TRAIN_LANG_PAIRS
]

_EPL_TEST_DATASETS = [
    {
        "src_lang": "<" + src_lang + ">",
        "tgt_lang": "<" + tgt_lang + ">",
        "src_fname": "wmt06/dev/dev2006.{src_lang}".format(src_lang=src_lang),
        "tgt_fname": "wmt06/dev/dev2006.{tgt_lang}".format(tgt_lang=tgt_lang),
    }
    for src_lang, tgt_lang in _EPL_TEST_LANG_PAIRS
]


def _parse_lines(path):
  """Parses lines from UNCorpus dataset."""
  lines = []
  with tf.gfile.GFile(path) as fp:
    for line in fp:
      lines.append(line.strip())
  return lines


def _compile_data(tmp_dir, datasets, filename):
  """Concatenate all `datasets` and save to `filename`."""
  filename = os.path.join(tmp_dir, filename)
  src_fname = filename + ".src"
  tgt_fname = filename + ".tgt"
  if tf.gfile.Exists(src_fname) and tf.gfile.Exists(tgt_fname):
    tf.logging.info("Skipping compile data, found files:\n%s\n%s",
                    src_fname, tgt_fname)
    return filename
  with tf.gfile.GFile(src_fname, mode="w") as src_resfile:
    with tf.gfile.GFile(tgt_fname, mode="w") as tgt_resfile:
      for d in datasets:
        logging.info("Loading %s-%s...", d["src_lang"], d["tgt_lang"])
        # Load source and target lines.
        src_fpath = os.path.join(FLAGS.europarl_orig_data_path + d["src_fname"])
        tgt_fpath = os.path.join(FLAGS.europarl_orig_data_path + d["tgt_fname"])
        src_lines = _parse_lines(src_fpath)
        tgt_lines = _parse_lines(tgt_fpath)
        assert len(src_lines) == len(tgt_lines)
        logging.info("...loaded %d parallel sentences", len(src_lines))
        # Filter overlap, if necessary.
        if "src_remove" in d:
          src_remove_path = os.path.join(FLAGS.europarl_overlap_data_path,
                                         d["src_remove"])
          if tf.gfile.Exists(src_remove_path):
            logging.info("...filtering src overlap")
            src_remove = set(_parse_lines(src_remove_path))
            logging.info("...total overlapping lines: %d", len(src_remove))
            logging.info("...lines before filtering: %d", len(src_lines))
            src_tgt_lines = [
                (src_line, tgt_line)
                for src_line, tgt_line in zip(src_lines, tgt_lines)
                if src_line not in src_remove]
            src_lines, tgt_lines = map(list, zip(*src_tgt_lines))
            logging.info("...lines after filtering: %d", len(src_lines))
        if "tgt_remove" in d:
          tgt_remove_path = os.path.join(FLAGS.europarl_overlap_data_path,
                                         d["tgt_remove"])
          if tf.gfile.Exists(tgt_remove_path):
            logging.info("...filtering tgt overlap")
            tgt_remove = set(_parse_lines(tgt_remove_path))
            logging.info("...total overlapping lines: %d", len(tgt_remove))
            logging.info("...lines before filtering: %d", len(src_lines))
            src_tgt_lines = [
                (src_line, tgt_line)
                for src_line, tgt_line in zip(src_lines, tgt_lines)
                if tgt_line not in tgt_remove]
            src_lines, tgt_lines = map(list, zip(*src_tgt_lines))
            logging.info("...lines after filtering: %d", len(src_lines))
        assert len(src_lines) == len(tgt_lines)
        # Prepend tags to each source and target line.
        src_lines = [d["src_lang"] + l for l in src_lines]
        tgt_lines = [d["tgt_lang"] + l for l in tgt_lines]
        # Write preprocessed source and target lines.
        logging.info("...writing preprocessed files")
        for src_line, tgt_line in zip(src_lines, tgt_lines):
          src_resfile.write(src_line)
          src_resfile.write("\n")
          tgt_resfile.write(tgt_line)
          tgt_resfile.write("\n")
  return filename


@registry.register_problem
class TranslateEuroparl(translate_multilingual.TranslateMultilingualProblem):
  """Problem spec for Europarl zeroshot translation."""

  def source_data_files(self, dataset_split):
    """Files to be passed to compile_data."""
    if dataset_split == problem.DatasetSplit.TRAIN:
      return _EPL_TRAIN_DATASETS
    else:
      return _EPL_TEST_DATASETS

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    auxiliary_tags = ["<de>", "<es>", "<fr>"]
    return self._generate_samples(data_dir, tmp_dir, dataset_split,
                                  auxiliary_tags=auxiliary_tags,
                                  compile_data_fn=_compile_data)

  def generate_text_for_vocab(self, data_dir, tmp_dir):
    return self._generate_text_for_vocab(
        data_dir,
        tmp_dir,
        datapath=FLAGS.europarl_orig_data_path,
        parse_lines_fn=_parse_lines)


@registry.register_problem
class TranslateEuroparlNonoverlap(TranslateEuroparl):
  """Problem spec for Europarl zeroshot translation without overlap."""

  def source_data_files(self, dataset_split):
    """Files to be passed to compile_data."""
    if dataset_split == problem.DatasetSplit.TRAIN:
      # Include overlap information.
      return [
          dict(list(d.items()) + list(o.items()))
          for d, o in zip(_EPL_TRAIN_DATASETS, _EPL_TRAIN_REMOVE_SETS)]
    else:
      return _EPL_TEST_DATASETS
