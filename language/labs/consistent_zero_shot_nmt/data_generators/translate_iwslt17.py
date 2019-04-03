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
"""Data generators for IWSLT17 zero-shot translation task."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re

from absl import flags
from absl import logging

from language.labs.consistent_zero_shot_nmt.data_generators import translate_multilingual
from tensor2tensor.data_generators import problem
from tensor2tensor.utils import registry

import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_string("iwslt17_orig_data_path", "", "Data directory for IWSLT17.")
flags.DEFINE_string("iwslt17_overlap_data_path", "",
                    "Overlap data directory for IWSLT17.")

__all__ = [
    "TranslateIwslt17",
    "TranslateIwslt17Nonoverlap",
    "TranslateIwslt17Autoenc",
    "TranslateIwslt17NonoverlapAutoenc",
]


# 20 pairs total.
_IWSLT17_ALL_LANG_PAIRS = [
    # en <> {de, it, nl, ro} (8 pairs).
    ("en", "de"),
    ("de", "en"),
    ("en", "nl"),
    ("nl", "en"),
    ("en", "it"),
    ("it", "en"),
    ("en", "ro"),
    ("ro", "en"),
    # de <> {it, ro} (4 pairs).
    ("de", "it"),
    ("it", "de"),
    ("de", "ro"),
    ("ro", "de"),
    # nl <> {it, ro} (4 pairs).
    ("nl", "it"),
    ("it", "nl"),
    ("nl", "ro"),
    ("ro", "nl"),
    # de <> nl and it <> ro (4 zero-shot pairs).
    ("de", "nl"),
    ("nl", "de"),
    ("it", "ro"),
    ("ro", "it"),
]

# 8 training pairs that contain en as source or target.
_IWSLT17_TRAIN_LANG_PAIRS = _IWSLT17_ALL_LANG_PAIRS[:8]

# 20 testing pairs (all directions).
_IWSLT17_TEST_LANG_PAIRS = _IWSLT17_ALL_LANG_PAIRS[:]

# 4 pairs used for autoencoding (en is excluded).
_IWSLT17_AUTOENC_LANG_PAIRS = [
    ("en", "de"),
    ("en", "nl"),
    ("en", "it"),
    ("en", "ro"),
]

_IWSLT17_TRAIN_DATASETS = [
    {
        "src_lang": "<" + src_lang + ">",
        "tgt_lang": "<" + tgt_lang + ">",
        "src_fname": "train.tags.{src_lang}-{tgt_lang}.{src_lang}".format(
            src_lang=src_lang, tgt_lang=tgt_lang),
        "tgt_fname": "train.tags.{src_lang}-{tgt_lang}.{tgt_lang}".format(
            src_lang=src_lang, tgt_lang=tgt_lang),
    }
    for src_lang, tgt_lang in _IWSLT17_TRAIN_LANG_PAIRS
]

_IWSLT17_TRAIN_REMOVE_SETS = [
    {
        "src_remove": "remove.{src_lang}-{tgt_lang}.{src_lang}".format(
            src_lang=src_lang, tgt_lang=tgt_lang),
        "tgt_remove": "remove.{src_lang}-{tgt_lang}.{tgt_lang}".format(
            src_lang=src_lang, tgt_lang=tgt_lang),
    }
    for src_lang, tgt_lang in _IWSLT17_TRAIN_LANG_PAIRS
]

_IWSLT17_AUTOENC_DATASETS = [
    {
        "src_lang": "<" + tgt_lang + ">",
        "tgt_lang": "<" + tgt_lang + ">",
        "src_fname": "train.tags.{src_lang}-{tgt_lang}.{tgt_lang}".format(
            src_lang=src_lang, tgt_lang=tgt_lang),
        "tgt_fname": "train.tags.{src_lang}-{tgt_lang}.{tgt_lang}".format(
            src_lang=src_lang, tgt_lang=tgt_lang),
    }
    for src_lang, tgt_lang in _IWSLT17_AUTOENC_LANG_PAIRS
]

_IWSLT17_TEST_DATASETS = [
    {
        "src_lang": "<" + src_lang + ">",
        "tgt_lang": "<" + tgt_lang + ">",
        "src_fname": "IWSLT17.TED.dev2010.{src_lang}-{tgt_lang}.{src_lang}.xml".format(  # pylint: disable=line-too-long
            src_lang=src_lang, tgt_lang=tgt_lang),
        "tgt_fname": "IWSLT17.TED.dev2010.{src_lang}-{tgt_lang}.{tgt_lang}.xml".format(  # pylint: disable=line-too-long
            src_lang=src_lang, tgt_lang=tgt_lang),
    }
    for src_lang, tgt_lang in _IWSLT17_TEST_LANG_PAIRS
]

_ALLOWED_TAGS = {"description", "seg", "title"}
_FLAT_HTML_REGEX = re.compile(r"<([^ ]*).*>(.*)</(.*)>")
_WHOLE_TAG_REGEX = re.compile(r"<[^<>]*>\Z")


def _parse_lines(path):
  """Parses lines from IWSLT17 dataset."""
  lines = []
  if tf.gfile.Exists(path):
    with tf.gfile.GFile(path) as fp:
      for line in fp:
        line = line.strip()
        # Skip lines that are tags entirely.
        if _WHOLE_TAG_REGEX.match(line):
          continue
        # Try to parse as content between an opening and closing tags.
        match = _FLAT_HTML_REGEX.match(line)
        # Always append text not contained between the tags.
        if match is None:
          lines.append(line)
        elif (match.group(1) == match.group(3) and
              match.group(1).lower() in _ALLOWED_TAGS):
          lines.append(match.group(2).strip())
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
        src_fpath = os.path.join(FLAGS.iwslt17_orig_data_path, d["src_fname"])
        tgt_fpath = os.path.join(FLAGS.iwslt17_orig_data_path, d["tgt_fname"])
        src_lines = _parse_lines(src_fpath)
        tgt_lines = _parse_lines(tgt_fpath)
        assert len(src_lines) == len(tgt_lines)
        logging.info("...loaded %d parallel sentences", len(src_lines))
        # Filter overlap, if necessary.
        if "src_remove" in d:
          logging.info("...filtering src overlap")
          src_remove_path = os.path.join(FLAGS.iwslt17_overlap_data_path,
                                         d["src_remove"])
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
          logging.info("...filtering tgt overlap")
          tgt_remove_path = os.path.join(FLAGS.iwslt17_overlap_data_path,
                                         d["tgt_remove"])
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
class TranslateIwslt17(translate_multilingual.TranslateMultilingualProblem):
  """Problem spec for IWSLT17 zeroshot translation."""

  def source_data_files(self, dataset_split):
    """Files to be passed to compile_data."""
    if dataset_split == problem.DatasetSplit.TRAIN:
      return _IWSLT17_TRAIN_DATASETS
    else:
      return _IWSLT17_TEST_DATASETS

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    auxiliary_tags = ["<de>", "<it>", "<nl>", "<ro>"]
    return self._generate_samples(data_dir, tmp_dir, dataset_split,
                                  auxiliary_tags=auxiliary_tags,
                                  compile_data_fn=_compile_data)

  def generate_text_for_vocab(self, data_dir, tmp_dir):
    return self._generate_text_for_vocab(
        data_dir,
        tmp_dir,
        datapath=FLAGS.iwslt17_orig_data_path,
        parse_lines_fn=_parse_lines)


@registry.register_problem
class TranslateIwslt17Nonoverlap(TranslateIwslt17):
  """Problem spec for IWSLT17 zeroshot translation without overlap."""

  def source_data_files(self, dataset_split):
    """Files to be passed to compile_data."""
    if dataset_split == problem.DatasetSplit.TRAIN:
      # Include overlap information.
      return [
          dict(list(d.items()) + list(o.items()))
          for d, o in zip(_IWSLT17_TRAIN_DATASETS, _IWSLT17_TRAIN_REMOVE_SETS)]
    else:
      return _IWSLT17_TEST_DATASETS


@registry.register_problem
class TranslateIwslt17Autoenc(TranslateIwslt17):
  """Problem spec for IWSLT17 zeroshot translation with autoencoding."""

  def source_data_files(self, dataset_split):
    """Files to be passed to compile_data."""
    if dataset_split == problem.DatasetSplit.TRAIN:
      return _IWSLT17_TRAIN_DATASETS + _IWSLT17_AUTOENC_DATASETS
    else:
      return _IWSLT17_TEST_DATASETS


@registry.register_problem
class TranslateIwslt17NonoverlapAutoenc(TranslateIwslt17Nonoverlap):
  """Problem spec for IWSLT17 zeroshot translation with autoencoding."""

  def source_data_files(self, dataset_split):
    """Files to be passed to compile_data."""
    if dataset_split == problem.DatasetSplit.TRAIN:
      data_files_nonoverlap = [
          dict(list(d.items()) + list(o.items()))
          for d, o in zip(_IWSLT17_TRAIN_DATASETS, _IWSLT17_TRAIN_REMOVE_SETS)]
      return data_files_nonoverlap + _IWSLT17_AUTOENC_DATASETS
    else:
      return _IWSLT17_TEST_DATASETS
