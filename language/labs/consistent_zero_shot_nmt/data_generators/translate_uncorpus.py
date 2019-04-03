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
"""Data generators for UNCorpus zero-shot translation task."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os

from absl import flags
from absl import logging

from language.labs.consistent_zero_shot_nmt.data_generators import translate_multilingual
from tensor2tensor.data_generators import problem
from tensor2tensor.utils import registry

import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_string("uncorpus_orig_data_exp1", "",
                    "Data directory for UNCorpus experiment 1.")
flags.DEFINE_string(
    "uncorpus_orig_data_exp1_lm", "",
    "Data directory for UNCorpus experiment 1 language model data.")
flags.DEFINE_string("uncorpus_orig_data_exp2", "",
                    "Data directory for UNCorpus experiment 2.")
flags.DEFINE_string(
    "uncorpus_orig_data_exp2_lm", "",
    "Data directory for UNCorpus experiment 2 language model data.")

__all__ = [
    "TranslateUncorpusExp1",
    "TranslateUncorpusExp2"
]

_UNC_ALL_LANG_PAIRS_EXP1 = [
    # en <> {fr, es} (4 pairs).
    ("en", "fr"),
    ("fr", "en"),
    ("en", "es"),
    ("es", "en"),
    # es <> fr (2 zero-shot pairs).
    ("es", "fr"),
    ("fr", "es"),
]

_UNC_LM_LANG_PAIRS_EXP1 = [
    ("es", "es"),
    ("fr", "fr"),
]

_UNC_ALL_LANG_PAIRS_EXP2 = [
    # en <> {fr, es, ru} (6 pairs).
    ("en", "fr"),
    ("fr", "en"),
    ("en", "es"),
    ("es", "en"),
    ("en", "ru"),
    ("ru", "en"),
    # {es, fr, ru} <> {es, fr, ru} (6 zero-shot pairs).
    ("es", "fr"),
    ("fr", "es"),
    ("es", "ru"),
    ("ru", "es"),
    ("fr", "ru"),
    ("ru", "fr"),
]

_UNC_LM_LANG_PAIRS_EXP2 = [
    ("es", "es"),
    ("fr", "fr"),
    ("ru", "ru"),
]

# 4 training pairs (excluding zero-shot).
_UNC_TRAIN_LANG_PAIRS_EXP1 = _UNC_ALL_LANG_PAIRS_EXP1[:4]
_UNC_TRAIN_LANG_PAIRS_EXP2 = _UNC_ALL_LANG_PAIRS_EXP2[:6]

# 6 testing pairs (all directions).
_UNC_TEST_LANG_PAIRS_EXP1 = _UNC_ALL_LANG_PAIRS_EXP1[:]
_UNC_TEST_LANG_PAIRS_EXP2 = _UNC_ALL_LANG_PAIRS_EXP2[:]

_UNC_TRAIN_DATASETS_EXP1 = [
    {
        "src_lang": "<" + src_lang + ">",
        "tgt_lang": "<" + tgt_lang + ">",
        "src_fname": "UNv1.0.1M-sestorain-exp1.{src_lang}-{tgt_lang}.{src_lang}"
                     .format(src_lang=src_lang, tgt_lang=tgt_lang),
        "tgt_fname": "UNv1.0.1M-sestorain-exp1.{src_lang}-{tgt_lang}.{tgt_lang}"
                     .format(src_lang=src_lang, tgt_lang=tgt_lang),
    }
    for src_lang, tgt_lang in _UNC_TRAIN_LANG_PAIRS_EXP1
]

_UNC_TRAIN_DATASETS_EXP2 = [
    {
        "src_lang": "<" + src_lang + ">",
        "tgt_lang": "<" + tgt_lang + ">",
        "src_fname": "UNv1.0.1M-sestorain-exp2.{src_lang}-{tgt_lang}.{src_lang}"
                     .format(src_lang=src_lang, tgt_lang=tgt_lang),
        "tgt_fname": "UNv1.0.1M-sestorain-exp2.{src_lang}-{tgt_lang}.{tgt_lang}"
                     .format(src_lang=src_lang, tgt_lang=tgt_lang),
    }
    for src_lang, tgt_lang in _UNC_TRAIN_LANG_PAIRS_EXP2
]

_UNC_TEST_DATASETS_EXP1 = [
    {
        "src_lang": "<" + src_lang + ">",
        "tgt_lang": "<" + tgt_lang + ">",
        "src_fname": "UNv1.0.devset.{src_lang}".format(src_lang=src_lang),
        "tgt_fname": "UNv1.0.devset.{tgt_lang}".format(tgt_lang=tgt_lang),
    }
    for src_lang, tgt_lang in _UNC_TEST_LANG_PAIRS_EXP1
]

_UNC_TEST_DATASETS_EXP2 = [
    {
        "src_lang": "<" + src_lang + ">",
        "tgt_lang": "<" + tgt_lang + ">",
        "src_fname": "UNv1.0.devset.{src_lang}".format(src_lang=src_lang),
        "tgt_fname": "UNv1.0.devset.{tgt_lang}".format(tgt_lang=tgt_lang),
    }
    for src_lang, tgt_lang in _UNC_TEST_LANG_PAIRS_EXP2
]

_UNC_TRAIN_DATASETS_EXP1_LM = [
    {
        "src_lang": "<" + src_lang + ">",
        "tgt_lang": "<" + tgt_lang + ">",
        "src_fname": "UNv1.0.sestorain-exp1-LM-concat-shuffled.{src_lang}"
                     .format(src_lang=src_lang),
        "tgt_fname": "UNv1.0.sestorain-exp1-LM-concat-shuffled.{tgt_lang}"
                     .format(tgt_lang=tgt_lang),
    }
    for src_lang, tgt_lang in _UNC_LM_LANG_PAIRS_EXP1
]

_UNC_TEST_DATASETS_EXP1_LM = [
    {
        "src_lang": "<" + src_lang + ">",
        "tgt_lang": "<" + tgt_lang + ">",
        "src_fname": "UNv1.0.devset.{src_lang}".format(src_lang=src_lang),
        "tgt_fname": "UNv1.0.devset.{tgt_lang}".format(tgt_lang=tgt_lang),
    }
    for src_lang, tgt_lang in _UNC_LM_LANG_PAIRS_EXP1
]

_UNC_TRAIN_DATASETS_EXP2_LM = [
    {
        "src_lang": "<" + src_lang + ">",
        "tgt_lang": "<" + tgt_lang + ">",
        "src_fname": "UNv1.0.sestorain-exp2-LM-concat-shuffled.{src_lang}"
                     .format(src_lang=src_lang),
        "tgt_fname": "UNv1.0.sestorain-exp2-LM-concat-shuffled.{tgt_lang}"
                     .format(tgt_lang=tgt_lang),
    }
    for src_lang, tgt_lang in _UNC_LM_LANG_PAIRS_EXP2
]

_UNC_TEST_DATASETS_EXP2_LM = [
    {
        "src_lang": "<" + src_lang + ">",
        "tgt_lang": "<" + tgt_lang + ">",
        "src_fname": "UNv1.0.devset.{src_lang}".format(src_lang=src_lang),
        "tgt_fname": "UNv1.0.devset.{tgt_lang}".format(tgt_lang=tgt_lang),
    }
    for src_lang, tgt_lang in _UNC_LM_LANG_PAIRS_EXP2
]


def _parse_lines(path):
  """Parses lines from UNCorpus dataset."""
  lines = []
  with tf.gfile.GFile(path) as fp:
    for line in fp:
      lines.append(line.strip())
  return lines


def _compile_data(data_path, tmp_dir, datasets, filename):
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
        src_fpath = os.path.join(data_path + d["src_fname"])
        tgt_fpath = os.path.join(data_path + d["tgt_fname"])
        src_lines = _parse_lines(src_fpath)
        tgt_lines = _parse_lines(tgt_fpath)
        assert len(src_lines) == len(tgt_lines)
        logging.info("...loaded %d parallel sentences", len(src_lines))
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
class TranslateUncorpusExp1(
    translate_multilingual.TranslateMultilingualProblem):
  """Problem spec for UNCorpus zeroshot translation, experiment 1."""

  def source_data_files(self, dataset_split):
    """Files to be passed to compile_data."""
    if dataset_split == problem.DatasetSplit.TRAIN:
      return _UNC_TRAIN_DATASETS_EXP1
    else:
      return _UNC_TEST_DATASETS_EXP1

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    auxiliary_tags = ["<es>", "<fr>"]
    compile_data_fn = functools.partial(_compile_data,
                                        FLAGS.uncorpus_orig_data_exp1)
    return self._generate_samples(data_dir, tmp_dir, dataset_split,
                                  auxiliary_tags=auxiliary_tags,
                                  compile_data_fn=compile_data_fn)

  def generate_text_for_vocab(self, data_dir, tmp_dir):
    return self._generate_text_for_vocab(
        data_dir,
        tmp_dir,
        datapath=FLAGS.uncorpus_orig_data_exp1,
        parse_lines_fn=_parse_lines)


@registry.register_problem
class TranslateUncorpusExp2(
    translate_multilingual.TranslateMultilingualProblem):
  """Problem spec for UNCorpus zeroshot translation, experiment 2."""

  def source_data_files(self, dataset_split):
    """Files to be passed to compile_data."""
    if dataset_split == problem.DatasetSplit.TRAIN:
      return _UNC_TRAIN_DATASETS_EXP2
    else:
      return _UNC_TEST_DATASETS_EXP2

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    auxiliary_tags = ["<es>", "<fr>", "<ru>"]
    compile_data_fn = functools.partial(_compile_data,
                                        FLAGS.uncorpus_orig_data_exp2)
    return self._generate_samples(data_dir, tmp_dir, dataset_split,
                                  auxiliary_tags=auxiliary_tags,
                                  compile_data_fn=compile_data_fn)

  def generate_text_for_vocab(self, data_dir, tmp_dir):
    return self._generate_text_for_vocab(
        data_dir,
        tmp_dir,
        datapath=FLAGS.uncorpus_orig_data_exp2,
        parse_lines_fn=_parse_lines)


@registry.register_problem
class TranslateUncorpusExp1Lm(
    translate_multilingual.TranslateMultilingualLmProblem):
  """Problem spec for UNCorpus language modeling, experiment 1."""

  def source_data_files(self, dataset_split):
    """Files to be passed to compile_data."""
    if dataset_split == problem.DatasetSplit.TRAIN:
      return _UNC_TRAIN_DATASETS_EXP1_LM
    else:
      return _UNC_TEST_DATASETS_EXP1_LM

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    compile_data_fn = functools.partial(_compile_data,
                                        FLAGS.uncorpus_orig_data_exp1_lm)
    return self._generate_samples(data_dir, tmp_dir, dataset_split,
                                  compile_data_fn=compile_data_fn)

  def generate_text_for_vocab(self, data_dir, tmp_dir):
    return self._generate_text_for_vocab(
        data_dir,
        tmp_dir,
        datapath=FLAGS.uncorpus_orig_data_exp1_lm,
        parse_lines_fn=_parse_lines)


@registry.register_problem
class TranslateUncorpusExp2Lm(
    translate_multilingual.TranslateMultilingualLmProblem):
  """Problem spec for UNCorpus language modeling, experiment 2."""

  def source_data_files(self, dataset_split):
    """Files to be passed to compile_data."""
    if dataset_split == problem.DatasetSplit.TRAIN:
      return _UNC_TRAIN_DATASETS_EXP2_LM
    else:
      return _UNC_TEST_DATASETS_EXP2_LM

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    compile_data_fn = functools.partial(_compile_data,
                                        FLAGS.uncorpus_orig_data_exp2_lm)
    return self._generate_samples(data_dir, tmp_dir, dataset_split,
                                  compile_data_fn=compile_data_fn)

  def generate_text_for_vocab(self, data_dir, tmp_dir):
    return self._generate_text_for_vocab(
        data_dir,
        tmp_dir,
        datapath=FLAGS.uncorpus_orig_data_exp2_lm,
        parse_lines_fn=_parse_lines)
