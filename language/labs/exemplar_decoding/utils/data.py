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
"""Data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from functools import partial  # pylint: disable=g-importing-member
import re

import numpy as np
import tensorflow as tf

SEPARATOR = "</h>"
SENTENCE_START = "<s>"
SENTENCE_END = "</s>"

PAD_TOKEN = "[PAD]"
UNKNOWN_TOKEN = "[UNK]"
START_DECODING = "[START]"
STOP_DECODING = "[STOP]"

SPECIAL_TOKENS = [
    STOP_DECODING,
    SENTENCE_START,
    SENTENCE_END,
    SEPARATOR,
    UNKNOWN_TOKEN,
    PAD_TOKEN,
    START_DECODING
]

INPUT_DTYPES = {
    "src_inputs": tf.int32,
    "src_len": tf.int32,
    "neighbor_ids": tf.int32,
    "neighbor_inputs": tf.int32,
    "neighbor_binary": tf.int32,
    "neighbor_len": tf.int32,
    "decoder_inputs": tf.int32,
    "decoder_len": tf.int32,
    "targets": tf.int32,
    "reference": tf.string,
    "reference_len": tf.int32
}

INPUT_SHAPES = {
    "src_inputs": [None],
    "src_len": [],
    "neighbor_ids": [None],
    "neighbor_inputs": [None],
    "neighbor_binary": [None],
    "neighbor_len": [],
    "decoder_inputs": [None],
    "decoder_len": [],
    "targets": [None],
    "reference": [None],
    "reference_len": []
}


# Used for BPE.
PATTERN = re.compile(r"(@@ )|(@@ ?$)")


def remove_repetive_unigram(text):
  ret = []
  last_token = START_DECODING
  tokens = text.split()
  for token in tokens:
    if token != last_token:
      ret.append(token)
      last_token = token
  return " ".join(ret)


def recover_tokenization(string):
  return PATTERN.sub("", string.strip())


def id2text(seq, vocab, use_bpe=False):
  tokens = [vocab.id2word(i) for i in seq]
  tokens = [x for x in tokens
            if x not in [STOP_DECODING, SEPARATOR, SENTENCE_START,
                         PAD_TOKEN, START_DECODING]]
  text = " ".join(tokens)
  if use_bpe:
    text = recover_tokenization(text)
  return remove_repetive_unigram(text)


def input_function(is_train, vocab, hps):
  """Input function."""
  if is_train:
    data_path = hps.train_path
  else:
    data_path = hps.dev_path

  dataset = tf.data.Dataset.from_generator(
      partial(generate_tensors,
              path=data_path,
              vocab=vocab,
              hps=hps,
              is_train=is_train),
      INPUT_DTYPES, INPUT_SHAPES)

  if is_train:
    dataset = dataset.shuffle(hps.batch_size * 1000,
                              reshuffle_each_iteration=True)
    dataset = dataset.repeat()
  if hps.dataset == "giga":
    if hps.encode_neighbor or hps.model == "nn2seq":
      bucket_boundaries = [50, 100, 150, 200, 250, 300]
    else:
      bucket_boundaries = [25, 50, 75, 100, 125, 150]
  elif hps.dataset == "nyt":
    bucket_boundaries = [100, 300, 500, 700, 900]
  elif hps.dataset == "cnnd":
    bucket_boundaries = [100, 200, 300, 400, 500, 600]

  padding_values = {
      "src_inputs": vocab.word2id(PAD_TOKEN),
      "src_len": vocab.word2id(PAD_TOKEN),
      "neighbor_ids": vocab.word2id(PAD_TOKEN),
      "neighbor_inputs": vocab.word2id(PAD_TOKEN),
      "neighbor_binary": vocab.word2id(PAD_TOKEN),
      "neighbor_len": vocab.word2id(PAD_TOKEN),
      "decoder_inputs": vocab.word2id(PAD_TOKEN),
      "decoder_len": vocab.word2id(PAD_TOKEN),
      "targets": vocab.word2id(PAD_TOKEN),
      "reference": PAD_TOKEN,
      "reference_len": vocab.word2id(PAD_TOKEN)
  }
  if is_train:
    dataset = dataset.apply(tf.contrib.data.bucket_by_sequence_length(
        element_length_func=lambda x: x["src_len"],
        bucket_boundaries=bucket_boundaries,
        bucket_batch_sizes=[hps.batch_size]*(len(bucket_boundaries)+1),
        padded_shapes=INPUT_SHAPES,
        padding_values=padding_values))
  else:
    dataset = dataset.apply(tf.contrib.data.bucket_by_sequence_length(
        element_length_func=lambda x: x["src_len"],
        bucket_boundaries=[100000],
        bucket_batch_sizes=[hps.batch_size]*2,
        padded_shapes=INPUT_SHAPES,
        padding_values=padding_values))

  dataset = dataset.prefetch(hps.batch_size)
  features = dataset.make_one_shot_iterator().get_next()
  return features, features["targets"]


def truncate_and_shift_right(sequence, max_len, start_id, stop_id):
  inp = [start_id] + sequence[:]
  target = sequence[:]
  if len(inp) > max_len:
    inp = inp[:max_len]
    target = target[:max_len]
  else:
    target.append(stop_id)
  return inp, target


def sample_neighbor(size=3):
  """Randomly sample a neighbor to encode."""
  assert size <= 3
  if size == 1:
    return 0
  f = np.random.sample()
  if size == 2:
    return 0 if f <= 0.6 else 1
  if size == 3:
    if f <= 0.6:
      return 0
    elif f <= 0.85:
      return 1
    else:
      return 2
  assert False, "should not reach here"


def generate_examples(base_path, hps, is_train):
  """Read data.

  Args:
      base_path: base path to the files. one instance per line.
      hps: hyperparameters
      is_train: Is training or not.
  Returns:
      A list of the instances.
  """

  src_lst, tgt_lst = [], []
  neighbor_lst, neighbor_binary_lst, neighbor_id_lst = [], [], []
  with tf.gfile.Open("%s/src" % base_path, "r") as fin:
    for line in fin:
      src = line.strip().decode("utf-8").split()
      # src = line.strip().split()
      src_lst.append(src)
  with tf.gfile.Open("%s/tgt" % base_path, "r") as fin:
    for line in fin:
      tgt = line.strip().decode("utf-8").split()
      # tgt = line.strip().split()
      tgt_lst.append(tgt)
  if hps.random_neighbor:
    with tf.gfile.Open("%s/random_neighbor_id" % base_path, "r") as fin:
      for line in fin:
        ids = [int(x) for x in line.strip().split(" ")]
        neighbor_id_lst.append(ids)
  else:
    if hps.use_cluster:
      path = "%s/cluster_id" % base_path
    else:
      path = "%s/neighbor_id" % base_path
    with tf.gfile.Open(path, "r") as fin:
      for line in fin:
        ids = [int(x) for x in line.strip().split(" ")]
        neighbor_id_lst.append(ids)
  with tf.gfile.Open(
      "%s/neighbor.cleaned" % (base_path), "r") as fin:
    for line in fin:
      line = line.strip()
      neighbors = line.decode("utf-8").split(SEPARATOR)
      # neighbors = line.split(SEPARATOR)
      neighbor = [x.strip() for x in neighbors]
      if hps.sample_neighbor and is_train:
        size = min(3, len(neighbors))
        neighbor = neighbors[sample_neighbor(size=size)].split()
      else:
        neighbor = neighbors[0].split()
      # neighbor = line.strip().split()
      neighbor_lst.append(neighbor)
  if hps.binary_neighbor:
    with tf.gfile.Open(
        "%s/neighbor.binary" % (base_path), "r") as fin:
      for line in fin:
        line = line.strip()
        neighbors = line.split(SEPARATOR)
        neighbor = [x.strip() for x in neighbors]
        if hps.sample_neighbor and is_train:
          assert False
          size = min(3, len(neighbors))
          neighbor = neighbors[sample_neighbor(size=size)].split()
        else:
          neighbor = [int(x) for x in neighbors[0].split()]
        # neighbor = line.strip().split()
        neighbor_binary_lst.append(neighbor)
  else:
    neighbor_binary_lst = [[0] * len(x) for x in tgt_lst]

  assert len(src_lst) == len(tgt_lst)
  assert len(src_lst) == len(neighbor_id_lst)
  assert len(src_lst) == len(neighbor_lst)
  assert len(src_lst) == len(neighbor_binary_lst)
  examples = []
  for s, t, n_id, n, nb in zip(
      src_lst,
      tgt_lst,
      neighbor_id_lst,
      neighbor_lst,
      neighbor_binary_lst):
    examples.append(
        {"src": s, "tgt": t, "neighbor_ids": n_id, "neighbor": n,
         "neighbor_binary": nb})
  return examples


def generate_tensors(path, vocab, hps, is_train):
  """Generate Tensors."""

  start_decoding = vocab.word2id(START_DECODING)
  stop_decoding = vocab.word2id(STOP_DECODING)

  examples = generate_examples(base_path=path, hps=hps, is_train=is_train)
  for example in examples:
    src, tgt, neighbor_ids, neighbor, neighbor_binary = (
        example["src"],
        example["tgt"],
        example["neighbor_ids"],
        example["neighbor"],
        example["neighbor_binary"])

    src_enc_inputs = [vocab.word2id(w) for w in src if w != SENTENCE_START]
    if not src_enc_inputs:
      src_enc_inputs = [vocab.word2id(SENTENCE_END)]
    if len(src_enc_inputs) > hps.max_enc_steps:
      src_enc_inputs = src_enc_inputs[:hps.max_enc_steps]
    src_enc_len = len(src_enc_inputs)

    neighbor_enc_inputs = [
        vocab.word2id(w) for w in neighbor if w != SENTENCE_START]
    if not neighbor_enc_inputs:
      neighbor_enc_inputs = [vocab.word2id(SENTENCE_END)]

    if len(neighbor_enc_inputs) > hps.max_dec_steps:
      neighbor_enc_inputs = neighbor_enc_inputs[:hps.max_enc_steps]
    neighbor_enc_len = len(neighbor_enc_inputs)
    assert src_enc_len > 0, src
    assert neighbor_enc_len > 0, neighbor

    if hps.binary_neighbor:
      assert len(neighbor_binary) == neighbor_enc_len

    tgt_ids = [vocab.word2id(w) for w in tgt if w != SENTENCE_START]
    # print (tgt_ids)
    for w in tgt:
      if w != SENTENCE_START:
        assert vocab.word2id(w) < hps.output_size + vocab.offset
    dec_inputs, targets = truncate_and_shift_right(
        tgt_ids, hps.max_dec_steps, start_decoding, stop_decoding)
    dec_len = len(dec_inputs)

    if hps.use_bpe:
      reference = " ".join(
          [x for x in example["tgt"] if x not in SENTENCE_START])
      reference = recover_tokenization(reference).split()
      reference_len = len(reference)
    else:
      reference = [x for x in example["tgt"] if x not in SENTENCE_START]
      reference_len = len(reference)
    yield {
        "src_inputs": src_enc_inputs,
        "src_len": src_enc_len,
        "neighbor_ids": neighbor_ids[:hps.num_neighbors],
        "neighbor_inputs": neighbor_enc_inputs,
        "neighbor_binary": neighbor_binary,
        "neighbor_len": neighbor_enc_len,
        "decoder_inputs": dec_inputs,
        "decoder_len": dec_len,
        "targets": targets,
        "reference": reference,
        "reference_len": reference_len
    }


class Vocab(object):
  """Vocabulary."""

  def __init__(self, vocab_file, max_size, dataset):
    self._word_to_id = {}
    self._id_to_word = {}
    self._count = 0

    self._word_to_id[STOP_DECODING] = self._count
    self._id_to_word[self._count] = STOP_DECODING
    self._count += 1
    self.offset = 1

    if dataset == "cnnd" or dataset == "nyt":

      self._word_to_id[SENTENCE_END] = self._count
      self._id_to_word[self._count] = SENTENCE_END
      self._count += 1
      self.offset = 2

    with tf.gfile.Open(vocab_file, "r") as vocab_f:
      for line in vocab_f:
        w = line.decode("utf-8").strip()
        if w in SPECIAL_TOKENS:
          continue
        self._word_to_id[w] = self._count
        self._id_to_word[self._count] = w
        self._count += 1
        if max_size != 0 and self._count >= max_size:
          break

    for w in SPECIAL_TOKENS:
      if w in self._word_to_id:
        continue
      self._word_to_id[w] = self._count
      self._id_to_word[self._count] = w
      self._count += 1

    self._unknown_id = self._word_to_id[UNKNOWN_TOKEN]

    self._tok_vocab = {}
    for w in SPECIAL_TOKENS:
      self._tok_vocab[w] = len(self._tok_vocab)

    with tf.gfile.Open("%s_tok" % vocab_file, "r") as vocab_f:
      for line in vocab_f:
        w = line.decode("utf-8").strip()
        self._tok_vocab[w] = len(self._tok_vocab)

  def word2id(self, word):
    assert word in self._word_to_id, word
    return self._word_to_id.get(word, self._unknown_id)

  def id2word(self, word_id):
    return self._id_to_word[word_id]

  def size(self):
    return self._count

  def is_token(self, w):
    return w in self._tok_vocab
