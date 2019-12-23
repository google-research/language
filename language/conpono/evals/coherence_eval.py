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
"""Evaluate paragraph reconstruction model on baseline."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple  # pylint: disable=g-importing-member
import json
import random
import time
from absl import app
from absl import flags
from bert import modeling
from bert import tokenization
import numpy as np
from scipy import stats
import tensorflow as tf


FLAGS = flags.FLAGS

WIKIA = "wikiA"
WIKID = "wikiD"

flags.DEFINE_integer("num_slices", 25, "Divide dataset into this many slices.")

flags.DEFINE_integer("slice_index", 0, "Evaluate this slice.")

flags.DEFINE_enum("dataset", "wikiA", [WIKIA, WIKID], "Evaluate this slice.")

flags.DEFINE_bool("do_reduce", False, "Collect eval numbers and aggregate.")

flags.DEFINE_string("output_dir", None, "The directory to write the output.")

flags.DEFINE_string("model_weights", None, "Weights to BERT model.")

flags.DEFINE_string("vocab_file", "", "BERT vocab file.")

flags.DEFINE_string("bert_config", "", "BERT config file.")

flags.DEFINE_string("data_dir", "", "Data directory.")


# pylint: disable=invalid-name

PARA_BREAK = "<para_break>"
State = namedtuple("State", ["prob", "output_order", "softmax_mask", "context"])
BEAM_SIZE = 8
vocab_file = FLAGS.vocab_file


def _restore_checkpoint(init_checkpoint):
  """Restore parameters from checkpoint."""
  tvars = tf.trainable_variables()
  (assignment_map,
   _) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
  tf.train.init_from_checkpoint(init_checkpoint, assignment_map)


def read_data(data_file, long_para_max):
  """Read Wiki-A eval dataset.

  Args:
    data_file: data file to read from
    long_para_max: max size required to capture entire paragraph

  Returns:
    tuple of short_para, long_para, short_para_max, long_para_max
       short_para is a list of paragraphs max size short_para_max
       long_para is a list of paragraphs max size long_para_max

  """
  # Read from Wiki-A
  data = []
  with tf.gfile.Open(data_file, "r") as handle:
    for line in handle:
      if line.strip():
        data.append(line.strip())

  data_paragraphs = []
  para = []
  for d in data:
    if d == PARA_BREAK:
      data_paragraphs.append(para)
      para = []
    else:
      para.append(d)
  if para:
    data_paragraphs.append(para)

  short_data_paragraphs = [d for d in data_paragraphs if len(d) < 17]
  long_data_paragraphs = [d for d in data_paragraphs if len(d) >= 17]
  return short_data_paragraphs, long_data_paragraphs, 16, long_para_max


def read_wiki_a():
  data_file = FLAGS.data_dir + "/wikiA.test.txt"
  return read_data(data_file, 66)


def read_celestial_body():
  data_file = FLAGS.data_dir + "/cbody.test.txt"
  return read_data(data_file, 30)


def output_order_with_beam(paragraphs, model_size, tokenizer):
  """Compute the predicted output for each paragraph."""

  tf.reset_default_graph()
  placeholders, model = create_cpc_model_and_placeholders(model_size)
  with tf.compat.v1.Session() as sess:
    _restore_checkpoint(FLAGS.model_weights)
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    sess.run(init_op)

    # start with the first sentence
    # compute order greedily
    output_order_list = []
    for para_idx, paragraph in enumerate(paragraphs):
      beam_states = []
      tmp_softmax_mask = [True] * (len(paragraph) - 1) + [False] * (
          model_size + 1 - len(paragraph))
      tmp_output_order = []
      tmp_context = range(1)
      beam_states.append(
          State(1, tmp_output_order, tmp_softmax_mask, tmp_context))
      next_beam_states = []
      for _ in range(1, len(paragraph)):
        for prev_state in beam_states:
          cc = " ".join([paragraph[i] for i in prev_state.context])
          e = create_cpc_input_from_text(
              tokenizer, cc, paragraph[1:], [], group_size=model_size)
          while len(tokenizer.tokenize(cc)) > 400:
            prev_state.context.pop(0)
            cc = " ".join([paragraph[i] for i in prev_state.context])
            e = create_cpc_input_from_text(
                tokenizer, cc, paragraph[1:], [], group_size=model_size)
          input_map = {
              placeholders.input_ids: e.tokens,
              placeholders.input_mask: e.mask,
              placeholders.segment_ids: e.seg_ids,
              placeholders.softmax_mask: prev_state.softmax_mask
          }
          results = sess.run([model.logits], feed_dict=input_map)
          new_cands = np.argsort(results[0][0])[::-1][:BEAM_SIZE]
          for cand in new_cands:
            tmp_output_order = list(prev_state.output_order)
            tmp_softmax_mask = list(prev_state.softmax_mask)
            tmp_context = list(prev_state.context)
            tmp_output_order.append(cand)
            tmp_softmax_mask[cand] = False
            tmp_context.append(cand + 1)
            next_beam_states.append(
                State(prev_state.prob + results[0][0][cand], tmp_output_order,
                      tmp_softmax_mask, tmp_context))
        # purge states
        beam_states = sorted(
            next_beam_states, key=lambda x: x.prob, reverse=True)[:BEAM_SIZE]
        next_beam_states = []
      output_order = sorted(
          beam_states, key=lambda x: x.prob, reverse=True)[0].output_order
      # append best
      print("Completed %d / %d" % (para_idx, len(paragraphs)))
      tf.logging.info("Completed %d / %d" % (para_idx, len(paragraphs)))
      output_order_list.append(output_order)
  return output_order_list


def create_cpc_model(model, num_choices, is_training, softmax_mask):
  """Creates a classification model.

  Args:
    model: the BERT model from modeling.py
    num_choices: number of negatives samples + 1
    is_training: training mode (bool)
    softmax_mask: which options to ignore

  Returns:
    tuple of (loss, per_example_loss, logits, probabilities) for model
  """

  output_layer = model.get_pooled_output()

  hidden_size = output_layer.shape[-1].value

  with tf.variable_scope("cpc_loss"):

    softmax_weights = tf.get_variable(
        "softmax_weights", [hidden_size, 8],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    softmax_weights = tf.reshape(softmax_weights[:, 4], [hidden_size, 1])

    with tf.variable_scope("loss"):
      if is_training:
        # I.e., 0.1 dropout
        output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

      matmul_out = tf.matmul(output_layer, softmax_weights)

      logits = tf.reshape(matmul_out, (-1, num_choices))  # , 8))

      negs = tf.reshape(tf.constant([-99999.] * num_choices), [-1, num_choices])
      softmax_mask = tf.reshape(softmax_mask, [-1, num_choices])
      logits = tf.where(softmax_mask, logits, negs)
      probabilities = tf.nn.softmax(logits, axis=-1)

  return (logits, probabilities)


def create_cpc_model_and_placeholders(num_choices):
  """Build model and placeholders."""

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config)
  is_training = False
  use_one_hot_embeddings = False
  seq_length = 512

  Placeholders = namedtuple("Placeholders", [
      "input_ids", "input_mask", "segment_ids", "labels", "label_types",
      "softmax_mask"
  ])

  input_ids = tf.placeholder(dtype=tf.int32, shape=[None, seq_length])
  input_mask = tf.placeholder(dtype=tf.int32, shape=[None, seq_length])
  segment_ids = tf.placeholder(dtype=tf.int32, shape=[None, seq_length])
  labels = tf.placeholder(dtype=tf.int32, shape=[None, 8])
  label_types = tf.placeholder(dtype=tf.int32, shape=[None, 8])
  softmax_mask = tf.placeholder(dtype=tf.bool, shape=[None])
  placeholders = Placeholders(input_ids, input_mask, segment_ids, labels,
                              label_types, softmax_mask)

  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  logits, probabilities = create_cpc_model(model, num_choices, False,
                                           softmax_mask)

  Model = namedtuple("Model", ["logits", "probabilities"])
  model = Model(logits, probabilities)

  return placeholders, model


def create_cpc_input_from_text(tokenizer,
                               context,
                               sents,
                               labels,
                               group_size=32,
                               max_seq_length=512):
  """Parse text into BERT input."""

  Input = namedtuple("Input", ["tokens", "mask", "seg_ids", "labels"])

  context = tokenizer.tokenize(context)
  sents = [tokenizer.tokenize(sent) for sent in sents]

  rng = random.Random()
  new_sents = []
  for sent in sents:
    new_snt = truncate_seq_pair(context, sent, max_seq_length - 3, rng)
    new_sents.append(new_snt)
  sents = new_sents

  tokens_list, input_mask_list, seg_id_list = [], [], []
  for sent in sents:
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in context:
      tokens.append(token)
      segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    for token in sent:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)
    input_mask = [1] * len(tokens)

    tokens = tokenizer.convert_tokens_to_ids(tokens)

    while len(tokens) < max_seq_length:
      tokens.append(0)
      input_mask.append(0)
      segment_ids.append(0)
    assert len(tokens) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    tokens_list.append(tokens)
    input_mask_list.append(input_mask)
    seg_id_list.append(segment_ids)

    zero_list = [0] * max_seq_length
  while len(tokens_list) < group_size:
    tokens_list.append(zero_list)
    input_mask_list.append(zero_list)
    seg_id_list.append(zero_list)

  return Input(tokens_list, input_mask_list, seg_id_list, labels)


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
  """Truncates a pair of sequences to a maximum sequence length."""
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_num_tokens:
      break

    trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
    assert len(trunc_tokens) >= 1

    # We want to sometimes truncate from the front and sometimes from the
    # back to add more randomness and avoid biases.
    if rng.random() < 0.5:
      del trunc_tokens[0]
    else:
      trunc_tokens.pop()
  return tokens_b


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case=True)

  tf.logging.info("Evaluating slice %d" % FLAGS.slice_index)

  print("Trying to read...")
  if FLAGS.dataset == WIKIA:
    data_func = read_wiki_a
  else:
    data_func = read_celestial_body
  (short_paragraphs, long_paragraphs, short_model_size,
   large_model_size) = data_func()
  print("Load data successful")

  if FLAGS.do_reduce:
    ktau = 0
    found = [False] * FLAGS.num_slices
    output_filename = FLAGS.output_dir + "/reconstruct_ktau.%d.json"
    while not all(found):
      for i in range(FLAGS.num_slices):
        if tf.gfile.Exists(output_filename % i):
          found[i] = True
      time.sleep(5)
    total = 0
    numerator = 0
    for i in range(FLAGS.num_slices):
      with tf.gfile.Open(output_filename % i, "r") as handle:
        data = json.load(handle)
        total += data["length"]
        numerator += data["kendall_tau"] * data["length"]
    ktau = numerator / total
    print("Kendall's Tau:", ktau)  # pylint: disable=superfluous-parens
    tf.logging.info("Kendall's Tau: " + str(ktau))
    with tf.gfile.Open(FLAGS.output_dir + "result.txt", "w") as handle:
      handle.write("Kendall's Tau: " + str(ktau))

  else:
    large_model_cutoff = int(FLAGS.num_slices - (FLAGS.num_slices / 5))

    if FLAGS.slice_index < large_model_cutoff:
      slice_size = int(len(short_paragraphs) / large_model_cutoff)
      model_size = short_model_size
      start = slice_size * FLAGS.slice_index
      end = slice_size * (FLAGS.slice_index + 1)
      if FLAGS.slice_index == large_model_cutoff - 1:
        paragraphs = short_paragraphs[start:]
      else:
        paragraphs = short_paragraphs[start:end]
    else:
      model_size = large_model_size
      slice_size = int(
          len(long_paragraphs) / (FLAGS.num_slices - large_model_cutoff))
      start = slice_size * (FLAGS.slice_index - large_model_cutoff)
      end = slice_size * (FLAGS.slice_index - large_model_cutoff + 1)
      if FLAGS.slice_index == FLAGS.num_slices - 1:
        paragraphs = long_paragraphs[start:]
      else:
        paragraphs = long_paragraphs[start:end]

    pred_orders = output_order_with_beam(paragraphs, model_size, tokenizer)
    kts = []
    for pred_order in pred_orders:
      tf.logging.info(pred_order)
      kt, _ = stats.kendalltau(pred_order, range(len(pred_order)))
      kts.append(kt)
    print(np.mean(kts))  # pylint: disable=superfluous-parens
    tf.logging.info(np.mean(kts))

    output_filename = FLAGS.output_dir + "/reconstruct_ktau.%d.json" % FLAGS.slice_index
    with tf.gfile.Open(output_filename, "w") as handle:
      handle.write(
          json.dumps({
              "length": len(paragraphs),
              "kendall_tau": np.mean(kts)
          }))


if __name__ == "__main__":
  flags.mark_flag_as_required("num_slices")
  flags.mark_flag_as_required("slice_index")
  flags.mark_flag_as_required("output_dir")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config")
  flags.mark_flag_as_required("data_dir")
  app.run(main)
