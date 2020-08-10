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
# Lint as: python3
"""Evaluate coherent permutation discrimination model on baseline."""

from collections import namedtuple  # pylint: disable=g-importing-member
import csv
import json
import random
import time
from absl import app
from absl import flags
from bert import modeling
from bert import tokenization
import numpy as np
import tensorflow.compat.v1 as tf


FLAGS = flags.FLAGS

flags.DEFINE_integer("num_slices", 25, "Divide dataset into this many slices.")

flags.DEFINE_integer("slice_index", 0, "Evaluate this slice.")

flags.DEFINE_bool("do_reduce", False, "Collect eval numbers and aggregate.")

flags.DEFINE_string("output_dir", None, "The directory to write the output.")

flags.DEFINE_string("model_weights", None, "The pretrained BERT weights.")

flags.DEFINE_string("vocab_file", "", "BERT vocab file.")

flags.DEFINE_string("bert_config", "", "BERT config file.")

flags.DEFINE_string("data_dir", "", "Data directory.")

# pylint: disable=invalid-name

PARA_BREAK = "<para_break>"
model_weights = FLAGS.model_weights
vocab_file = FLAGS.vocab_file


def _restore_checkpoint(init_checkpoint):
  """Restore parameters from checkpoint."""
  tvars = tf.trainable_variables()
  (assignment_map,
   _) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
  tf.train.init_from_checkpoint(init_checkpoint, assignment_map)


def read_data(data_file):
  """Read permutation discrimination eval dataset.

  Args:
    data_file: data file to read from

  Returns:
    A list of examples (21 permutations where the first is the correct one).
  """
  perms = []
  with tf.gfile.Open(data_file, "r") as handle:
    tsvreader = csv.reader(handle, delimiter="\t")
    for line in tsvreader:
      if line:
        perms.append(line)

  examples = []
  for perm in perms:
    target = perm[1].split("<PUNC>")
    distractors = [
        list(para.split("<PUNC>")) for para in perm[2].split("<BREAK>")
    ]
    examples.append([target] + distractors)

  return examples


def read_permutations():
  perm_file = FLAGS.data_dir + "/wikiAperm.test.txt"
  return read_data(perm_file)


def disc_coherence_scores(examples, tokenizer):
  """Get discriminative coherence scores."""

  tf.reset_default_graph()
  placeholders, model = create_cpc_model_and_placeholders(2)
  with tf.compat.v1.Session() as sess:
    _restore_checkpoint(FLAGS.model_weights)
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    sess.run(init_op)

    # To classify which of 2 paragraphs is coherent, we compute the probability
    # of each paragraph and then select the one with higher likelihood as the
    # coherent paragraph.
    # Since our model is bi-directional, we estimate the probability of a
    # paragraph by taking the mean of the probabilities each triple of sentences
    # where we set the middle sentences as the context and the sentence before
    # and after as the targets.
    all_scores = []
    for paragraphs in examples:
      scores = []
      for paragraph in paragraphs:
        partial_paragraph_probs = []
        for i in range(len(paragraph) - 2):
          sents = paragraph[i:i + 3]
          context = sents[1]
          targets = sents[:1] + sents[2:3]
          e = create_cpc_input_from_text(
              tokenizer, context, targets, [], group_size=2)
          input_map = {
              placeholders.input_ids: e.tokens,
              placeholders.input_mask: e.mask,
              placeholders.segment_ids: e.seg_ids,
              placeholders.softmax_mask: [True] * 8
          }

          results = sess.run([model.logits], feed_dict=input_map)
          # partial_prob = np.sum(np.diagonal(results[0][0][3:5, :]))
          diag = np.diagonal(results[0][0][3:5, :])
          diag = 1 / (1 + np.exp(-diag))
          partial_prob = np.sum(diag)
          partial_paragraph_probs.append(partial_prob)
        scores.append(np.sum(partial_paragraph_probs))
      all_scores.append(scores)
      print("Finished", len(all_scores))
      tf.logging.info("Finished %d" % len(all_scores))
    acc = np.sum([list(np.argsort(score)).index(0) for score in all_scores
                 ]) / float(len(all_scores) * 20)
  return acc


def create_cpc_model(model, num_choices, is_training):
  """Creates a classification model.

  Args:
    model: the BERT model from modeling.py
    num_choices: number of negatives samples + 1
    is_training: training mode (bool)

  Returns:
    tuple of (loss, per_example_loss, logits, probabilities) for model
  """

  output_layer = model.get_pooled_output()

  hidden_size = output_layer.shape[-1].value

  with tf.variable_scope("cpc_loss"):

    softmax_weights = tf.get_variable(
        "softmax_weights", [hidden_size, 8],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    with tf.variable_scope("loss"):
      if is_training:
        # I.e., 0.1 dropout
        output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

      matmul_out = tf.matmul(output_layer, softmax_weights)

      logits = tf.reshape(matmul_out, (-1, num_choices, 8))
      logits = tf.transpose(logits, perm=[0, 2, 1])

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

  logits, probabilities = create_cpc_model(model, num_choices, False)

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
  for sent in sents:
    truncate_seq_pair(context, sents, max_seq_length - 3, rng)

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


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  tf.logging.info("Evaluating slice %d" % FLAGS.slice_index)

  tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case=True)
  print("Trying to read...")
  examples = read_permutations()
  print("Load data successful")

  if FLAGS.do_reduce:
    acc = 0
    found = [False] * FLAGS.num_slices
    output_filename = FLAGS.output_dir + "/disc_acc.%d.json"
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
        numerator += data["acc"] * data["length"]
    acc = numerator / total
    tf.logging.info("Accuracy: " + str(acc))
    with tf.gfile.Open(FLAGS.output_dir + "result.txt", "w") as handle:
      handle.write("Accuracy: " + str(acc))

  else:
    slice_size = int(len(examples) / FLAGS.num_slices)
    start = slice_size * FLAGS.slice_index
    end = slice_size * (FLAGS.slice_index + 1)
    if FLAGS.slice_index == (FLAGS.num_slices - 1):
      examples_to_eval = examples[start:]
    else:
      examples_to_eval = examples[start:end]
    accuracy = disc_coherence_scores(examples_to_eval, tokenizer)
    tf.logging.info(accuracy)

    output_filename = FLAGS.output_dir + "/disc_acc.%d.json" % FLAGS.slice_index
    with tf.gfile.Open(output_filename, "w") as handle:
      handle.write(
          json.dumps({
              "length": len(examples_to_eval),
              "acc": accuracy
          }))


if __name__ == "__main__":
  flags.mark_flag_as_required("num_slices")
  flags.mark_flag_as_required("slice_index")
  flags.mark_flag_as_required("output_dir")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config")
  flags.mark_flag_as_required("data_dir")
  app.run(main)
