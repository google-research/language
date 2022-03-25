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
"""Script to train our recurrent model on BoolQ or MultiNLI."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import json
from absl import flags
from language.boolq.utils import best_checkpoint_exporter
from language.boolq.utils import ops
from language.boolq.utils import tokenization
from language.common.utils import experiment_utils
from language.common.inputs import char_utils
from language.common.inputs import embedding_utils
from language.common.layers import common_layers
from language.common.layers import cudnn_layers
from language.common.utils import tensor_utils
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator
from tensorflow.contrib import lookup as contrib_lookup

# Dataset parameters, these need to be pointed at the appropriate targets

flags.DEFINE_enum("dataset", "boolq", ["boolq", "multinli"],
                  "run on BoolQ instead of MultiNLI")

flags.DEFINE_string("train_data_path", None,
                    "Path to boolq/multinli training data")

flags.DEFINE_string("dev_data_path", None, "Path to boolq/multinli eval data")

flags.DEFINE_string(
    "fasttext_embeddings", None, "Path to fasttext wiki-news embeddings "
    "without any header lines")

flags.DEFINE_bool(
    "train", True, "Train a model, otherwise evaluate a model"
    " (set `checkpoint_file` to evaluate from an"
    " existing checkpoint.")

# Training parameters (our paper uses the defaults)

flags.DEFINE_integer("max_passage_len", 512,
                     "Max number of tokens for BoolQ passages.")

flags.DEFINE_float("learning_rate", 0.001, "Adam learning rate")

flags.DEFINE_string("checkpoint_file", None, "Checkpoint file to initialize on")

# Model parameters (our paper uses the defaults)

flags.DEFINE_integer("max_vocab_size", 100000,
                     "Maximum size of the vocabulary.")

flags.DEFINE_bool("lowercase", True, "lowercase text")

flags.DEFINE_float("embed_dropout_rate", 0.25, "Dropout after word embeddings")

flags.DEFINE_float("dropout_rate", 0.2, "Dropout between layers")

flags.DEFINE_integer("lstm_dim", 200, "Size of LSTM layers")

flags.DEFINE_integer("ffn_dim", 100, "Size of top fully connected layer")

FLAGS = flags.FLAGS


def load_embeddings():
  """Load the fastText embeddings."""
  return embedding_utils.PretrainedWordEmbeddings(
      lowercase=FLAGS.lowercase,
      embeddings_path=FLAGS.fasttext_embeddings,
      max_vocab_size=FLAGS.max_vocab_size,
      skip_header=True)


def load_boolq_file(filename, num_par=2):
  """Build a tf.data.Data from a file of boolq examples."""
  tokenizer = tokenization.NltkTokenizer()
  examples = []
  with tf.gfile.Open(filename) as f:
    for line in f:
      obj = json.loads(line)
      context = tokenizer.tokenize(obj["passage"])
      if FLAGS.max_passage_len:
        context = context[:FLAGS.max_passage_len]
      question = tokenizer.tokenize(obj["question"])
      examples.append((question, context, obj["answer"]))

  def get_data():
    out = list(examples)
    np.random.shuffle(out)
    return out

  ds = tf.data.Dataset.from_generator(get_data, (tf.string, tf.string, tf.bool),
                                      ([None], [None], []))

  def to_dict(p, h, label):
    return {"hypothesis": p, "premise": h, "label": label}

  return ds.map(to_dict, num_parallel_calls=num_par)


NLI_LABEL_MAP = {"contradiction": 0, "entailment": 1, "neutral": 2}


def _nli_line_to_tensors(tf_line, tokenizer):
  """Map a tensor line from a NLI file to tensor dictionary."""
  def _tensorize(line):
    example = json.loads(line)
    label = NLI_LABEL_MAP.get(example["gold_label"], -1)
    return (tokenizer.tokenize(example["sentence1"]),
            tokenizer.tokenize(example["sentence2"]), example["pairID"],
            np.int32(label))

  tf_sentence1, tf_sentence2, tf_id, tf_label = (
      tensor_utils.shaped_py_func(
          func=_tensorize,
          inputs=[tf_line],
          types=[tf.string, tf.string, tf.string, tf.int32],
          shapes=[[None], [None], [], []],
          stateful=False))
  return {
      "premise": tf_sentence1,
      "hypothesis": tf_sentence2,
      "id": tf_id,
      "label": tf_label
  }


def load_nli_file(data_path, num_par=2):
  """Build a tf.data.Data from a file of NLI examples."""
  tokenizer = tokenization.NltkTokenizer()
  dataset = tf.data.TextLineDataset(data_path)
  dataset = dataset.map(
      functools.partial(_nli_line_to_tensors, tokenizer=tokenizer),
      num_parallel_calls=num_par)
  dataset = dataset.filter(lambda x: tf.greater_equal(x["label"], 0))
  return dataset


def load_data(is_train, num_par=4):
  """Loads the tf.data.Dataset to run on."""
  if is_train:
    src = FLAGS.train_data_path
  else:
    src = FLAGS.dev_data_path

  if src is None:
    raise ValueError("Missing data path")

  if FLAGS.dataset == "boolq":
    return load_boolq_file(src, num_par)
  else:
    return load_nli_file(src, num_par)


def build_tensorize_text_fn(embeddings):
  """Builds a function to turn text into word/char ids."""
  tbl = contrib_lookup.index_table_from_tensor(
      mapping=embeddings.get_vocab(), num_oov_buckets=1)

  def fn(string_tensor):
    """Builds the output tensor dictionary."""
    out = {}
    if FLAGS.lowercase:
      string_tensor = ops.lowercase_op(string_tensor)
    out["wids"] = tf.to_int32(tbl.lookup(string_tensor))
    out["cids"] = char_utils.batch_word_to_char_ids(string_tensor, 50)
    out["len"] = tf.shape(string_tensor)[-1]
    return out

  return fn


def embed_text(tensors, embeddings):
  """Build embeddings using the word/char ids from `build_tensorize_text_fn`."""
  wids = tensors["wids"]
  cids = tensors["cids"]

  embedding_weights = embeddings.get_initialized_params(trainable=False)
  word_vecs = tf.nn.embedding_lookup(embedding_weights, wids)
  char_emb = common_layers.character_cnn(cids)
  return tf.concat([word_vecs, char_emb], -1)


def apply_lstm(x, seq_len):
  """Run a bi-directional LSTM over the `x`.

  Args:
    x: <tf.float32>[batch, seq_len, dim]
    seq_len: <tf.int32>[batch] for None, sequence lengths of `seq2`

  Returns:
    out, <tf.float32>[batch, seq_len, out_dim]
  """
  return cudnn_layers.stacked_bilstm(
      input_emb=x,
      input_len=seq_len,
      hidden_size=FLAGS.lstm_dim,
      num_layers=1,
      dropout_ratio=0.0,
      mode=tf_estimator.ModeKeys.TRAIN,
      use_cudnn=None)


def apply_highway_lstm(x, seq_len):
  """Run a bi-directional LSTM with highway connections over `x`.

  Args:
    x: <tf.float32>[batch, seq_len, dim]
    seq_len: <tf.int32>[batch] for None, sequence lengths of `seq2`

  Returns:
    out, <tf.float32>[batch, seq_len, out_dim]
  """
  lstm_out = apply_lstm(x, seq_len)
  proj = ops.affine(x, FLAGS.lstm_dim * 4, "w", bias_name="b")
  gate, transform = tf.split(proj, 2, 2)
  gate = tf.sigmoid(gate)
  transform = tf.tanh(transform)
  return lstm_out * gate + (1 - gate) * transform


def compute_attention(t1, t2):
  """Build an attention matrix between 3-tensors `t1` and `t2`.

  Args:
    t1: <tf.float32>[batch, seq_len1, dim1]
    t2: <tf.float32>[batch, seq_len2, dim2]

  Returns:
    the similarity scores <tf.float32>[batch, seq_len1, seq_len2]
  """
  dim = t1.shape.as_list()[2]
  init = tf.constant_initializer(1.0 / dim)

  t1_logits = ops.last_dim_weighted_sum(t1, "t1_w")
  t2_logits = ops.last_dim_weighted_sum(t2, "t2_w")

  dot_w = tf.get_variable(
      "dot_w", shape=dim, initializer=init, dtype=tf.float32)
  # Compute x * dot_weights first, then batch mult with x
  dots = t1 * tf.expand_dims(tf.expand_dims(dot_w, 0), 0)
  dot_logits = tf.matmul(dots, t2, transpose_b=True)

  return dot_logits + \
         tf.expand_dims(t1_logits, 2) + \
         tf.expand_dims(t2_logits, 1)


def mask_attention(attention, seq_len1, seq_len2):
  """Masks an attention matrix.

  Args:
    attention: <tf.float32>[batch, seq_len1, seq_len2]
    seq_len1: <tf.int32>[batch]
    seq_len2: <tf.int32>[batch]

  Returns:
    the masked scores <tf.float32>[batch, seq_len1, seq_len2]
  """
  dim1 = tensor_utils.shape(attention, 1)
  dim2 = tensor_utils.shape(attention, 2)
  m1 = tf.sequence_mask(seq_len1, dim1)
  m2 = tf.sequence_mask(seq_len2, dim2)
  joint_mask = tf.logical_and(tf.expand_dims(m1, 2), tf.expand_dims(m2, 1))
  return ops.mask_logits(attention, joint_mask)


def pool(embed, seq_len):
  """Pool `embed` along the second dimension.

  Args:
    embed: <tf.float32>[batch, seq_len, dim]
    seq_len:  <tf.int32>[batch] sequence lengths of `embed`

  Returns:
    out: <tf.float32>[batch, dim]
  """
  attention_w = ops.last_dim_weighted_sum(embed, "w")
  attention_w = ops.mask_logits(attention_w, seq_len)
  attention_w = tf.expand_dims(tf.nn.softmax(attention_w), 1)

  # [batch, 1, len] * [batch, len, dim] -> [batch, 1, dim]
  return tf.squeeze(tf.matmul(attention_w, embed), 1)


def variational_dropout(x, dropout_rate, is_train):
  if is_train:
    shape = tensor_utils.shape(x)
    return tf.nn.dropout(x, 1.0 - dropout_rate, [shape[0], 1, shape[2]])
  else:
    return x


def dropout(x, dropout_rate, is_train):
  if is_train:
    return tf.nn.dropout(x, 1.0 - dropout_rate)
  else:
    return x


def predict(is_train, embeddings, premise_tensors, hypothesis_tensors):
  """Compute the class logit predictions.

  Args:
    is_train: bool, are we in train mode
    embeddings: `embedding_utils.PretrainedWordEmbeddings` to embed with
    premise_tensors: batched premise tensors from `build_tensorize_text_fn`
    hypothesis_tensors: batched hypothesis tensors from
      `build_tensorize_text_fn`

  Returns:
    logits: <tf.float32>[batch, 3] Scores for each class (for checkpoint
            comparability, the BoolQ models also produce a neutral logit)
  """
  with tf.variable_scope("embed"):
    premise = embed_text(premise_tensors, embeddings)
    premise_lens = premise_tensors["len"]
  with tf.variable_scope("embed", reuse=True):
    hypothesis = embed_text(hypothesis_tensors, embeddings)
    hypothesis_len = hypothesis_tensors["len"]

  e_drop = FLAGS.embed_dropout_rate
  premise = variational_dropout(premise, e_drop, is_train)
  hypothesis = variational_dropout(hypothesis, e_drop, is_train)

  with tf.variable_scope("embed/encode-text/layer-1"):
    premise = apply_highway_lstm(premise, premise_lens)
  with tf.variable_scope("embed/encode-text/layer-1", reuse=True):
    hypothesis = apply_highway_lstm(hypothesis, hypothesis_len)

  with tf.variable_scope("fuse"):
    with tf.variable_scope("attention"):
      atten = compute_attention(premise, hypothesis)
      atten = mask_attention(atten, premise_lens, hypothesis_len)

    attended_h = tf.matmul(tf.nn.softmax(atten), hypothesis)
    attended_p = tf.matmul(
        tf.nn.softmax(tf.transpose(atten, [0, 2, 1])), premise)
    premise = tf.concat([premise, attended_h, attended_h * premise], 2)
    hypothesis = tf.concat([hypothesis, attended_p, attended_p * hypothesis], 2)

  with tf.variable_scope("post-process/layer-0"):
    premise = apply_highway_lstm(premise, premise_lens)
  with tf.variable_scope("post-process/layer-0", reuse=True):
    hypothesis = apply_highway_lstm(hypothesis, hypothesis_len)

  drop = FLAGS.dropout_rate
  premise = variational_dropout(premise, drop, is_train)
  hypothesis = variational_dropout(hypothesis, drop, is_train)

  with tf.variable_scope("pool/atten-pool"):
    premise = pool(premise, premise_lens)
  with tf.variable_scope("pool/atten-pool", reuse=True):
    hypothesis = pool(hypothesis, hypothesis_len)

  joint_embed = tf.concat([premise, hypothesis], 1)
  with tf.variable_scope("post-processs-pooled/layer-1"):
    joint_embed = tf.layers.dense(
        joint_embed, units=FLAGS.ffn_dim, activation="relu")

  joint_embed = dropout(joint_embed, drop, is_train)

  with tf.variable_scope("predict"):
    # Use three classes even for BoolQ so the checkpoint are compatible
    n_classes = 3
    logits = tf.layers.dense(joint_embed, units=n_classes, activation=None)

  return logits


def get_train_op(loss):
  """Build the gradient descent op for the model on `loss`."""
  gs = tf.train.get_global_step()
  is_boolq = FLAGS.dataset == "boolq"
  lr = tf.train.exponential_decay(
      global_step=gs,
      learning_rate=FLAGS.learning_rate,
      staircase=True,
      decay_steps=50 if is_boolq else 100,
      decay_rate=0.999)
  opt = tf.train.AdamOptimizer(lr)
  grad_and_vars = opt.compute_gradients(loss)
  return opt.apply_gradients(grad_and_vars, tf.train.get_global_step())


def load_batched_dataset(is_train, embeddings):
  """Loads a dataset that has been batched and preprocessed."""
  tensorize_text_fn = build_tensorize_text_fn(embeddings)
  unbatched = load_data(is_train)

  def tensorize(x):
    x["premise"] = tensorize_text_fn(x["premise"])
    x["hypothesis"] = tensorize_text_fn(x["hypothesis"])
    return x

  unbatched = unbatched.map(tensorize)

  hist_bins = list(range(5, 500, 5))
  batched = unbatched.apply(
      ops.bucket_by_quantiles(lambda x: x["premise"]["len"], FLAGS.batch_size,
                              10, hist_bins))
  if is_train:
    batched = batched.shuffle(1000, reshuffle_each_iteration=True)
    batched = batched.repeat()

  # Get (features, label) format for tf.estimator
  return batched.map(lambda x: (x, x["label"]))


def train():
  """Train the model."""
  embeddings = load_embeddings()

  # Need a named parameter `param` since this will be called
  # with named arguments, so pylint: disable=unused-argument
  def model_function(features, labels, mode, params):
    """Builds the `tf.estimator.EstimatorSpec` to train/eval with."""
    is_train = mode == tf_estimator.ModeKeys.TRAIN
    logits = predict(is_train, embeddings, features["premise"],
                     features["hypothesis"])

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=tf.to_int32(labels), logits=logits)
    loss = tf.reduce_mean(loss)
    if mode == tf_estimator.ModeKeys.TRAIN:
      train_op = get_train_op(loss)
    else:
      # Don't build the train_op unnecessarily, since the ADAM variables can
      # cause problems with loading checkpoints on CPUs.
      train_op = None
    metrics = dict(
        accuracy=tf.metrics.accuracy(
            tf.argmax(logits, 1, output_type=tf.int32), tf.to_int32(labels)))

    checkpoint_file = FLAGS.checkpoint_file
    if checkpoint_file is None:
      scaffold = None
    else:
      saver = tf.train.Saver(tf.trainable_variables())

      def _init_fn(_, sess):
        saver.restore(sess, checkpoint_file)

      scaffold = tf.train.Scaffold(init_fn=_init_fn)

    return tf_estimator.EstimatorSpec(
        mode=mode,
        scaffold=scaffold,
        loss=loss,
        predictions=None,
        train_op=train_op,
        eval_metric_ops=metrics)

  def compare_fn(best_eval_result, current_eval_result):
    return best_eval_result["accuracy"] < current_eval_result["accuracy"]

  exporter = best_checkpoint_exporter.BestCheckpointExporter(
      event_file_pattern="eval_default/*.tfevents.*",
      compare_fn=compare_fn,
  )

  experiment_utils.run_experiment(
      model_fn=model_function,
      train_input_fn=lambda: load_batched_dataset(True, embeddings),
      eval_input_fn=lambda: load_batched_dataset(False, embeddings),
      exporters=[exporter])


def evaluate():
  """Evaluate a model on the dev set."""
  sess = tf.Session()
  tf.logging.info("Building graph...")

  embeddings = load_embeddings()
  tf_data = load_batched_dataset(False, embeddings)
  it = tf_data.make_initializable_iterator()
  features, labels = it.get_next()

  logits = predict(False, embeddings, features["premise"],
                   features["hypothesis"])
  accuracy, update_ops = tf.metrics.accuracy(
      tf.argmax(logits, 1, output_type=tf.int32), tf.to_int32(labels))

  tf.logging.info("Running initializers...")
  checkpoint_file = FLAGS.checkpoint_file
  if checkpoint_file is not None:
    saver = tf.train.Saver(tf.trainable_variables())
    tf.logging.info("Restoring from checkpoint: " + checkpoint_file)
    saver.restore(sess, checkpoint_file)
  else:
    tf.logging.warning("No checkpoint given, evaling model with random weights")
    sess.run(tf.global_variables_initializer())
  sess.run(tf.local_variables_initializer())
  sess.run(tf.tables_initializer())
  sess.run(it.initializer)

  tf.logging.info("Starting loop....")
  while True:
    try:
      sess.run(update_ops)
    except tf.errors.OutOfRangeError:
      break
  tf.logging.info("Done")

  accuracy = sess.run(accuracy)
  print("Accuracy: %f" % accuracy)


def main(_):
  flags.mark_flag_as_required("fasttext_embeddings")
  if FLAGS.train:
    train()
  else:
    evaluate()


if __name__ == "__main__":
  tf.app.run()
