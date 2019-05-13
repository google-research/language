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
"""NYT experiments."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from functools import partial  # pylint: disable=g-importing-member

from absl import flags
from language.common.utils import experiment_utils
import language.labs.exemplar_decoding.models.model_function as model_function
import language.labs.exemplar_decoding.utils.data as data
import tensorflow as tf

flags.DEFINE_string("train_path", None, "Path to train examples.")

flags.DEFINE_string("dev_path", None, "Path to eval examples.")

flags.DEFINE_string("vocab_path", None, "Path to vocabulary file.")

flags.DEFINE_string("dataset", "giga", "Dataset to use.")

flags.DEFINE_string("rnn_cell", "hyper_lstm",
                    "RNN cell to use. [`lstm`, `gru`, `hyper_lstm`]")

flags.DEFINE_string("att_type", "my",
                    "Attention type [`luong`, `bahdanau`, `hyper`, `my`].")

flags.DEFINE_bool("use_bpe", True, "Use BPE or not.")

flags.DEFINE_bool("use_copy", False, "Use copy or not.")

flags.DEFINE_bool("reuse_attention", False, "Use copy or not.")

flags.DEFINE_bool("use_bridge", True, "Use bridge or not.")

flags.DEFINE_bool("use_residual", True, "Use residual connection or not.")

flags.DEFINE_bool("random_neighbor", False, "Using random neighbors or not.")

flags.DEFINE_bool("use_cluster", False, "Use cluster or not.")

flags.DEFINE_bool("encode_neighbor", True, "Attending over neighbors or not.")

flags.DEFINE_bool("sum_neighbor", False, "Attending over neighbors or not.")

flags.DEFINE_bool("att_neighbor", False, "Attending over neighbors or not.")

flags.DEFINE_bool("binary_neighbor", False, "Binary neighbor features or not.")

flags.DEFINE_bool("tie_embedding", True,
                  "Tie softmax weights and embeddings or not.")

flags.DEFINE_integer("num_neighbors", 10, "# nearest neighbors to consider.")

flags.DEFINE_string(
    "model",
    "hypernet",
    "Model options: [`seq2seq`, `nn2seq`, `hypernet`]")

flags.DEFINE_string("trainer", "adam", "[sgd, adam, amsgrad]")

flags.DEFINE_float("learning_rate", 0.001, "Initial learning rate.")

flags.DEFINE_integer("lr_schedule", 40000,
                     "Divide the learning by 5 every this amount steps.")

flags.DEFINE_integer("total_steps", 400000,
                     "Total steps. Currently not useful.")

flags.DEFINE_float(
    "weight_decay",
    1e-2,
    "Weight_decay. It's weighted by step size. Larger values are recommended")

flags.DEFINE_integer("emb_dim", 128, "Dimension of word embeddings.")

flags.DEFINE_integer("binary_dim", 4, "Dimension of binary feature embeddings.")

flags.DEFINE_integer("neighbor_dim", 128,
                     "Dimension of nonparametric embeddings.")

flags.DEFINE_float("emb_drop", 0.1, "Embedding dropout.")

flags.DEFINE_float("out_drop", 0.2, "Output embedding dropout.")

flags.DEFINE_float("drop", 0.11, "Dropout.")

flags.DEFINE_float("encoder_drop", 0., "Encoder RNN dropout.")

flags.DEFINE_float("decoder_drop", 0., "Decoder RNN dropout.")

flags.DEFINE_integer("encoder_dim", 128,
                     "Dimension of encoder RNN hidden states.")

flags.DEFINE_integer("num_encoder_layers", 1, "# encoder RNN layers.")

flags.DEFINE_integer("decoder_dim", 128,
                     "Dimension of decoder RNN hidden states.")

flags.DEFINE_integer("num_decoder_layers", 1, "# decoder RNN layers.")

flags.DEFINE_integer("num_mlp_layers", 1, "# MLP layers.")

flags.DEFINE_integer("rank", 128, "Rank of RNN params.")

flags.DEFINE_float(
    "sigma_norm",
    1.0,
    "Normalize sigma; use 0.0 to disable; -1 to use softmax")

flags.DEFINE_float("sampling_probability", 0.0,
                   "Sampling_probability. Use 0.0 to disable.")

flags.DEFINE_integer("beam_width", 10,
                     "Beam width for beam search decoding. Use 0 to disable.")

flags.DEFINE_integer("max_enc_steps", 800,
                     "Max timesteps of encoder (max source tokens)")

flags.DEFINE_integer("max_dec_steps", 120,
                     "Max timesteps of decoder (max summary tokens)")

flags.DEFINE_integer("vocab_size", 26000, "Size of vocabulary.")

flags.DEFINE_float("max_grad_norm", 1.0, "Maximum gradient norm")

flags.DEFINE_float("length_norm", 1.0, "\alpha for length normalization.")

flags.DEFINE_float("coverage_penalty", 0.0, "\alpha for length normalization.")

flags.DEFINE_bool("predict_mode", False,
                  "Use this to run predictions on test set.")

flags.DEFINE_bool("sample_neighbor", False, "Self attention.")

FLAGS = flags.FLAGS


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  if FLAGS.model == "seq2seq":
    assert FLAGS.rnn_cell == "lstm"
    assert FLAGS.att_type != "hyper"
  if FLAGS.model == "hypernet" and FLAGS.rank != FLAGS.decoder_dim:
    print ("WARNING: recommended rank value: decoder_dim.")
  if FLAGS.att_neighbor:
    assert FLAGS.neighbor_dim == FLAGS.encoder_dim or FLAGS.att_type == "my"

  if FLAGS.use_copy or FLAGS.att_neighbor:
    assert FLAGS.att_type == "my"
  # These numbers are the target vocabulary sizes of the datasets.
  # It allows for using different vocabularies for source and targets,
  # following the implementation in Open-NMT.
  # I will later put these into command line arguments.
  if FLAGS.use_bpe:
    if FLAGS.dataset == "nyt":
      output_size = 10013
    elif FLAGS.dataset == "giga":
      output_size = 24654
    elif FLAGS.dataset == "cnnd":
      output_size = 10232
  else:
    if FLAGS.dataset == "nyt":
      output_size = 68885
    elif FLAGS.dataset == "giga":
      output_size = 107389
    elif FLAGS.dataset == "cnnd":
      output_size = 21000

  vocab = data.Vocab(FLAGS.vocab_path, FLAGS.vocab_size, FLAGS.dataset)
  hps = tf.contrib.training.HParams(
      sample_neighbor=FLAGS.sample_neighbor,
      use_cluster=FLAGS.use_cluster,
      binary_neighbor=FLAGS.binary_neighbor,
      att_neighbor=FLAGS.att_neighbor,
      encode_neighbor=FLAGS.encode_neighbor,
      sum_neighbor=FLAGS.sum_neighbor,
      dataset=FLAGS.dataset,
      rnn_cell=FLAGS.rnn_cell,
      output_size=output_size+vocab.offset,
      train_path=FLAGS.train_path,
      dev_path=FLAGS.dev_path,
      tie_embedding=FLAGS.tie_embedding,
      use_bpe=FLAGS.use_bpe,
      use_copy=FLAGS.use_copy,
      reuse_attention=FLAGS.reuse_attention,
      use_bridge=FLAGS.use_bridge,
      use_residual=FLAGS.use_residual,
      att_type=FLAGS.att_type,
      random_neighbor=FLAGS.random_neighbor,
      num_neighbors=FLAGS.num_neighbors,
      model=FLAGS.model,
      trainer=FLAGS.trainer,
      learning_rate=FLAGS.learning_rate,
      lr_schedule=FLAGS.lr_schedule,
      total_steps=FLAGS.total_steps,
      emb_dim=FLAGS.emb_dim,
      binary_dim=FLAGS.binary_dim,
      neighbor_dim=FLAGS.neighbor_dim,
      drop=FLAGS.drop,
      emb_drop=FLAGS.emb_drop,
      out_drop=FLAGS.out_drop,
      encoder_drop=FLAGS.encoder_drop,
      decoder_drop=FLAGS.decoder_drop,
      weight_decay=FLAGS.weight_decay,
      encoder_dim=FLAGS.encoder_dim,
      num_encoder_layers=FLAGS.num_encoder_layers,
      decoder_dim=FLAGS.decoder_dim,
      num_decoder_layers=FLAGS.num_decoder_layers,
      num_mlp_layers=FLAGS.num_mlp_layers,
      rank=FLAGS.rank,
      sigma_norm=FLAGS.sigma_norm,
      batch_size=FLAGS.batch_size,
      sampling_probability=FLAGS.sampling_probability,
      beam_width=FLAGS.beam_width,
      max_enc_steps=FLAGS.max_enc_steps,
      max_dec_steps=FLAGS.max_dec_steps,
      vocab_size=FLAGS.vocab_size,
      max_grad_norm=FLAGS.max_grad_norm,
      length_norm=FLAGS.length_norm,
      cp=FLAGS.coverage_penalty,
      predict_mode=FLAGS.predict_mode)

  train_input_fn = partial(
      data.input_function, is_train=True, vocab=vocab, hps=hps)
  eval_input_fn = partial(
      data.input_function, is_train=False, vocab=vocab, hps=hps)

  model_fn = partial(model_function.model_function, vocab=vocab, hps=hps)
  experiment_utils.run_experiment(
      model_fn=model_fn,
      train_input_fn=train_input_fn,
      eval_input_fn=eval_input_fn)


if __name__ == "__main__":
  tf.app.run()
