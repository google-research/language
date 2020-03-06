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
"""Calculate the interpolations between two sentences using mixup on BERT embeddings using mixup (https://arxiv.org/abs/1710.09412)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from bert import modeling
from bert import tokenization

from bert_extraction.steal_bert_classifier.embedding_perturbations import embedding_util as em_util
from bert_extraction.steal_bert_classifier.models import run_classifier as rc

import tensorflow.compat.v1 as tf

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("interpolate_scheme", "beta",
                    "Interpolation scheme between input points.")
flags.DEFINE_float("alpha", 0.4,
                   "The alpha value for sampling Beta distribution.")


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  processors = {
      "sst-2": rc.SST2Processor,
      "mnli": rc.MnliProcessor,
  }

  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                FLAGS.init_checkpoint)
  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  task_name = FLAGS.task_name.lower()
  processor = processors[task_name]()
  label_list = processor.get_labels()
  predict_examples = processor.get_test_examples(FLAGS.predict_input_file)

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  predict_file = os.path.join(FLAGS.output_dir,
                              "mixup_%s.tf_record" % FLAGS.exp_name)

  rc.file_based_convert_examples_to_features(predict_examples, label_list,
                                             FLAGS.max_seq_length, tokenizer,
                                             predict_file)

  predict_input_fn = rc.file_based_input_fn_builder(
      input_file=predict_file,
      seq_length=FLAGS.max_seq_length,
      is_training=True,
      drop_remainder=False)

  predict_dataset = predict_input_fn({"batch_size": FLAGS.predict_batch_size})

  predict_iterator1 = predict_dataset.make_one_shot_iterator()
  predict_iterator2 = predict_dataset.make_one_shot_iterator()

  predict_dict1 = predict_iterator1.get_next()
  predict_dict2 = predict_iterator2.get_next()

  # Extract only the BERT non-contextual word embeddings, see their outputs
  embed1_out, embed_var = em_util.run_bert_embeddings(
      predict_dict1["input_ids"], bert_config)
  embed2_out, _ = em_util.run_bert_embeddings(predict_dict2["input_ids"],
                                              bert_config)

  if FLAGS.interpolate_scheme == "beta":
    # Interpolate two embeddings using samples from a beta(alpha, alpha) distro
    beta_distro = tf.distributions.Beta(FLAGS.alpha, FLAGS.alpha)
    interpolate = beta_distro.sample()
  elif FLAGS.interpolate_scheme == "fixed":
    # Interpolate two embeddings using a fixed interpolation constant
    interpolate = tf.constant(FLAGS.alpha)

  new_embed = interpolate * embed1_out + (1 - interpolate) * embed2_out

  # Get nearest neighbour in embedding space for interpolated embeddings
  nearest_neighbour, _ = em_util.get_nearest_neighbour(
      source=new_embed, reference=embed_var)
  nearest_neighbour = tf.cast(nearest_neighbour, tf.int32)

  # Check whether nearest neighbour is a new word
  new_vectors = tf.logical_and(
      tf.not_equal(nearest_neighbour, predict_dict1["input_ids"]),
      tf.not_equal(nearest_neighbour, predict_dict2["input_ids"]))

  # Combine the two input masks
  token_mask = tf.logical_or(
      tf.cast(predict_dict1["input_mask"], tf.bool),
      tf.cast(predict_dict2["input_mask"], tf.bool))

  # Mask out new vectors with original tokens mask
  new_vectors_masked = tf.logical_and(new_vectors, token_mask)

  tvars = tf.trainable_variables()

  assignment_map, _ = modeling.get_assignment_map_from_checkpoint(
      tvars, FLAGS.init_checkpoint)

  tf.train.init_from_checkpoint(FLAGS.init_checkpoint, assignment_map)

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  total_score = 0
  total_tokens = 0

  total_steps = len(predict_examples) // FLAGS.predict_batch_size + 1

  # Count the total words where new embeddings are produced via interpolation
  all_predict_input1 = []
  all_predict_input2 = []
  all_nearest_neighbours = []

  for i in range(total_steps):
    tf.logging.info("%d/%d, total_score = %d / %d", i, total_steps, total_score,
                    total_tokens)
    pd1, pd2, nn, tm, nvm = sess.run([
        predict_dict1, predict_dict2, nearest_neighbour, token_mask,
        new_vectors_masked
    ])

    # populate global lists of inputs and mix-ups
    all_nearest_neighbours.extend(nn.tolist())
    all_predict_input1.extend(pd1["input_ids"].tolist())
    all_predict_input2.extend(pd2["input_ids"].tolist())
    total_score += nvm.sum()
    total_tokens += tm.sum()

  tf.logging.info("Total score = %d", total_score)

  with tf.gfile.GFile(FLAGS.predict_output_file, "w") as f:
    for pd1, pd2, nn in zip(all_predict_input1, all_predict_input2,
                            all_nearest_neighbours):
      pd1_sent = " ".join(tokenizer.convert_ids_to_tokens(pd1))
      pd2_sent = " ".join(tokenizer.convert_ids_to_tokens(pd2))
      nn_sent = " ".join(tokenizer.convert_ids_to_tokens(nn))
      full_line = pd1_sent + "\t" + pd2_sent + "\t" + nn_sent + "\n"
      f.write(full_line)


if __name__ == "__main__":
  flags.mark_flag_as_required("output_dir")
  flags.mark_flag_as_required("data_dir")
  flags.mark_flag_as_required("task_name")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  tf.app.run()
