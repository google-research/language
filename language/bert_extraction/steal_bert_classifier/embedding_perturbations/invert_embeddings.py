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
"""Perform a continuous-embedding-space model inversion using a fixed BERT checkpoint and objective.

Since text is discrete, model inversion is performed on the embedding space and
nearest neighbours are taken after inversion. For more details on model
inversion attacks, see https://www.cs.cmu.edu/~mfredrik/papers/fjr2015ccs.pdf.
"""

import functools

from bert import modeling
from bert import tokenization

from bert_extraction.steal_bert_classifier.embedding_perturbations import embedding_util as em_util
from bert_extraction.steal_bert_classifier.models import run_classifier as rc

import tensorflow.compat.v1 as tf

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_integer("total_steps", 100, "Total number of optimization steps.")
flags.DEFINE_string("input_template", "[EMPTY]<freq>*",
                    "CSV format to carry out inversion on templates of text.")
flags.DEFINE_string(
    "prob_vector", None,
    "probability vector to allow custom distillation-like cross-entropy obj.")
flags.DEFINE_string("obj_type", "max_self_entropy",
                    "The kind of loss function to optimize embeddings.")


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  processors = {
      "sst-2": rc.SST2Processor,
      "mnli": rc.MnliProcessor,
  }

  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                FLAGS.init_checkpoint)
  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  task_name = FLAGS.task_name.lower()
  processor = processors[task_name]()
  num_labels = len(processor.get_labels())

  # This flexible input variable will be optimized to carry out model inversion
  flex_input = tf.get_variable(
      name="flex_input",
      shape=[
          FLAGS.train_batch_size, FLAGS.max_seq_length, bert_config.hidden_size
      ])

  # Use the input template to mix original embeddings with flex input embeddings
  # Different segments in template are separated by " <piece> "
  # Each segment is associated with a word piece (or [EMPTY] to get flex inputs)
  # and a frequency. (which is separated by "<freq>"). * can be used to choose a
  # frequency till the end of the string
  #
  # Here is an example 2-sequence template for tasks like MNLI to optimize
  # 20 vectors, (10 for each sequence)
  # [CLS]<freq>1 <piece> [EMPTY]<freq>10 <piece> [SEP]<freq>1 <piece> \
  # [EMPTY]<freq>10 <piece> [SEP]<freq>1 <piece> [PAD]<freq>*
  (input_tensor, embed_var, flex_input_mask, bert_input_mask,
   token_type_ids) = em_util.template_to_input_tensor(
       template=FLAGS.input_template,
       flex_input=flex_input,
       config=bert_config,
       tokenizer=tokenizer,
       max_seq_length=FLAGS.max_seq_length)

  # Get the nearest neighbours of the input tensor
  # Useful for converting input tensor back to a string representation
  nearest_neighbours, cos_sim = em_util.get_nearest_neighbour(
      source=input_tensor, reference=embed_var)

  # Convert the nearest neighbours back into embeddings. This is done since text
  # is discrete, and we want to create actual textual outputs.
  nn_embeddings, _ = em_util.run_bert_embeddings(
      input_ids=nearest_neighbours, config=bert_config)

  mean_masked_cos_sim = tf.reduce_mean(
      tf.boolean_mask(cos_sim, flex_input_mask))

  # With this probability vector, a custom cross-entropy goal can be specified.
  # When this is used, the inputs are optimized to encourage the classifier to
  # produce a softmax output similar to prob_vector.
  prob_vector = tf.constant([[float(x) for x in FLAGS.prob_vector.split(",")]])

  model_fn_partial = functools.partial(
      em_util.model_fn,
      bert_input_mask=bert_input_mask,
      token_type_ids=token_type_ids,
      bert_config=bert_config,
      num_labels=num_labels,
      obj_type=FLAGS.obj_type,
      prob_vector=prob_vector)

  parent_scope = tf.get_variable_scope()
  with tf.variable_scope(parent_scope):
    flex_input_obj, _, _ = model_fn_partial(input_tensor=input_tensor)

  if FLAGS.obj_type[:3] == "max":
    flex_input_loss = -1 * flex_input_obj
  elif FLAGS.obj_type[:3] == "min":
    flex_input_loss = flex_input_obj

  with tf.variable_scope(parent_scope, reuse=True):
    nn_input_obj, _, _ = model_fn_partial(input_tensor=nn_embeddings)

  opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
  invert_op = opt.minimize(flex_input_loss, var_list=[flex_input])

  tvars = tf.trainable_variables()

  assignment_map, _ = modeling.get_assignment_map_from_checkpoint(
      tvars, FLAGS.init_checkpoint)

  tf.logging.info("Variables mapped = %d / %d", len(assignment_map), len(tvars))

  tf.train.init_from_checkpoint(FLAGS.init_checkpoint, assignment_map)

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  # optimize a part of the flex_input (depending on the template)
  for i in range(FLAGS.total_steps):
    fio, nio, _, mcs = sess.run(
        [flex_input_obj, nn_input_obj, invert_op, mean_masked_cos_sim])
    tf.logging.info(
        "Step %d / %d. flex-input obj = %.4f, nn obj = %.4f, cos sim = %.4f", i,
        FLAGS.total_steps, fio, nio, mcs)

  # Find nearest neighbours for the final optimized vectors
  batched_nn, batched_nn_sim = sess.run([nearest_neighbours, cos_sim])

  for nn, _ in zip(batched_nn, batched_nn_sim):
    tf.logging.info("Sentence = %s", em_util.detokenize(nn, tokenizer))

  return


if __name__ == "__main__":
  flags.mark_flag_as_required("learning_rate")
  flags.mark_flag_as_required("task_name")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  tf.app.run()
