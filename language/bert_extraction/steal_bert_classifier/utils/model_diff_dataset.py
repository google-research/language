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
"""Compare the difference between two models using a reference file."""

import os

from bert import modeling
from bert import tokenization

from bert_extraction.steal_bert_classifier.models import run_classifier

import numpy as np
from scipy import stats
import tensorflow.compat.v1 as tf
from tensorflow.contrib import cluster_resolver as contrib_cluster_resolver
from tensorflow.contrib import tpu as contrib_tpu

flags = tf.flags

FLAGS = flags.FLAGS

## Other parameters

flags.DEFINE_string("bert_config_file1", None,
                    "BERT config file for the first model.")

flags.DEFINE_string("bert_config_file2", None,
                    "BERT config file for the second model.")

flags.DEFINE_string(
    "init_checkpoint1", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_string(
    "init_checkpoint2", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool("use_random", False,
                  "Use randomly initialized weights for doing a model diff.")

flags.DEFINE_string("diff_type", "kld1",
                    "Type of difference function to be used.")

flags.DEFINE_string("diff_input_file", None,
                    "Dataset over which diff is computed")

flags.DEFINE_string("diff_output_file", None,
                    "Output file to export the predictions")


class TFCheckpointNotFoundError(OSError):
  pass


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  processors = {
      "sst-2": run_classifier.SST2Processor,
      "mnli": run_classifier.MnliProcessor
  }

  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                FLAGS.init_checkpoint1)
  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                FLAGS.init_checkpoint2)

  if not tf.train.checkpoint_exists(FLAGS.init_checkpoint1):
    raise TFCheckpointNotFoundError("checkpoint1 does not exist!")

  if not tf.train.checkpoint_exists(FLAGS.init_checkpoint2) and \
     not FLAGS.use_random:
    raise TFCheckpointNotFoundError("checkpoint2 does not exist!")

  bert_config1 = modeling.BertConfig.from_json_file(FLAGS.bert_config_file1)
  bert_config2 = modeling.BertConfig.from_json_file(FLAGS.bert_config_file2)

  if FLAGS.max_seq_length > bert_config1.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config1.max_position_embeddings))

  task_name = FLAGS.task_name.lower()

  if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name,))

  processor = processors[task_name]()

  label_list = processor.get_labels()

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  all_results = []

  predict_examples = processor.get_test_examples(FLAGS.diff_input_file)
  num_actual_predict_examples = len(predict_examples)

  # For single sentence tasks (like SST2) eg.text_b is None
  original_data = [(eg.text_a, eg.text_b) for eg in predict_examples]
  if FLAGS.use_tpu:
    # TPU requires a fixed batch size for all batches, therefore the number
    # of examples must be a multiple of the batch size, or else examples
    # will get dropped. So we pad with fake examples which are ignored
    # later on.
    while len(predict_examples) % FLAGS.predict_batch_size != 0:
      predict_examples.append(run_classifier.PaddingInputExample())

  predict_file = os.path.join(FLAGS.init_checkpoint1,
                              FLAGS.exp_name + ".predict.tf_record")

  run_classifier.file_based_convert_examples_to_features(
      predict_examples, label_list, FLAGS.max_seq_length, tokenizer,
      predict_file)

  for bert_config_type, output_dir in [(bert_config1, FLAGS.init_checkpoint1),
                                       (bert_config2, FLAGS.init_checkpoint2)]:
    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
      tpu_cluster_resolver = contrib_cluster_resolver.TPUClusterResolver(
          FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = contrib_tpu.InputPipelineConfig.PER_HOST_V2
    run_config = contrib_tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=contrib_tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    model_fn = run_classifier.model_fn_builder(
        bert_config=bert_config_type,
        num_labels=len(label_list),
        # This init checkpoint is eventually overriden by the estimator
        init_checkpoint=FLAGS.output_dir,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=None,
        num_warmup_steps=None,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = contrib_tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    tf.logging.info("***** Running prediction*****")
    tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                    len(predict_examples), num_actual_predict_examples,
                    len(predict_examples) - num_actual_predict_examples)
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

    predict_drop_remainder = True if FLAGS.use_tpu else False
    predict_input_fn = run_classifier.file_based_input_fn_builder(
        input_file=predict_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=predict_drop_remainder)

    result = list(estimator.predict(input_fn=predict_input_fn))
    all_results.append(result)

  all_results[0] = all_results[0][:num_actual_predict_examples]
  all_results[1] = all_results[1][:num_actual_predict_examples]

  assert len(all_results[0]) == len(all_results[1])

  # Assuming model1's predictions are gold labels, calculate model2's accuracy
  score = 0
  for prob1, prob2 in zip(all_results[0], all_results[1]):
    if np.argmax(prob1["probabilities"]) == np.argmax(prob2["probabilities"]):
      score += 1

  tf.logging.info("Agreement score = %.6f",
                  float(score) / num_actual_predict_examples)

  # Calculate the average value of |v1 - v2|, the distance on the simplex
  # Unlike KL divergence, this is a bounded metric
  # However, these results are not comparable across tasks
  # with different number classes
  distances = []
  for prob1, prob2 in zip(all_results[0], all_results[1]):
    distances.append(
        np.linalg.norm(prob1["probabilities"] - prob2["probabilities"]))

  tf.logging.info("Average length |p1 - p2| = %.8f", np.mean(distances))
  tf.logging.info("Max length |p1 - p2| = %.8f", np.max(distances))
  tf.logging.info("Min length |p1 - p2| = %.8f", np.min(distances))
  tf.logging.info("Std length |p1 - p2| = %.8f", np.std(distances))

  if FLAGS.diff_type == "kld1":
    all_kld = []

    for prob1, prob2 in zip(all_results[0], all_results[1]):
      all_kld.append(
          stats.entropy(prob1["probabilities"], prob2["probabilities"]))

    tf.logging.info("Average kl-divergence (p1, p2) = %.8f", np.mean(all_kld))
    tf.logging.info("Max kl-divergence (p1, p2) = %.8f", np.max(all_kld))
    tf.logging.info("Min kl-divergence (p1, p2) = %.8f", np.min(all_kld))
    tf.logging.info("Std kl-divergence (p1, p2) = %.8f", np.std(all_kld))

  elif FLAGS.diff_type == "kld2":
    all_kld = []

    for prob1, prob2 in zip(all_results[0], all_results[1]):
      all_kld.append(
          stats.entropy(prob2["probabilities"], prob1["probabilities"]))

    tf.logging.info("Average kl-divergence (p2, p1) = %.8f", np.mean(all_kld))
    tf.logging.info("Max kl-divergence (p2, p1) = %.8f", np.max(all_kld))
    tf.logging.info("Min kl-divergence (p2, p1) = %.8f", np.min(all_kld))
    tf.logging.info("Std kl-divergence (p2, p1) = %.8f", np.std(all_kld))

  if FLAGS.diff_output_file:
    output = ""

    # Removing padded examples
    all_results[0] = all_results[0][:len(original_data)]
    all_results[1] = all_results[1][:len(original_data)]

    with tf.gfile.GFile(FLAGS.diff_output_file, "w") as f:
      for i, (eg, prob1, prob2) in enumerate(
          zip(original_data, all_results[0], all_results[1])):

        if i % 1000 == 0:
          tf.logging.info("Writing instance %d", i + 1)

        p1_items = [p1.item() for p1 in prob1["probabilities"]]
        p2_items = [p2.item() for p2 in prob2["probabilities"]]

        prob1_str = "%.6f\t%.6f\t%.6f" % (p1_items[0], p1_items[1], p1_items[2])
        prob2_str = "%.6f\t%.6f\t%.6f" % (p2_items[0], p2_items[1], p2_items[2])

        output = "%s\t%s\t%s\t%s\n" % (eg[0], eg[1], prob1_str, prob2_str)
        f.write(output)

  return


if __name__ == "__main__":
  flags.mark_flag_as_required("task_name")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file1")
  flags.mark_flag_as_required("bert_config_file2")
  flags.mark_flag_as_required("init_checkpoint1")
  flags.mark_flag_as_required("init_checkpoint2")
  flags.mark_flag_as_required("diff_input_file")
  tf.app.run()
