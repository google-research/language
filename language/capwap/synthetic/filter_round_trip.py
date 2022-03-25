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
"""Filter for round trip consistency in QA generations.

This uses a RC model with a no answer option.

This code follows the round-trip consistency check from the paper:
Chris Alberti, Daniel Andor, Emily Pitler, Jacob Devlin, and Michael Collins.
2019. Synthetic QA Corpora Generation with Roundtrip Consistency. In ACL.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os

from absl import app
from absl import flags
from language.capwap.utils import experiment_utils
from language.capwap.utils import reward_utils
from language.capwap.utils import text_utils
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator
import tensorflow_hub as hub

DATA_DIR = os.getenv("CAPWAP_DATA", "data")

flags.DEFINE_string("input_file", None, "Input TFRecord file.")

flags.DEFINE_string("output_file", None, "Where to write to.")

flags.DEFINE_integer("max_answer_length", 10,
                     "Maximum answer length for prediction.")

flags.DEFINE_integer("seq_length", 128, "Padded input length.")

flags.DEFINE_float("no_answer_bias", 0, "Bias for CLS prediction.")

flags.DEFINE_string("rc_model", os.path.join(DATA_DIR, "rc_model"),
                    "TF Hub handle for BERT QA model.")

flags.DEFINE_string("vocab_path", os.path.join(DATA_DIR, "uncased_vocab.txt"),
                    "Path to BERT directory.")

FLAGS = flags.FLAGS


def clean(text):
  """Postprocessing."""
  text = text.strip()
  text = " ".join(text.split())
  return text


def input_fn(params, input_file):
  """tf.data.Dataset."""

  def _parse_example(serialized_example):
    """Parse a serialized example proto."""
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            "unique_ids": tf.FixedLenFeature([], tf.int64),
            "input_ids": tf.FixedLenFeature([params["seq_length"]], tf.int64),
            "input_mask": tf.FixedLenFeature([params["seq_length"]], tf.int64),
            "segment_ids": tf.FixedLenFeature([params["seq_length"]], tf.int64),
            "start_positions": tf.FixedLenFeature([], tf.int64),
            "end_positions": tf.FixedLenFeature([], tf.int64),
            "answer_types": tf.FixedLenFeature([], tf.int64),
        })
    # Remove special [Q] token inserted before start of question.
    for k in ["input_ids", "input_mask", "segment_ids"]:
      v = features[k]
      features[k] = tf.concat([[v[0]], v[2:]], axis=0)
    return features

  dataset = tf.data.TFRecordDataset(input_file, buffer_size=16 * 1024 * 1024)
  dataset = dataset.map(
      _parse_example, num_parallel_calls=params["num_input_threads"])
  dataset = dataset.batch(params["batch_size"], drop_remainder=True)
  dataset = dataset.prefetch(params["prefetch_batches"])
  return dataset


def model_fn(features, labels, mode, params):
  """A model function satisfying the tf.estimator API."""
  del labels
  assert mode == tf_estimator.ModeKeys.PREDICT, "Mode should be PREDICT."
  rc_model = hub.Module(params["rc_model"])
  outputs = rc_model(
      inputs=dict(
          input_ids=tf.cast(features["input_ids"], tf.int32),
          input_mask=tf.cast(features["input_mask"], tf.int32),
          segment_ids=tf.cast(features["segment_ids"], tf.int32)),
      signature="extractive_qa",
      as_dict=True)
  start, end, _ = reward_utils.max_scoring_span(
      start_scores=outputs["start_logits"],
      end_scores=outputs["end_logits"],
      max_length=params["max_answer_length"],
      no_answer_bias=params["no_answer_bias"])
  is_consistent = tf.logical_and(
      tf.logical_and(tf.greater(start, 0), tf.greater(end, 0)),
      tf.logical_and(
          tf.equal(start, tf.cast(features["start_positions"] - 1, tf.int32)),
          tf.equal(end, tf.cast(features["end_positions"] - 1, tf.int32))))
  return tf_estimator.tpu.TPUEstimatorSpec(
      mode=mode,
      predictions=dict(
          unique_ids=features["unique_ids"],
          input_ids=features["input_ids"],
          start=start,
          end=end,
          is_consistent=is_consistent))


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  tf.logging.set_verbosity(tf.logging.INFO)
  tf.logging.info("***** Generating captions *****")

  # Load vocab
  vocab = text_utils.Vocab.load(FLAGS.vocab_path)

  # Update params.
  params = dict(
      seq_length=FLAGS.seq_length,
      model_dir=os.path.dirname(FLAGS.output_file),
      max_answer_length=FLAGS.max_answer_length,
      batch_size=FLAGS.batch_size,
      rc_model=FLAGS.rc_model,
      eval_batch_size=FLAGS.eval_batch_size,
      no_answer_bias=FLAGS.no_answer_bias,
      num_input_threads=FLAGS.num_input_threads,
      predict_batch_size=FLAGS.predict_batch_size,
      prefetch_batches=FLAGS.prefetch_batches,
      use_tpu=FLAGS.use_tpu,
  )

  # Get estimator.
  estimator = experiment_utils.get_estimator(model_fn, params)

  # Write predictions.
  tf.logging.info("Writing predictions to disk...")
  tf.io.gfile.makedirs(os.path.dirname(FLAGS.output_file))
  with tf.io.gfile.GFile(FLAGS.output_file, "w") as f:
    iterator = estimator.predict(
        input_fn=functools.partial(input_fn, input_file=FLAGS.input_file),
        yield_single_examples=True)
    total = 0
    for i, ex in enumerate(iterator, 1):
      if ex["is_consistent"]:
        tokens = [vocab.i2t(idx) for idx in ex["input_ids"]]
        breakpoint = tokens.index(vocab.PAD)
        question = clean(" ".join(vocab.clean(tokens[1:breakpoint])))
        context = clean(" ".join(vocab.clean(tokens[breakpoint:])))
        answer = clean(" ".join(tokens[ex["start"]:ex["end"] + 1]))
        output = [str(ex["unique_ids"]), question, answer, context]
        output = "\t".join(output)
        f.write(output + "\n")
        total += 1
        if total % 10000 == 0:
          tf.logging.info("Wrote %d predictions", total)
      if i % 10000 == 0:
        tf.logging.info("Processed %d examples", i)
    tf.logging.info("Done.")


if __name__ == "__main__":
  tf.disable_v2_behavior()
  app.run(main)
