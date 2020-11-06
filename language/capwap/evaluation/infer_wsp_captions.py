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
"""A simple wrapper to use to generate data for weakly-supervised pretrainin.

Here we jointly predict captions (top-k rollouts) and their rewards R(y | q, a).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import json
import os
import time

from absl import app
from absl import flags
from language.capwap import datasets
from language.capwap.datasets import vqa_dataset
from language.capwap.utils import checkpoint_utils
from language.capwap.utils import experiment_utils
from language.capwap.utils import io_utils
from language.capwap.utils import reward_utils
from language.capwap.utils import tensor_utils
from language.capwap.utils import text_utils
from language.capwap.utils import transformer_utils
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub

DATA_DIR = os.getenv("QA2CAPTION_DATA", "data")

flags.DEFINE_string("checkpoint", None, "Model checkpoint.")

flags.DEFINE_string("output_file", None, "Path to write to.")

flags.DEFINE_string("input_pattern", None, "Path to eval data.")

flags.DEFINE_integer("condition_length", 40,
                     "Max conditioning sequence length.")

flags.DEFINE_integer("question_length", 30, "Max question length.")

flags.DEFINE_integer("answer_length", 10, "Max answer length.")

flags.DEFINE_float("no_answer_bias", 2, "Bias for CLS prediction.")

flags.DEFINE_string("rc_model", os.path.join(DATA_DIR, "rc_model"),
                    "TF Hub handle for BERT QA model.")

flags.DEFINE_integer("decode_length", 30, "Max decoding length.")

flags.DEFINE_integer("num_rollouts", 16, "Beam search width.")

flags.DEFINE_float("beam_length_penalty", 0.6, "Beam search length penalty.")

FLAGS = flags.FLAGS


def model_fn(features, labels, mode, params, vocab):
  """Model function."""
  del labels
  assert mode == tf.estimator.ModeKeys.PREDICT, "Mode should be PREDICT."

  # Initialize transformer model.
  model = transformer_utils.TransformerModel(
      config=transformer_utils.TransformerConfig.from_dict(params),
      is_training=(mode == tf.estimator.ModeKeys.TRAIN))

  # image_features: [batch_size, num_regions, feature_size]
  # image_positions: [batch_size, num_regions]
  # image_mask: [batch_size, num_regions]
  image_features = features["object_features"].features
  image_positions = features["object_features"].positions
  image_mask = features["object_features"].mask

  # Expand mask by 1 for IMG token.
  batch_size = tensor_utils.shape(image_mask, 0)
  input_mask = tf.pad(image_mask, [[0, 0], [1, 0]], constant_values=1)

  # [batch_size, num_regions + 1, num_layers, num_heads, head_size]
  _, input_cache = model.compute_image_transformer(
      input_ids=tf.fill([batch_size, 1], vocab.t2i(vocab.IMG)),
      input_image=image_features,
      input_image_mask=input_mask,
      input_positions=image_positions)

  # Add conditioning information to input cache.
  if params.get("conditional_decoding"):
    # Add additional (text) conditioning information to the input cache.
    # The conditioning information gets to see the image information.
    # The new input consists of both the image and the extra encoded text.
    # This is used for the LEARN function of Alg. 1 in the paper.

    # [batch_size, num_regions + condition_length + 1]
    input_mask = tf.concat([input_mask, features["condition_inputs"].mask], 1)

    # [batch_size, condition_length, num_layers, num_heads, head_size]
    _, condition_cache = model.compute_transformer(
        input_ids=features["condition_inputs"].token_ids,
        input_segment_id=features["condition_inputs"].segment_ids,
        input_positions=features["condition_inputs"].positions,
        attention_mask=tf.expand_dims(input_mask, 1),
        input_cache=input_cache,
        reuse=tf.AUTO_REUSE,
        conditional=True)

    # [batch_size, input_length, num_layers, num_heads, head_size]
    input_cache = transformer_utils.TransformerCache(
        keys=tf.concat([input_cache.keys, condition_cache.keys], 1),
        values=tf.concat([input_cache.values, condition_cache.values], 1))

  # Initialize QA model.
  rc_model = hub.Module(params["rc_model"])

  # Compute rollouts.
  rollouts = reward_utils.compute_rollouts(
      model=model,
      rc_model=rc_model,
      features=features,
      encoder_cache=input_cache,
      encoder_cache_mask=input_mask,
      vocab=vocab,
      params=params)

  # Add to predictions.
  predictions = dict(
      image_id=features["image_id"],
      question_id=features["question_id"],
      token_ids=rollouts.token_ids[:, :, 1:],
      scores=rollouts.scores,
  )

  # Add all rewards.
  for k, v in rollouts.rewards.items():
    predictions[k] = v

  # Initialize base model.
  def scaffold_fn():
    """Init op run on host."""
    checkpoint_utils.init_from_checkpoint(params["checkpoint"])
    return tf.train.Scaffold()

  return tf.estimator.tpu.TPUEstimatorSpec(
      mode=mode,
      predictions=predictions,
      scaffold_fn=scaffold_fn,
  )


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  tf.logging.set_verbosity(tf.logging.INFO)
  tf.logging.info("***** Generating captions *****")

  # Load params.
  model_dir = os.path.dirname(FLAGS.checkpoint)
  with tf.io.gfile.GFile(os.path.join(model_dir, "params.json")) as f:
    params = json.load(f)

  # Load vocab
  vocab = text_utils.Vocab.load(params["vocab_path"])

  # Update params.
  params.update(
      dict(
          answer_length=FLAGS.answer_length,
          batch_size=FLAGS.batch_size,
          beam_length_penalty=FLAGS.beam_length_penalty,
          rc_model=FLAGS.rc_model,
          checkpoint=FLAGS.checkpoint,
          condition_length=FLAGS.condition_length,
          decode_length=FLAGS.decode_length,
          eval_batch_size=FLAGS.eval_batch_size,
          expand_by_question=True,
          model_dir=experiment_utils.get_tempdir(),
          no_answer_bias=FLAGS.no_answer_bias,
          num_input_threads=FLAGS.num_input_threads,
          num_rollouts=FLAGS.num_rollouts,
          predict_batch_size=FLAGS.predict_batch_size,
          prefetch_batches=FLAGS.prefetch_batches,
          question_length=FLAGS.question_length,
          use_tpu=FLAGS.use_tpu,
      ))

  # Get estimator.
  estimator = experiment_utils.get_estimator(
      functools.partial(model_fn, vocab=vocab), params)

  # Write predictions.
  tf.logging.info("Writing predictions to disk...")
  tf.io.gfile.makedirs(os.path.dirname(FLAGS.output_file))
  vqa_fn = functools.partial(
      vqa_dataset.get_dataset, vocab=vocab, file_pattern=FLAGS.input_pattern)
  with tf.io.gfile.GFile(FLAGS.output_file, "w") as f:
    iterator = estimator.predict(
        input_fn=functools.partial(datasets.input_fn, get_dataset_fns=[vqa_fn]),
        yield_single_examples=True)
    t0 = time.time()
    total = 0
    for i, ex in enumerate(iterator, 1):
      f.write(json.dumps(ex, cls=io_utils.NumpyEncoder) + "\n")
      if i % 1000 == 0:
        tf.logging.info("Wrote %d predictions", i)
      total += 1
    tfinal = time.time() - t0
    tf.logging.info("Done. Examples per second = %2.4f", (total / tfinal))


if __name__ == "__main__":
  tf.disable_v2_behavior()
  app.run(main)
