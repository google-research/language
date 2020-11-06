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
"""Run inference to generate captions."""

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
from language.capwap.datasets import captions_dataset
from language.capwap.datasets import vqa_dataset
from language.capwap.models import supervised_model
from language.capwap.utils import experiment_utils
from language.capwap.utils import io_utils
from language.capwap.utils import text_utils
import tensorflow.compat.v1 as tf

DATA_DIR = os.getenv("QA2CAPTION_DATA", "data")

flags.DEFINE_string("checkpoint", None, "Model checkpoint.")

flags.DEFINE_string("input_pattern", None, "Path to eval data.")

flags.DEFINE_string("output_file", None, "Path to write to.")

flags.DEFINE_integer("decode_length", 30, "Max decoding length.")

flags.DEFINE_integer("beam_size", 3, "Beam search width.")

flags.DEFINE_float("beam_length_penalty", 0.6, "Beam search length penalty.")

flags.DEFINE_string("vocab_path", os.path.join(DATA_DIR, "uncased_vocab.txt"),
                    "Path to BERT vocab file.")

FLAGS = flags.FLAGS


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
  vocab = text_utils.Vocab.load(FLAGS.vocab_path)

  # Update params.
  params.update(
      dict(
          batch_size=FLAGS.batch_size,
          beam_length_penalty=FLAGS.beam_length_penalty,
          beam_size=FLAGS.beam_size,
          caption_length=FLAGS.decode_length,
          conditional_decoding=params.get("conditional_decoding", False),
          decode_length=FLAGS.decode_length,
          eval_batch_size=FLAGS.eval_batch_size,
          model_dir=experiment_utils.get_tempdir(),
          num_input_threads=FLAGS.num_input_threads,
          predict_batch_size=FLAGS.predict_batch_size,
          prefetch_batches=FLAGS.prefetch_batches,
          use_tpu=FLAGS.use_tpu,
          warm_start_path=FLAGS.checkpoint,
      ))

  # If the model we are evaluating uses conditional decoding, then we want to
  # set expand_by_question=True so that we get unique captions for every image.
  # Otherwise p(y|x,q) = p(y|x), and we can just reuse the same decoding for
  # one (randomly sampled) questions for all the others on the same image.
  params["expand_by_question"] = params["conditional_decoding"]

  # Get estimator.
  model_fn = functools.partial(supervised_model.model_fn, vocab=vocab)
  estimator = experiment_utils.get_estimator(model_fn, params)

  # If conditional_decoding is set, assume a VQA dataset (for text planner).
  if params["conditional_decoding"]:
    dataset_cls = vqa_dataset
  else:
    dataset_cls = captions_dataset

  # Write predictions.
  tf.logging.info("Writing predictions to disk...")
  tf.io.gfile.makedirs(os.path.dirname(FLAGS.output_file))
  with tf.io.gfile.GFile(FLAGS.output_file, "w") as f:
    iterator = estimator.predict(
        input_fn=functools.partial(
            datasets.input_fn,
            get_dataset_fns=[
                functools.partial(
                    dataset_cls.get_dataset,
                    vocab=vocab,
                    file_pattern=FLAGS.input_pattern)
            ]),
        yield_single_examples=True)
    t0 = time.time()
    total = 0
    for i, ex in enumerate(iterator, 1):
      f.write(json.dumps(ex, cls=io_utils.NumpyEncoder) + "\n")
      if i % 1000 == 0:
        tf.logging.info("Wrote %d predictions", i)
      total += 1
    tfinal = time.time() - t0
    tf.logging.info("Wrote %d predictions", total)
    tf.logging.info("Done. Examples per second = %2.4f", (total / tfinal))


if __name__ == "__main__":
  flags.mark_flag_as_required("checkpoint")
  flags.mark_flag_as_required("input_pattern")
  flags.mark_flag_as_required("output_file")
  tf.disable_v2_behavior()
  app.run(main)
