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
"""Run reinforcement learning with QA rewards.

Validation is done w.r.t. QA-based metrics only.
Training and validation are done in different processes.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import json
import os

from absl import app
from absl import flags
from language.capwap import datasets
from language.capwap.datasets import captions_dataset
from language.capwap.datasets import vqa_dataset
from language.capwap.models import reinforce_model
from language.capwap.utils import experiment_utils
from language.capwap.utils import metric_utils
from language.capwap.utils import text_utils
import tensorflow.compat.v1 as tf

DATA_DIR = os.getenv("CAPWAP_DATA", "data")

# Data for REINFORCE.
flags.DEFINE_string("target_vqa_train_pattern", None,
                    "Path to visual QA training data.")

flags.DEFINE_string("target_vqa_eval_pattern", None,
                    "Path to visual QA eval data.")

flags.DEFINE_string("target_question_eval_file", None,
                    "Questions in RC format to evaluate on.")

# MIXER data.
flags.DEFINE_string("ood_caption_train_pattern",
                    os.path.join(DATA_DIR, "COCO/processed/captions/train-*"),
                    "Path to supervised out-of-domain training data.")

flags.DEFINE_string("wsp_caption_train_pattern", None,
                    "Path to weakly-supervised *image/text* training data.")

flags.DEFINE_float("target_vqa_weight", 1.0,
                   "Weights for QA REINFORCE training data.")

flags.DEFINE_float("ood_caption_weight", 0.0,
                   "Weights for supervised out-of-domain training data.")

flags.DEFINE_float("wsp_caption_weight", 0.0,
                   "Weights for weakly-supervised *image/text* training data.")

# Model options.
flags.DEFINE_string("vocab_path", os.path.join(DATA_DIR, "uncased_vocab.txt"),
                    "Path to BERT vocab file.")

flags.DEFINE_string("rc_model", os.path.join(DATA_DIR, "rc_model"),
                    "TF Hub handle for BERT QA model.")

flags.DEFINE_string("base_model", None, "Base model to start from.")

flags.DEFINE_string("reward", "f1_score", "Reward type.")

flags.DEFINE_integer("decode_length", 30, "Max decoding length.")

flags.DEFINE_integer("question_length", 30, "Max question length.")

flags.DEFINE_integer("answer_length", 10, "Max answer length.")

flags.DEFINE_integer("num_rollouts", 16, "Beam size for REINFORCE.")

flags.DEFINE_integer("caption_length", 30, "Max caption length.")

flags.DEFINE_float("no_answer_bias", 2.0,
                   "Bias to apply to 'no answer' option.")

flags.DEFINE_integer("beam_size", 3, "Beam search width for inference.")

flags.DEFINE_float("beam_length_penalty", 0.6, "Beam search length penalty.")

FLAGS = flags.FLAGS


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  tf.logging.set_verbosity(tf.logging.INFO)
  tf.logging.info("***** Running Reinforce QA *****")

  if FLAGS.mode not in ["train", "evaluate"]:
    raise ValueError("Invalid mode.")

  if not FLAGS.target_vqa_weight > 0:
    raise ValueError("Weight on RL visual QA data should be positive.")

  if not FLAGS.base_model:
    raise ValueError("Requires a base pre-trained model.")

  # Load base model params.
  base_dir = os.path.dirname(FLAGS.base_model)
  with tf.io.gfile.GFile(os.path.join(base_dir, "params.json")) as f:
    params = json.load(f)

  # Load vocab
  tf.logging.info("Loading vocab from %s", FLAGS.vocab_path)
  vocab = text_utils.Vocab.load(FLAGS.vocab_path)

  # Add params.
  params.update(
      dict(
          answer_length=FLAGS.answer_length,
          base_model=FLAGS.base_model,
          batch_size=FLAGS.batch_size,
          beam_length_penalty=FLAGS.beam_length_penalty,
          beam_size=FLAGS.beam_size,
          caption_length=FLAGS.caption_length,
          conditional_decoding=False,
          decode_length=FLAGS.decode_length,
          eval_batch_size=FLAGS.eval_batch_size,
          learning_rate=FLAGS.learning_rate,
          mix_batches=FLAGS.mix_batches,
          model_dir=experiment_utils.get_model_dir(),
          no_answer_bias=FLAGS.no_answer_bias,
          num_input_threads=FLAGS.num_input_threads,
          num_rollouts=FLAGS.num_rollouts,
          num_train_steps=FLAGS.num_train_steps,
          num_warmup_steps=FLAGS.num_warmup_steps,
          ood_caption_train_pattern=FLAGS.ood_caption_train_pattern,
          ood_caption_weight=FLAGS.ood_caption_weight,
          predict_batch_size=FLAGS.predict_batch_size,
          prefetch_batches=FLAGS.prefetch_batches,
          question_length=FLAGS.question_length,
          rc_model=FLAGS.rc_model,
          reward=FLAGS.reward,
          target_question_eval_file=FLAGS.target_question_eval_file,
          target_vqa_eval_pattern=FLAGS.target_vqa_eval_pattern,
          target_vqa_train_pattern=FLAGS.target_vqa_train_pattern,
          use_tpu=FLAGS.use_tpu,
          vocab_path=FLAGS.vocab_path,
          vocab_size=len(vocab),
          warm_start_path=FLAGS.warm_start_path,
      ))

  # Get model_fn.
  model_fn = functools.partial(reinforce_model.model_fn, vocab=vocab)

  # Run training.
  if FLAGS.mode == "train":
    tf.logging.info("Running in mode TRAIN")
    experiment_utils.save_params(params)

    # Add the three types of data with their respective weights.
    get_dataset_fns = []
    train_weights = []

    # RL target data.
    vqa_fn = functools.partial(
        vqa_dataset.get_dataset,
        vocab=vocab,
        file_pattern=FLAGS.target_vqa_train_pattern)
    get_dataset_fns.append(vqa_fn)
    train_weights.append(FLAGS.target_vqa_weight)

    # Out-of-domain captioning data (for MIXER).
    if FLAGS.ood_caption_weight > 0:
      caption_fn = functools.partial(
          captions_dataset.get_dataset,
          vocab=vocab,
          file_pattern=FLAGS.ood_caption_train_pattern)
      get_dataset_fns.append(caption_fn)
      train_weights.append(FLAGS.ood_caption_weight)

    # Weakly-supervised captioning data generated by text planner (for MIXER).
    if FLAGS.wsp_caption_weight > 0:
      wsp_caption_fn = functools.partial(
          captions_dataset.get_dataset,
          vocab=vocab,
          file_pattern=FLAGS.wsp_caption_train_pattern)
      get_dataset_fns.append(wsp_caption_fn)
      train_weights.append(FLAGS.wsp_caption_weight)

    # Build and train the estimator.
    estimator = experiment_utils.get_estimator(model_fn, params)
    estimator.train(
        input_fn=functools.partial(
            datasets.input_fn,
            get_dataset_fns=get_dataset_fns,
            weights=train_weights,
            mix_batches=FLAGS.mix_batches),
        max_steps=FLAGS.num_train_steps)

  # Run evaluation.
  elif FLAGS.mode == "evaluate":
    tf.logging.info("Running in mode EVALUATE")
    params["context_length"] = FLAGS.decode_length
    params["no_answer_bias"] = -1e4
    vqa_fn = functools.partial(
        vqa_dataset.get_dataset,
        vocab=vocab,
        file_pattern=FLAGS.target_vqa_eval_pattern)
    estimator = experiment_utils.get_estimator(model_fn, params)
    experiment_utils.evaluate_checkpoints(
        estimator=estimator,
        vqa_input_fn=functools.partial(
            datasets.input_fn, get_dataset_fns=[vqa_fn]),
        vqa_eval_fn=functools.partial(
            metric_utils.evaluate_questions,
            vocab=vocab,
            question_file=FLAGS.target_question_eval_file,
            params=params),
        max_checkpoint_number=FLAGS.num_train_steps)


if __name__ == "__main__":
  flags.mark_flag_as_required("target_vqa_train_pattern")
  flags.mark_flag_as_required("target_vqa_eval_pattern")
  flags.mark_flag_as_required("target_question_eval_file")
  flags.mark_flag_as_required("base_model")
  tf.disable_v2_behavior()
  app.run(main)
