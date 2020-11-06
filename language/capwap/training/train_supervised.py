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
"""Run supervised training with a MLE objective.

This is used to run:
1) Out-of-domain supervised pre-training of caption policy;
2) MLE training of the text planner;
3) Weakly-supervised pre-training/adaptation of caption
  (once noisy labels are generated).

Validation is done w.r.t. either reference- and QA-based metrics.
Training and validation are done in different processes.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os

from absl import app
from absl import flags
from language.capwap import datasets
from language.capwap.datasets import captions_dataset
from language.capwap.datasets import text_dataset
from language.capwap.datasets import vqa_dataset
from language.capwap.datasets import wsp_dataset
from language.capwap.models import supervised_model
from language.capwap.utils import experiment_utils
from language.capwap.utils import metric_utils
from language.capwap.utils import text_utils
import tensorflow.compat.v1 as tf

DATA_DIR = os.getenv("CAPWAP_DATA", "data")

# Used only if there's a target QA dataset to validate on.
flags.DEFINE_string("target_vqa_eval_pattern", None,
                    "Path to target visual QA eval data.")

flags.DEFINE_string("target_question_eval_file", None,
                    "Questions in RC format to evaluate on.")

# Training data options.
flags.DEFINE_string("ood_caption_train_pattern",
                    os.path.join(DATA_DIR, "COCO/processed/captions/train-*"),
                    "Path to supervised out-of-domain training data.")

flags.DEFINE_string("wsp_caption_train_pattern", None,
                    "Path to weakly-supervised *image/text* training data.")

flags.DEFINE_string("wsp_text_train_pattern", None,
                    "Path to weakly-supervised *text only* training data.")

flags.DEFINE_float("ood_caption_weight", 0.0,
                   "Weights for supervised out-of-domain training data.")

flags.DEFINE_float("wsp_caption_weight", 0.0,
                   "Weights for weakly-supervised *image/text* training data.")

flags.DEFINE_float("wsp_text_weight", 0.0,
                   "Weights for weakly-supervised *text* training data.")

# Metrics on the OOD references are used in the case of no target QA dataset.
flags.DEFINE_string("ood_caption_eval_pattern",
                    os.path.join(DATA_DIR, "COCO/processed/captions/val-*"),
                    "Path to supervised out-of-domain validation data.")

flags.DEFINE_string("coco_annotations",
                    os.path.join(DATA_DIR, "COCO/annotations"),
                    "Path to raw annotations for COCO (i.e., the OOD data).")

# Model options.
flags.DEFINE_string("vocab_path", os.path.join(DATA_DIR, "uncased_vocab.txt"),
                    "Path to BERT vocab file.")

flags.DEFINE_string("rc_model", os.path.join(DATA_DIR, "rc_model"),
                    "TF Hub handle for BERT QA model.")

flags.DEFINE_integer("caption_length", 30, "Max caption length.")

flags.DEFINE_integer("num_image_regions", 100,
                     "Number of identified objects to use.")

flags.DEFINE_integer("image_feature_size", 2048, "Object feature dimension.")

flags.DEFINE_integer("hidden_size", 512, "Transformer hidden size")

flags.DEFINE_integer("num_hidden_layers", 6, "Number of Transformer layers.")

flags.DEFINE_integer("num_attention_heads", 8,
                     "Number of Transformer attention heads.")

flags.DEFINE_integer("intermediate_size", 2048,
                     "Transformer feed-forward hidden size.")

flags.DEFINE_float("hidden_dropout_prob", 0.2,
                   "Dropout probability for all fully-connected layers.")

flags.DEFINE_float("attention_probs_dropout_prob", 0.1,
                   "Dropout probability for attention probabilities.")

flags.DEFINE_integer("max_positions", 256,
                     "Max text sequence length model might be used with.")

flags.DEFINE_integer("max_segments", 8,
                     "Max segment types model might be used with.")

flags.DEFINE_integer("max_conditions", 64,
                     "Max condition sequence length model might be used with.")

flags.DEFINE_integer("max_image_regions", 128,
                     "Max encoded image regions model might be used with.")

# Options for training the conditional text planner.
flags.DEFINE_boolean("conditional_decoding", False, "Use conditional decoding.")

flags.DEFINE_float("span_sample_p", 0.3, "Geometric distribtion parameter.")

flags.DEFINE_integer("span_length", 10, "Max random span length selected.")

flags.DEFINE_integer("condition_length", 40,
                     "Max conditioning sequence length.")

flags.DEFINE_integer("question_length", 30, "Max question length.")

flags.DEFINE_integer("answer_length", 10, "Max answer length.")

# Options for decoding.
flags.DEFINE_integer("decode_length", 30, "Max decoding length.")

flags.DEFINE_integer("beam_size", 3, "Beam search width.")

flags.DEFINE_float("beam_length_penalty", 0.6, "Beam search length penalty.")

FLAGS = flags.FLAGS


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  if FLAGS.mode not in ["train", "evaluate"]:
    raise ValueError("Invalid mode.")

  if not (FLAGS.ood_caption_weight or FLAGS.wsp_caption_weight or
          FLAGS.wsp_text_weight):
    raise ValueError("Should have non-zero weight on at least one data option.")

  tf.logging.set_verbosity(tf.logging.INFO)

  # Load vocab
  tf.logging.info("Loading vocab from %s", FLAGS.vocab_path)
  vocab = text_utils.Vocab.load(FLAGS.vocab_path)

  # Build experiment params.
  params = dict(
      answer_length=FLAGS.answer_length,
      attention_probs_dropout_prob=FLAGS.attention_probs_dropout_prob,
      batch_size=FLAGS.batch_size,
      beam_length_penalty=FLAGS.beam_length_penalty,
      beam_size=FLAGS.beam_size,
      caption_length=FLAGS.caption_length,
      coco_annotations=FLAGS.coco_annotations,
      condition_length=FLAGS.condition_length,
      conditional_decoding=FLAGS.conditional_decoding,
      decode_length=FLAGS.decode_length,
      eval_batch_size=FLAGS.eval_batch_size,
      hidden_dropout_prob=FLAGS.hidden_dropout_prob,
      hidden_size=FLAGS.hidden_size,
      image_feature_size=FLAGS.image_feature_size,
      intermediate_size=FLAGS.intermediate_size,
      learning_rate=FLAGS.learning_rate,
      max_conditions=FLAGS.max_conditions,
      max_image_regions=FLAGS.max_image_regions,
      max_positions=FLAGS.max_positions,
      max_segments=FLAGS.max_segments,
      mix_batches=FLAGS.mix_batches,
      model_dir=experiment_utils.get_model_dir(),
      num_attention_heads=FLAGS.num_attention_heads,
      num_hidden_layers=FLAGS.num_hidden_layers,
      num_image_regions=FLAGS.num_image_regions,
      num_input_threads=FLAGS.num_input_threads,
      num_train_steps=FLAGS.num_train_steps,
      num_warmup_steps=FLAGS.num_warmup_steps,
      ood_caption_eval_pattern=FLAGS.ood_caption_eval_pattern,
      ood_caption_train_pattern=FLAGS.ood_caption_train_pattern,
      ood_caption_weight=FLAGS.ood_caption_weight,
      predict_batch_size=FLAGS.predict_batch_size,
      prefetch_batches=FLAGS.prefetch_batches,
      question_length=FLAGS.question_length,
      rc_model=FLAGS.rc_model,
      span_length=FLAGS.span_length,
      span_sample_p=FLAGS.span_sample_p,
      target_question_eval_file=FLAGS.target_question_eval_file,
      target_vqa_eval_pattern=FLAGS.target_vqa_eval_pattern,
      use_tpu=FLAGS.use_tpu,
      vocab_path=FLAGS.vocab_path,
      vocab_size=len(vocab),
      warm_start_path=FLAGS.warm_start_path,
      wsp_caption_weight=FLAGS.wsp_caption_weight,
      wsp_text_train_pattern=FLAGS.wsp_text_train_pattern,
      wsp_text_weight=FLAGS.wsp_text_weight,
  )

  # Get model_fn.
  model_fn = functools.partial(supervised_model.model_fn, vocab=vocab)

  # Run training.
  if FLAGS.mode == "train":
    tf.logging.info("Running in mode TRAIN")
    experiment_utils.save_params(params)

    # Add the three types of data with their respective weights.
    get_dataset_fns = []
    train_weights = []

    # Out-of-domain captioning data.
    if FLAGS.ood_caption_weight > 0:
      caption_fn = functools.partial(
          captions_dataset.get_dataset,
          vocab=vocab,
          file_pattern=FLAGS.ood_caption_train_pattern)
      get_dataset_fns.append(caption_fn)
      train_weights.append(FLAGS.ood_caption_weight)

    # Weakly-supervised captioning data generated by (or for) text planner.
    if FLAGS.wsp_caption_weight > 0:
      # If we are doing conditional decoding (training the planner),
      # then we use the wsp_dataset. If we aren't (using the planner outputs),
      # then we use the normal captions_dataset.
      dataset = wsp_dataset if FLAGS.conditional_decoding else captions_dataset

      wsp_caption_fn = functools.partial(
          dataset.get_dataset,
          vocab=vocab,
          file_pattern=FLAGS.wsp_caption_train_pattern)
      get_dataset_fns.append(wsp_caption_fn)
      train_weights.append(FLAGS.wsp_caption_weight)

    # Weakly-supervised text data generated by (or for) text planner.
    # Note that there is no image (just conditional language modelling).
    if FLAGS.wsp_text_weight > 0:
      wsp_text_fn = functools.partial(
          text_dataset.get_dataset,
          vocab=vocab,
          file_pattern=FLAGS.wsp_text_train_pattern)
      get_dataset_fns.append(wsp_text_fn)
      train_weights.append(FLAGS.wsp_text_weight)

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
    params["expand_by_question"] = FLAGS.conditional_decoding

    # Build estimator.
    estimator = experiment_utils.get_estimator(model_fn, params)

    # If there is no target QA dataset, we use reference metrics for eval.
    # Otherwise, we use the QA metrics on the target dataset for eval.
    if not FLAGS.target_vqa_eval_pattern:
      caption_fn = functools.partial(
          captions_dataset.get_dataset,
          vocab=vocab,
          file_pattern=FLAGS.ood_caption_eval_pattern)
      experiment_utils.evaluate_checkpoints(
          estimator=estimator,
          caption_input_fn=functools.partial(
              datasets.input_fn, get_dataset_fns=[caption_fn]),
          caption_eval_fn=functools.partial(
              metric_utils.evaluate_captions,
              coco_annotations=FLAGS.coco_annotations,
              vocab=vocab),
          max_checkpoint_number=FLAGS.num_train_steps)
    else:
      vqa_fn = functools.partial(
          vqa_dataset.get_dataset,
          vocab=vocab,
          file_pattern=FLAGS.target_vqa_eval_pattern)
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
  tf.disable_v2_behavior()
  app.run(main)
