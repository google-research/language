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
"""Run a CANINE model for TyDi QA."""

from absl import logging
from language.canine import modeling as canine_modeling
from language.canine.tydiqa import char_splitter
from language.canine.tydiqa import run_tydi_lib
from language.canine.tydiqa import tydi_modeling
import tensorflow.compat.v1 as tf


flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "model_config_file", None,
    "The config json file corresponding to the pre-trained CANINE model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string("train_records_file", None,
                    "Precomputed tf records for training.")

flags.DEFINE_string(
    "record_count_file", None,
    "File containing number of precomputed training records "
    "(in terms of 'features', meaning slices of articles). "
    "This is used for computing how many steps to take in "
    "each fine tuning epoch.")

flags.DEFINE_integer(
    "candidate_beam", None,
    "How many wordpiece offset to be considered as boundary at inference time.")

flags.DEFINE_string(
    "predict_file", None,
    "TyDi json for predictions. E.g., dev-v1.1.jsonl.gz or test-v1.1.jsonl.gz. "
    "Used only for `--do_predict`.")

flags.DEFINE_string(
    "precomputed_predict_file", None,
    "TyDi tf.Example records for predictions, created separately by "
    "`prepare_tydi_data.py` Used only for `--do_predict`.")

flags.DEFINE_string(
    "output_prediction_file", None,
    "Where to print predictions in TyDi prediction format, to be passed to"
    "tydi_eval.py.")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained mBERT model).")

flags.DEFINE_integer(
    "max_seq_length", None,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "doc_stride", None,
    "When splitting up a long document into chunks, how much stride to "
    "take between chunks.")

flags.DEFINE_integer(
    "max_question_length", None,
    "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_predict", False, "Whether to run prediction.")

flags.DEFINE_integer("train_batch_size", None, "Total batch size for training.")

flags.DEFINE_integer("predict_batch_size", None,
                     "Total batch size for predictions.")

flags.DEFINE_integer(
    "predict_file_shard_size", None, "[Optional] If specified, the maximum "
    "number of examples to put into each temporary TF example file used as "
    "model input at prediction time.")

flags.DEFINE_float("learning_rate", None, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", None,
                   "Total number of training epochs to perform.")

flags.DEFINE_integer("max_to_predict", None,
                     "Maximum number of examples to predict (for debugging). "
                     "`None` or `0` will disable this and predict all.")

flags.DEFINE_float(
    "warmup_proportion", None, "Proportion of training to perform linear "
    "learning rate warmup for. E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer(
    "max_answer_length", None, "An upper bound on the number of subword pieces "
    "that a generated answer may contain. This is needed because the start and "
    "end predictions are not conditioned on one another.")

flags.DEFINE_float(
    "include_unknowns", None,
    "If positive, probability of including answers of type `UNKNOWN`.")

flags.DEFINE_bool(
    "verbose_logging", False,
    "If true, all of the warnings related to data processing will be printed. "
    "A number of warnings are expected for a normal TyDi evaluation.")

flags.DEFINE_integer(
    "max_passages", None, "Maximum number of passages to consider for a single "
    "article. If an article contains more than this, they will be discarded "
    "during training. BERT's WordPiece vocabulary must be modified to include "
    "these within the [unused*] vocab IDs.")

flags.DEFINE_integer(
    "max_position", None,
    "Maximum passage position for which to generate special tokens.")

flags.DEFINE_bool(
    "fail_on_invalid", True,
    "Stop immediately on encountering an invalid example? "
    "If false, just print a warning and skip it.")

### TPU-specific flags:

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


class CanineTyDiRunner(run_tydi_lib.TyDiRunner):
  """CANINE version of TyDiRunner."""

  def __init__(self):
    super(CanineTyDiRunner, self).__init__(
        model_config_file=None,
        output_dir=FLAGS.output_dir,
        train_records_file=FLAGS.train_records_file,
        record_count_file=FLAGS.record_count_file,
        candidate_beam=FLAGS.candidate_beam,
        predict_file=FLAGS.predict_file,
        precomputed_predict_file=FLAGS.precomputed_predict_file,
        output_prediction_file=FLAGS.output_prediction_file,
        init_checkpoint=FLAGS.init_checkpoint,
        max_seq_length=FLAGS.max_seq_length,
        doc_stride=FLAGS.doc_stride,
        max_question_length=FLAGS.max_question_length,
        do_train=FLAGS.do_train,
        do_predict=FLAGS.do_predict,
        train_batch_size=FLAGS.train_batch_size,
        predict_batch_size=FLAGS.predict_batch_size,
        predict_file_shard_size=FLAGS.predict_file_shard_size,
        learning_rate=FLAGS.learning_rate,
        num_train_epochs=FLAGS.num_train_epochs,
        warmup_proportion=FLAGS.warmup_proportion,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        iterations_per_loop=FLAGS.iterations_per_loop,
        max_answer_length=FLAGS.max_answer_length,
        include_unknowns=FLAGS.include_unknowns,
        verbose_logging=FLAGS.verbose_logging,
        max_passages=FLAGS.max_passages,
        max_position=FLAGS.max_position,
        fail_on_invalid=FLAGS.fail_on_invalid,
        use_tpu=FLAGS.use_tpu,
        tpu_name=FLAGS.tpu_name,
        tpu_zone=FLAGS.tpu_zone,
        gcp_project=FLAGS.gcp_project,
        master=FLAGS.master,
        num_tpu_cores=FLAGS.num_tpu_cores,
        max_to_predict=FLAGS.max_to_predict)
    self.model_config = canine_modeling.CanineModelConfig.from_json_file(
        FLAGS.model_config_file)

  def validate_flags_or_throw(self):
    """Validate the input FLAGS or throw an exception."""
    if FLAGS.model_config_file is None:
      raise ValueError("model_config_file is required.")
    if self.output_dir is None:
      raise ValueError("output_dir is required.")

    if not self.do_train and not self.do_predict:
      raise ValueError("At least one of `{do_train,do_predict}` must be True.")

    if self.do_train:
      if not self.train_records_file:
        raise ValueError("If `do_train` is True, then `train_records_file` "
                         "must be specified.")
      if not self.record_count_file:
        raise ValueError("If `do_train` is True, then `record_count_file` "
                         "must be specified.")
      if not self.train_batch_size:
        raise ValueError("If `do_train` is True, then `train_batch_size` "
                         "must be specified.")
      if not self.learning_rate:
        raise ValueError("If `do_train` is True, then `learning_rate` "
                         "must be specified.")
      if not self.num_train_epochs:
        raise ValueError("If `do_train` is True, then `num_train_epochs` "
                         "must be specified.")
      if not self.warmup_proportion:
        raise ValueError("If `do_train` is True, then `warmup_proportion` "
                         "must be specified.")
    else:
      if self.train_batch_size is None:
        # TPUEstimator errors if train_batch_size is not a positive integer,
        # even if we're not actually training.
        self.train_batch_size = 1

    if self.do_predict:
      if not self.predict_file:
        raise ValueError("If `do_predict` is True, "
                         "then `predict_file` must be specified.")
      if not self.max_answer_length:
        raise ValueError("If `do_predict` is True, "
                         "then `max_answer_length` must be specified.")
      if not self.candidate_beam:
        raise ValueError("If `do_predict` is True, "
                         "then `candidate_beam` must be specified.")
      if not self.predict_batch_size:
        raise ValueError("If `do_predict` is True, "
                         "then `predict_batch_size` must be specified.")
      if not self.output_prediction_file:
        raise ValueError("If `do_predict` is True, "
                         "then `output_prediction_file` must be specified.")

      if not self.precomputed_predict_file:
        if not self.max_passages:
          raise ValueError("If `precomputed_predict_file` is not specified, "
                           "then `max_passages` must be specified.")
        if not self.max_position:
          raise ValueError("If `precomputed_predict_file` is not specified, "
                           "then `max_position` must be specified.")
        if not self.doc_stride:
          raise ValueError("If `precomputed_predict_file` is not specified, "
                           "then `doc_stride` must be specified.")
        if not self.max_question_length:
          raise ValueError("If `precomputed_predict_file` is not specified, "
                           "then `max_question_length` must be specified.")
        if self.max_seq_length <= self.max_question_length + 3:
          raise ValueError(
              f"The max_seq_length ({self.max_seq_length}) must be greater "
              f"than max_question_length ({self.max_question_length}) + 3")
        if not self.include_unknowns:
          raise ValueError("If `precomputed_predict_file` is not specified, "
                           "then `include_unknowns` must be specified.")

    if self.max_seq_length > self.model_config.max_positions:
      raise ValueError(
          f"Cannot use sequence length {self.max_seq_length} "
          "because the CANINE model was only trained up to sequence length "
          f"{self.model_config.max_positions}")

  def get_tokenizer(self):
    return char_splitter.CharacterSplitter()

  def get_tydi_model_builder(self):
    return tydi_modeling.CanineModelBuilder(
        model_config=self.model_config)


def main(_):
  logging.set_verbosity(logging.INFO)
  CanineTyDiRunner().run()


if __name__ == "__main__":
  tf.disable_v2_behavior()
  # Required with both `do_train` and `do_predict`:
  flags.mark_flag_as_required("model_config_file")
  flags.mark_flag_as_required("output_dir")
  flags.mark_flag_as_required("init_checkpoint")
  flags.mark_flag_as_required("max_seq_length")
  tf.app.run()
