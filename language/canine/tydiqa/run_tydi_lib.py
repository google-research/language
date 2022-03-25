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
"""Functions used by run_tydi.py."""

import abc
import collections
import json
import os
from typing import Optional, Text

from absl import logging
from language.canine.tydiqa import postproc
from language.canine.tydiqa import preproc
from language.canine.tydiqa import tf_io
from language.canine.tydiqa import tydi_modeling
from language.canine.tydiqa import tydi_tokenization_interface
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator


class TyDiRunner(metaclass=abc.ABCMeta):
  """See run_tydi.py. All attributes are copied directly from the flags."""

  def __init__(
      self,
      model_config_file: Optional[Text] = None,
      output_dir: Optional[Text] = None,
      train_records_file: Optional[Text] = None,
      record_count_file: Optional[Text] = None,
      candidate_beam: int = 30,
      predict_file: Optional[Text] = None,
      precomputed_predict_file: Optional[Text] = None,
      output_prediction_file: Optional[Text] = None,
      init_checkpoint: Optional[Text] = None,
      max_seq_length: int = 512,
      doc_stride: int = 128,
      max_question_length: int = 64,
      do_train: bool = False,
      do_predict: bool = False,
      train_batch_size: int = 16,
      predict_batch_size: int = 8,
      predict_file_shard_size: Optional[int] = None,
      learning_rate: float = 5e-5,
      num_train_epochs: float = 3.0,
      warmup_proportion: float = 0.1,
      save_checkpoints_steps: int = 1000,
      iterations_per_loop: int = 1000,
      max_answer_length: int = 100,
      include_unknowns: float = -1.0,
      verbose_logging: bool = False,
      max_passages: int = 45,
      max_position: int = 45,
      fail_on_invalid: bool = True,
      use_tpu: bool = False,
      tpu_name: Optional[Text] = None,
      tpu_zone: Optional[Text] = None,
      gcp_project: Optional[Text] = None,
      master: Optional[Text] = None,
      num_tpu_cores: int = 8,
      max_to_predict: Optional[int] = None,
  ):
    self.model_config_file = model_config_file
    self.output_dir = output_dir
    self.train_records_file = train_records_file
    self.record_count_file = record_count_file
    self.candidate_beam = candidate_beam
    self.predict_file = predict_file
    self.precomputed_predict_file = precomputed_predict_file
    self.output_prediction_file = output_prediction_file
    self.init_checkpoint = init_checkpoint
    self.max_seq_length = max_seq_length
    self.doc_stride = doc_stride
    self.max_question_length = max_question_length
    self.do_train = do_train
    self.do_predict = do_predict
    self.train_batch_size = train_batch_size
    self.predict_batch_size = predict_batch_size
    self.predict_file_shard_size = predict_file_shard_size
    self.learning_rate = learning_rate
    self.num_train_epochs = num_train_epochs
    self.warmup_proportion = warmup_proportion
    self.save_checkpoints_steps = save_checkpoints_steps
    self.iterations_per_loop = iterations_per_loop
    self.max_answer_length = max_answer_length
    self.include_unknowns = include_unknowns
    self.verbose_logging = verbose_logging
    self.max_passages = max_passages
    self.max_position = max_position
    self.fail_on_invalid = fail_on_invalid
    self.use_tpu = use_tpu
    self.tpu_name = tpu_name
    self.tpu_zone = tpu_zone
    self.gcp_project = gcp_project
    self.master = master
    self.num_tpu_cores = num_tpu_cores
    self.max_to_predict = max_to_predict

  @abc.abstractmethod
  def validate_flags_or_throw(self):
    """Validate the input FLAGS or throw an exception."""
    raise NotImplementedError()

  def run(self) -> None:
    """Main entry point for this class."""
    self.validate_flags_or_throw()

    tf.gfile.MakeDirs(self.output_dir)

    tf.disable_v2_behavior()

    num_train_steps = None
    num_warmup_steps = None
    if self.do_train:
      with tf.gfile.Open(self.record_count_file, "r") as f:
        num_train_features = int(f.read().strip())
      num_train_steps = int(num_train_features / self.train_batch_size *
                            self.num_train_epochs)
      logging.info("record_count_file: %s", self.record_count_file)
      logging.info("num_records (features): %d", num_train_features)
      logging.info("num_train_epochs: %d", self.num_train_epochs)
      logging.info("train_batch_size: %d", self.train_batch_size)
      logging.info("num_train_steps: %d", num_train_steps)

      num_warmup_steps = int(num_train_steps * self.warmup_proportion)

    model_fn = self.get_model_fn(num_train_steps, num_warmup_steps)
    estimator = self.get_estimator(model_fn)

    if self.do_train:
      logging.info("Running training on precomputed features")
      logging.info("  Num split examples = %d", num_train_features)
      logging.info("  Batch size = %d", self.train_batch_size)
      logging.info("  Num steps = %d", num_train_steps)
      train_filenames = tf.gfile.Glob(self.train_records_file)
      train_input_fn = tf_io.input_fn_builder(
          input_file=train_filenames,
          seq_length=self.max_seq_length,
          is_training=True,
          drop_remainder=True)
      estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if self.do_predict:
      self.predict(estimator)

  def get_model_fn(self, num_train_steps: Optional[int],
                   num_warmup_steps: Optional[int]):
    tydi_model_builder = self.get_tydi_model_builder()
    return tydi_model_builder.model_fn_builder(
        init_checkpoint=self.init_checkpoint,
        learning_rate=self.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=self.use_tpu)

  @abc.abstractmethod
  def get_tydi_model_builder(self):
    raise NotImplementedError()

  def get_estimator(self, model_fn):
    """Creates the TPUEstimator object."""
    # If TPU is not available, this falls back to normal Estimator on CPU/GPU.
    estimator = tf_estimator.tpu.TPUEstimator(
        use_tpu=self.use_tpu,
        model_fn=model_fn,
        config=self.get_tpu_run_config(),
        train_batch_size=self.train_batch_size,
        predict_batch_size=self.predict_batch_size)
    return estimator

  def get_tpu_run_config(self):
    """Creates the RunConfig object."""
    tpu_cluster_resolver = None
    if self.use_tpu and self.tpu_name:
      tpu_cluster_resolver = tf_estimator.cluster_resolver.TPUClusterResolver(
          self.tpu_name, zone=self.tpu_zone, project=self.gcp_project)

    is_per_host = tf_estimator.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf_estimator.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=self.master,
        model_dir=self.output_dir,
        save_checkpoints_steps=self.save_checkpoints_steps,
        tpu_config=tf_estimator.tpu.TPUConfig(
            iterations_per_loop=self.iterations_per_loop,
            per_host_input_for_training=is_per_host))
    return run_config

  def predict(self, estimator):
    """Run prediction."""
    if not self.precomputed_predict_file:
      predict_examples_iter = preproc.read_tydi_examples(
          input_file=self.predict_file,
          tokenizer=self.get_tokenizer(),
          is_training=False,
          max_passages=self.max_passages,
          max_position=self.max_position,
          fail_on_invalid=self.fail_on_invalid,
          open_fn=tf_io.gopen)
      shards_iter = self.write_tf_feature_files(predict_examples_iter)
    else:
      # Uses zeros for example and feature counts since they're unknown,
      # and we only use them for logging anyway.
      shards_iter = [(1, (self.precomputed_predict_file, 0, 0))]

    # Accumulates all of the prediction results to be written to the output.
    full_tydi_pred_dict = {}
    total_num_examples = 0
    for shard_num, (shard_filename_glob, shard_num_examples,
                    shard_num_features) in shards_iter:
      total_num_examples += shard_num_examples
      logging.info(
          "Shard %d: Running prediction for %s; %d examples, %d features.",
          shard_num, shard_filename_glob, shard_num_examples,
          shard_num_features)

      # Runs the model on the shard and store the individual results.
      # If running predict on TPU, you will need to specify the number of steps.
      predict_input_fn = tf_io.input_fn_builder(
          input_file=tf.gfile.Glob(shard_filename_glob),
          seq_length=self.max_seq_length,
          is_training=False,
          drop_remainder=False)
      all_results = []
      for result in estimator.predict(
          predict_input_fn, yield_single_examples=True):
        if len(all_results) % 10000 == 0:
          logging.info("Shard %d: Predicting for feature %d/%s", shard_num,
                       len(all_results), shard_num_features)
        unique_id = int(result["unique_ids"])
        start_logits = [float(x) for x in result["start_logits"].flat]
        end_logits = [float(x) for x in result["end_logits"].flat]
        answer_type_logits = [
            float(x) for x in result["answer_type_logits"].flat
        ]
        all_results.append(
            tydi_modeling.RawResult(
                unique_id=unique_id,
                start_logits=start_logits,
                end_logits=end_logits,
                answer_type_logits=answer_type_logits))

        # Allow `None` or `0` to disable this behavior.
        if self.max_to_predict and len(all_results) == self.max_to_predict:
          logging.info(
              "WARNING: Stopping predictions early since "
              "max_to_predict == %d", self.max_to_predict)
          break

      # Reads the prediction candidates from the (entire) prediction input file.
      candidates_dict = self.read_candidates(self.predict_file)
      predict_features = []
      for shard_filename in tf.gfile.Glob(shard_filename_glob):
        for r in tf.python_io.tf_record_iterator(shard_filename):
          predict_features.append(tf.train.Example.FromString(r))
      logging.info("Shard %d: Post-processing predictions.", shard_num)
      logging.info("  Num candidate examples loaded (includes all shards): %d",
                   len(candidates_dict))
      logging.info("  Num candidate features loaded: %d", len(predict_features))
      logging.info("  Num prediction result features: %d", len(all_results))
      logging.info("  Num shard features: %d", shard_num_features)

      tydi_pred_dict = postproc.compute_pred_dict(
          candidates_dict,
          predict_features, [r._asdict() for r in all_results],
          candidate_beam=self.candidate_beam,
          max_answer_length=self.max_answer_length)

      logging.info("Shard %d: Post-processed predictions.", shard_num)
      logging.info("  Num shard examples: %d", shard_num_examples)
      logging.info("  Num post-processed results: %d", len(tydi_pred_dict))
      if shard_num_examples != len(tydi_pred_dict):
        logging.warning("  Num missing predictions: %d",
                        shard_num_examples - len(tydi_pred_dict))
      for key, value in tydi_pred_dict.items():
        if key in full_tydi_pred_dict:
          logging.warning("ERROR: '%s' already in full_tydi_pred_dict!", key)
        full_tydi_pred_dict[key] = value

    logging.info("Prediction finished for all shards.")
    logging.info("  Total input examples: %d", total_num_examples)
    logging.info("  Total output predictions: %d", len(full_tydi_pred_dict))

    with tf.gfile.Open(self.output_prediction_file, "w") as output_file:
      for prediction in full_tydi_pred_dict.values():
        output_file.write((json.dumps(prediction) + "\n").encode())

  def write_tf_feature_files(self, tydi_examples_iter):
    """Converts TyDi examples to features and writes them to files."""
    logging.info("Converting examples started.")

    total_feature_count_frequencies = collections.defaultdict(int)
    total_num_examples = 0
    total_num_features = 0
    if self.predict_file_shard_size:
      shard_iter = self.sharded_iterator(tydi_examples_iter,
                                         self.predict_file_shard_size)
    else:
      # No sharding, so treat the whole input as one "shard".
      shard_iter = [tydi_examples_iter]
    for shard_num, examples in enumerate(shard_iter, 1):
      features_writer = tf_io.FeatureWriter(
          filename=os.path.join(self.output_dir,
                                "features.tf_record-%03d" % shard_num),
          is_training=False)
      num_features_to_ids, shard_num_examples = (
          preproc.convert_examples_to_features(
              tydi_examples=examples,
              tokenizer=self.get_tokenizer(),
              is_training=False,
              max_question_length=self.max_question_length,
              max_seq_length=self.max_seq_length,
              doc_stride=self.doc_stride,
              include_unknowns=self.include_unknowns,
              output_fn=features_writer.process_feature))
      features_writer.close()

      if shard_num_examples == 0:
        continue

      shard_num_features = 0
      for num_features, ids in num_features_to_ids.items():
        shard_num_features += (num_features * len(ids))
        total_feature_count_frequencies[num_features] += len(ids)
      total_num_examples += shard_num_examples
      total_num_features += shard_num_features
      logging.info("Shard %d: Converted %d input examples into %d features.",
                   shard_num, shard_num_examples, shard_num_features)
      logging.info("  Total so far: %d input examples, %d features.",
                   total_num_examples, total_num_features)
      yield (shard_num, (features_writer.filename, shard_num_examples,
                         shard_num_features))

    logging.info("Converting examples finished.")
    logging.info("  Total examples = %d", total_num_examples)
    logging.info("  Total features = %d", total_num_features)
    logging.info("  total_feature_count_frequencies = %s",
                 sorted(total_feature_count_frequencies.items()))

  def sharded_iterator(self, iterator, shard_size):
    """Returns an iterator of iterators of at most size `shard_size`."""
    exhaused = False
    while not exhaused:

      def shard():
        for i, item in enumerate(iterator, 1):
          yield item
          if i == shard_size:
            return
        nonlocal exhaused
        exhaused = True

      yield shard()

  @abc.abstractmethod
  def get_tokenizer(self) -> tydi_tokenization_interface.TokenizerWithOffsets:
    raise NotImplementedError()

  def read_candidates(self, input_pattern):
    """Read candidates from an input pattern."""
    input_paths = tf.gfile.Glob(input_pattern)
    final_dict = {}
    for input_path in input_paths:
      file_obj = tf_io.gopen(input_path)
      final_dict.update(postproc.read_candidates_from_one_split(file_obj))
    return final_dict
