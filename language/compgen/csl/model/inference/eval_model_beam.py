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
"""Binary to evaluate model using Beam.

This binary can also be configured to run alongside a training job
and poll for new model checkpoints, writing eval metrics (e.g. for TensorBoard).
"""

import os
import time

from absl import app
from absl import flags
from absl import logging

import apache_beam as beam
from language.compgen.csl.common import beam_utils
from language.compgen.csl.common import json_utils
from language.compgen.csl.common import txt_utils
from language.compgen.csl.common import writer_utils
from language.compgen.csl.model.inference import eval_utils
from language.compgen.csl.model.inference import inference_utils
from language.compgen.nqg.tasks import tsv_utils
from tensorflow.io import gfile

FLAGS = flags.FLAGS

flags.DEFINE_string("input", "", "Input tsv file.")

flags.DEFINE_string("output", "predictions.txt", "Output tsv file.")

flags.DEFINE_integer("limit", 0,
                     "Index of example to begin processing (Ignored if 0).")

flags.DEFINE_integer("offset", 0,
                     "Index of example to end processing (Ignored if 0).")

flags.DEFINE_bool("verbose", False, "Whether to logging.info debug output.")

flags.DEFINE_string("model_dir", "", "Model directory.")

flags.DEFINE_bool("poll", False, "Whether to poll.")

flags.DEFINE_bool("write", False, "Whether to write metrics to model_dir.")

flags.DEFINE_string("subdir", "eval_test",
                    "Sub-directory of model_dir for writing metrics.")

flags.DEFINE_string("checkpoint", "", "Checkpoint prefix, or None for latest.")

flags.DEFINE_string("config", "", "Config file.")

flags.DEFINE_string("rules", "", "QCFG rules txt file.")

flags.DEFINE_string("target_grammar", "", "Optional target CFG.")

flags.DEFINE_string("fallback_predictions", "",
                    "Optional fallback predictions txt file.")

flags.DEFINE_list(
    "pipeline_options", ["--runner=DirectRunner"],
    "A comma-separated list of command line arguments to be used as options "
    "for the Beam Pipeline.")


def merge_predictions(examples, filename):
  """Merge multiple predcition files into one."""
  source_to_prediction = {}
  output_files = gfile.glob("%s-*-of-*" % filename)
  for output_file in output_files:
    predictions = tsv_utils.read_tsv(output_file)
    for prediction in predictions:
      source, predicted_target = prediction
      source_to_prediction[source] = predicted_target
  new_predictions = []
  for example in examples:
    new_predictions.append((source_to_prediction[example[0]]))
  txt_utils.write_txt(new_predictions, filename)


class EvalModelFn(beam.DoFn):
  """Beam wrapper for the inference wrapper."""

  def __init__(self, config, checkpoint):
    self.config = config
    self.checkpoint = checkpoint
    self.wrapper = None

  def get_wrapper(self):
    self.wrapper = inference_utils.get_inference_wrapper(
        self.config, FLAGS.rules, FLAGS.target_grammar, FLAGS.verbose)
    inference_utils.get_checkpoint(self.wrapper, FLAGS.model_dir,
                                   self.checkpoint)

  def process(self, elements):
    if self.wrapper is None:
      self.get_wrapper()

    example, fallback_prediction = elements
    metrics_dict, predictions = eval_utils.eval_model(
        self.wrapper, [example], [fallback_prediction], verbose=FLAGS.verbose)
    beam_utils.dict_to_beam_counts(metrics_dict, "EvalModel")

    source = example[0]
    prediction = predictions[0]
    yield "%s\t%s" % (source, prediction)


def run_inference(writer,
                  config,
                  examples,
                  fallback_predictions,
                  checkpoint,
                  step=None):
  """Run inference."""

  elements = [(ex, pred) for ex, pred in zip(examples, fallback_predictions)]
  filename = os.path.join(FLAGS.model_dir, FLAGS.subdir,
                          "%s-%d" % (FLAGS.output, step))

  def _eval_model(pipeline):
    predictions = (
        pipeline
        | "ImportExamples" >> beam.Create(elements)
        | "EvalExamples" >> beam.ParDo(EvalModelFn(config, checkpoint)))
    if FLAGS.write:
      _ = (predictions | "WriteExamples" >> beam.io.WriteToText(filename))

  pipeline_options = beam.options.pipeline_options.PipelineOptions(
      FLAGS.pipeline_options)
  with beam.Pipeline(pipeline_options) as pipeline:
    _eval_model(pipeline)

  counters = pipeline.result.metrics().query()["counters"]
  counts = {}
  for counter in counters:
    counts[counter.key.metric.name] = counter.committed
  metrics_dict = eval_utils.compute_metrics(counts)
  for metric_name, metric_value in metrics_dict.items():
    logging.info("%s at %s: %s", metric_name, step, metric_value)
  if FLAGS.write:
    writer_utils.write_metrics(writer, metrics_dict, step)
    merge_predictions(examples, filename)


def main(unused_argv):
  config = json_utils.json_file_to_dict(FLAGS.config)
  wrapper = inference_utils.get_inference_wrapper(config, FLAGS.rules,
                                                  FLAGS.target_grammar,
                                                  FLAGS.verbose)
  writer = None
  if FLAGS.write:
    write_dir = os.path.join(FLAGS.model_dir, FLAGS.subdir)
    writer = writer_utils.get_summary_writer(write_dir)

  examples = tsv_utils.read_tsv(FLAGS.input)
  fallback_predictions = [None] * len(examples)
  if FLAGS.fallback_predictions:
    fallback_predictions = txt_utils.read_txt(FLAGS.fallback_predictions)
  if len(examples) != len(fallback_predictions):
    raise ValueError("len(examples) != len(fallback_predictions).")

  slice_start = FLAGS.offset
  slice_end = FLAGS.limit if FLAGS.limit else None
  examples = examples[slice(slice_start, slice_end)]
  fallback_predictions = fallback_predictions[slice(slice_start, slice_end)]

  if FLAGS.poll:
    last_checkpoint = None
    while True:
      checkpoint, step = inference_utils.get_checkpoint(wrapper,
                                                        FLAGS.model_dir,
                                                        FLAGS.checkpoint)
      if checkpoint == last_checkpoint:
        logging.info("Waiting for new checkpoint...\nLast checkpoint: %s",
                     last_checkpoint)
      else:
        run_inference(writer, config, examples, fallback_predictions,
                      checkpoint, step)
        last_checkpoint = checkpoint
      if step and step >= config["training_steps"]:
        # Stop eval job after completing eval for last training step.
        break
      time.sleep(10)
  else:
    checkpoint, step = inference_utils.get_checkpoint(wrapper, FLAGS.model_dir,
                                                      FLAGS.checkpoint)
    run_inference(writer, config, examples, fallback_predictions, checkpoint,
                  step)


if __name__ == "__main__":
  app.run(main)
