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
"""Write tf.Example protos for model training using Beam.

This requires a dataset tsv file and a set of QCFG rules as input.
"""

from absl import app
from absl import flags
from absl import logging
import apache_beam as beam
from language.compgen.csl.common import json_utils
from language.compgen.csl.model.data import example_converter
from language.compgen.csl.qcfg import qcfg_file
from language.compgen.nqg.tasks import tsv_utils
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_string("input", "", "Input tsv file.")

flags.DEFINE_string("output", "", "Output TF example file.")

flags.DEFINE_string("config", "", "Config file.")

flags.DEFINE_string("rules", "", "Input rules file.")

flags.DEFINE_integer("offset", 0, "Start index for examples to process.")

flags.DEFINE_integer("limit", 0, "End index for examples to process if >0.")

flags.DEFINE_list(
    "pipeline_options", ["--runner=DirectRunner"],
    "A comma-separated list of command line arguments to be used as options "
    "for the Beam Pipeline.")


class ConvertExampleFn(beam.DoFn):
  """Beam wrapper around the example converter."""

  def __init__(self, rules, config):
    self._converter = example_converter.ExampleConverter(rules, config)
    self._name = "ConvertExample"

  def process(self, element):
    """Tries to convert an example, skipping it if conversion fails."""
    beam.metrics.Metrics.counter(self._name, "num_examples").inc()
    try:
      tf_example = self._converter.convert(element)
      num_nodes = self._converter.max_sizes["num_nodes"]
      beam.metrics.Metrics.distribution(self._name,
                                        "num_nodes").update(num_nodes)
      yield tf_example
    except ValueError:
      beam.metrics.Metrics.counter(self._name, "num_failed").inc()


def main(unused_argv):
  config = json_utils.json_file_to_dict(FLAGS.config)
  examples = tsv_utils.read_tsv(FLAGS.input)
  rules = qcfg_file.read_rules(FLAGS.rules)

  slice_start = FLAGS.offset
  slice_end = FLAGS.limit if FLAGS.limit else None
  examples = examples[slice(slice_start, slice_end)]

  def _convert_examples(pipeline):
    _ = (
        pipeline
        | "ImportExamples" >> beam.Create(examples)
        | "ConvertExamples" >> beam.ParDo(ConvertExampleFn(rules, config))
        | "WriteExamples" >> beam.io.tfrecordio.WriteToTFRecord(
            FLAGS.output, coder=beam.coders.ProtoCoder(tf.train.Example)))

  pipeline_options = beam.options.pipeline_options.PipelineOptions(
      FLAGS.pipeline_options)
  with beam.Pipeline(pipeline_options) as pipeline:
    _convert_examples(pipeline)

  metrics = pipeline.result.metrics().query()
  for distribution in metrics["distributions"]:
    logging.info("max %s: %s", distribution.key.metric.name,
                 distribution.committed.max)
  for counter in metrics["counters"]:
    logging.info("count %s: %s", counter.key.metric.name, counter.committed)


if __name__ == "__main__":
  app.run(main)
