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
"""QCFG-based data augmentation using Beam."""

from absl import app
from absl import flags
from absl import logging
import apache_beam as beam
from language.compgen.csl.augment import sampler_utils
from language.compgen.nqg.tasks import tsv_utils


FLAGS = flags.FLAGS

flags.DEFINE_string("augment_config", "", "Augment config file.")

flags.DEFINE_string("output", "", "Output TSV file.")

flags.DEFINE_integer("num_examples", 1000,
                     "The number of examples to generate.")

flags.DEFINE_string("rules", "", "The QCFG rules.")

flags.DEFINE_string("target_grammar", "", "Optional target CFG.")

flags.DEFINE_string("model_dir", "", "Optional model directory.")

flags.DEFINE_string("checkpoint", "", "Checkpoint prefix, or None for latest.")

flags.DEFINE_string("model_config", "", "Model config file.")

flags.DEFINE_bool("verbose", False, "Whether to print debug output.")

flags.DEFINE_bool(
    "allow_duplicates", True,
    "Whether to allow duplicate examples. If not allow_duplicates, "
    "the number of generated examples might be smaller than num_examples.")

flags.DEFINE_list(
    "pipeline_options", ["--runner=DirectRunner"],
    "A comma-separated list of command line arguments to be used as options "
    "for the Beam Pipeline.")


def sample_example(i, sampler):
  beam.metrics.Metrics.counter("SampleExamples", "num_examples").inc()
  return sampler.sample_example(i)


def main(unused_argv):
  sampler = sampler_utils.get_sampler_wrapper(
      augment_config=FLAGS.augment_config,
      model_dir=FLAGS.model_dir,
      model_config=FLAGS.model_config,
      rules=FLAGS.rules,
      target_grammar_file=FLAGS.target_grammar,
      checkpoint=FLAGS.checkpoint,
      verbose=FLAGS.verbose)

  def _sample_examples(pipeline):
    seeds = range(FLAGS.num_examples)
    examples = (
        pipeline
        | "Create" >> beam.Create(seeds)
        | "SampleExamples" >> beam.Map(sample_example, sampler=sampler)
        | "Format" >> beam.Map(lambda ex: "%s\t%s" % (ex[0], ex[1])))
    if not FLAGS.allow_duplicates:
      examples = examples | "RemoveDuplicates" >> beam.Distinct()
    _ = examples | "WriteExamples" >> beam.io.WriteToText(FLAGS.output)

  pipeline_options = beam.options.pipeline_options.PipelineOptions(
      FLAGS.pipeline_options)
  with beam.Pipeline(pipeline_options) as pipeline:
    _sample_examples(pipeline)

  metrics = pipeline.result.metrics().query()
  for counter in metrics["counters"]:
    logging.info("%s: %s", counter.key.metric.name, counter.committed)
  tsv_utils.merge_shared_tsvs(FLAGS.output)


if __name__ == "__main__":
  app.run(main)
