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
"""QCFG-based data augmentation."""

from absl import app
from absl import flags
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

flags.DEFINE_bool("allow_duplicates", True,
                  "Whether to allow duplicate examples.")

flags.DEFINE_bool("save_sampler", False, "Whether to save sampler.")


def main(unused_argv):
  sampler = sampler_utils.get_sampler_wrapper(
      augment_config=FLAGS.augment_config,
      model_dir=FLAGS.model_dir,
      model_config=FLAGS.model_config,
      rules=FLAGS.rules,
      target_grammar_file=FLAGS.target_grammar,
      checkpoint=FLAGS.checkpoint,
      verbose=FLAGS.verbose)

  examples = []
  if FLAGS.allow_duplicates:
    while len(examples) < FLAGS.num_examples:
      source, target = sampler.sample_example(len(examples))
      examples.append((source, target))
  else:
    examples_set = set()
    while len(examples_set) < FLAGS.num_examples:
      source, target = sampler.sample_example(len(examples_set))
      examples_set.add((source, target))
    examples = list(examples_set)
  tsv_utils.write_tsv(examples, FLAGS.output)
  if FLAGS.save_sampler:
    sampler.save()


if __name__ == "__main__":
  app.run(main)
