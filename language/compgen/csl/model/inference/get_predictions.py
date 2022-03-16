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
"""Takes a TSV and generates a TXT file with predicted targets."""

from absl import app
from absl import flags
from language.compgen.csl.common import json_utils
from language.compgen.csl.model.inference import inference_parser
from language.compgen.csl.model.inference import inference_utils
from language.compgen.nqg.tasks import tsv_utils
from tensorflow.io import gfile

FLAGS = flags.FLAGS

flags.DEFINE_string("input", "", "Input tsv file.")

flags.DEFINE_string("output", "", "Output tsv file.")

flags.DEFINE_string("model_dir", "", "Model directory.")

flags.DEFINE_string("checkpoint", "", "Checkpoint prefix, or None for latest.")

flags.DEFINE_string("config", "", "Config file.")

flags.DEFINE_string("rules", "", "QCFG rules txt file.")

flags.DEFINE_string("target_grammar", "", "Optional target CFG.")

flags.DEFINE_bool("verbose", False, "Whether to print debug output.")

flags.DEFINE_integer("limit", 0,
                     "Index of example to begin processing (Ignored if 0).")

flags.DEFINE_integer("offset", 0,
                     "Index of example to end processing (Ignored if 0).")


def main(unused_argv):
  config = json_utils.json_file_to_dict(FLAGS.config)
  wrapper = inference_utils.get_inference_wrapper(config, FLAGS.rules,
                                                  FLAGS.target_grammar,
                                                  FLAGS.verbose)
  _ = inference_utils.get_checkpoint(wrapper, FLAGS.model_dir, FLAGS.checkpoint)
  examples = tsv_utils.read_tsv(FLAGS.input)

  num_predictions_match = 0
  predictions = []
  for idx, example in enumerate(examples):
    if FLAGS.offset and idx < FLAGS.offset:
      continue
    if FLAGS.limit and idx >= FLAGS.limit:
      break

    if FLAGS.verbose:
      print("Processing example %s: (%s, %s)" % (idx, example[0], example[1]))

    source = example[0]
    original_target = example[1]

    predicted_target = inference_parser.get_top_output(source, wrapper)
    if FLAGS.verbose:
      print("predicted_target: %s" % predicted_target)

    if predicted_target == original_target:
      num_predictions_match += 1
    else:
      if FLAGS.verbose:
        print("predictions do not match.")

    predictions.append(predicted_target)

  print("num_predictions_match: %s" % num_predictions_match)

  with gfile.GFile(FLAGS.output, "w") as txt_file:
    for prediction in predictions:
      txt_file.write("%s\n" % prediction)


if __name__ == "__main__":
  app.run(main)
