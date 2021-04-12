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
"""Main script for applying reversible and lossy transformation."""



from absl import app
from absl import flags

from language.compir.transform import apply_transformation_utils

FLAGS = flags.FLAGS

flags.DEFINE_enum(
    "dataset",
    "scan",
    ["scan", "cfq", "atis", "geo", "scholar"],
    "The dataset to use.",
)

flags.DEFINE_enum(
    "split",
    "iid",
    ["iid", "mcd1", "mcd2", "mcd3", "template", "length", "turnleft"],
    "The split to use (iid or compositional).",
)

# The follwing transformations prepare data for seq2seq_1.
# none (no transformation), rir (reversible), lird (lossy-direct),
# lird_rir (lossy-direct and reversible), lirind (lossy-indirect),
# lirind_rir (lossy-indirect and reversible).
# The following transformations prepare data for seq2seq_2, which recovers
# programs in the original formalism given the utterance and a lossy
# representation, where, e.g., lird2 expects predictions for the test set made
# by training seq2seq_1 on the data created by the lird transformation.
# lird2 (lossy-direct), lird_rir2 (lossy-direct and reversible),
# lirind2 (lossy indirect), lirind_rir2 (lossy-indirect and reversible).
flags.DEFINE_enum(
    "transformation", "none", [
        "none", "rir", "lird", "lird_rir", "lirind", "lirind_rir", "lird2",
        "lird_rir2", "lirind2", "lirind_rir2"
    ], "The transformation to be applied when preparing data for seq2seq_1 or"
    "seq2seq_2.")

flags.DEFINE_string("train_data_path", "", "Path to the training data.")

flags.DEFINE_string("test_data_path", "", "Path to the test data.")

flags.DEFINE_string(
    "prediction_path", None,
    "Path to test data predictions, relevant when preparing data for seq2seq_2,"
    "when recovering programs from lossy intermediate representaitons.")

flags.DEFINE_string("output_path", "", "Path where output files are written.")


def main(argv):

  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  apply_transformation_utils.transform(FLAGS.dataset, FLAGS.split,
                                       FLAGS.transformation,
                                       FLAGS.train_data_path,
                                       FLAGS.test_data_path, FLAGS.output_path,
                                       FLAGS.prediction_path)


if __name__ == "__main__":
  app.run(main)
