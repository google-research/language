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
"""Main file for generating data and gathering statistics.

The arguemnts follow the original gSCAN repository. The code supports data
generation for all original splits (uniform, compositional, target_lengths) and
the new spatial relation splits. It also allows gather dataset statistics and
visualize examples for the existing dataset file. See
https://github.com/LauraRuis/groundedSCAN/blob/master/GroundedScan/__main__.py
for more details.
"""

from absl import app
from absl import flags
from absl import logging

from language.gscan.data import dataset
import tensorflow as tf

FLAGS = flags.FLAGS

# General arguments.
flags.DEFINE_enum(
    "mode", None, ["generate", "gather_stats", "visualize"],
    "Whether to generate data, gather dataset statistics, or visualize data examples."
)
flags.DEFINE_string("output_directory", None, "The output directory.")
flags.DEFINE_string("load_dataset_from", None, "Path to datsaet file to load.")
flags.DEFINE_string("save_dataset_as", "dataset.txt",
                    "Filename to save dataset in.")
flags.DEFINE_bool(
    "count_equivalent_examples", False,
    "Whether or not to count the number of equivalent examples in the "
    "training and test set at the end of generation.")
flags.DEFINE_bool("make_dev_set", True, "Whether to make a dev set.")

# Dataset arguments.
flags.DEFINE_enum(
    "split", "generalization",
    ["uniform", "generalization", "target_lengths", "spatial_relation"],
    "The split to generate.")
flags.DEFINE_integer("max_examples", 0, "Max. number of examples to generate.")
flags.DEFINE_integer(
    "max_examples_per_target", 0,
    "Max. number of examples to generate per unique target object.")
flags.DEFINE_integer(
    "k_shot_generalization", 0,
    "Number of examples of a particular split to add to the training set.")
flags.DEFINE_integer(
    "num_resampling", 1,
    "Number of time to resample a semantically equivalent situation (which will"
    "likely result in different situations in terms of object locations).")
flags.DEFINE_integer(
    "visualize_per_template", 0,
    "How many visualization to generate per command template.")
flags.DEFINE_integer("visualize_per_split", 0,
                     "How many visualization to generate per test split.")
flags.DEFINE_float(
    "percentage_train", 0.9,
    "Percentage of examples to put in the training set (rest is test set).")
flags.DEFINE_float(
    "percentage_dev", 0.01,
    "Percentage of examples to put in the training set (rest is test set).")
flags.DEFINE_integer(
    "cut_off_target_length", 0,
    "Examples of what target length to put in the test set for split=target_lengths"
)

# World arguments.
flags.DEFINE_integer("grid_size", 6,
                     "Number of rows (and columns) in the grid world.")
flags.DEFINE_integer("min_other_objects", 0,
                     "Minimum amount of objects to put in the grid world.")
flags.DEFINE_integer("max_objects", 12,
                     "Maximum amount of objects to put in the grid world.")
flags.DEFINE_integer("min_object_size", 1, "Smallest object size.")
flags.DEFINE_integer("max_object_size", 4, "Biggest object size.")
flags.DEFINE_integer("max_distract_objects", 2,
                     "maximum number of distract objects.")
flags.DEFINE_float(
    "other_objects_sample_percentage", 0.5,
    "Percentage of possible objects distinct from the target to place in the world."
)

# Grammar and Vocabulary arguments
flags.DEFINE_enum("type_grammar", "adverb", [
    "simple_intrans", "simple_trans", "normal", "adverb",
    "relation_simple_intrans", "relation_simple_trans", "relation_normal",
    "relation_adverb"
], "The grammar type to use.")
flags.DEFINE_enum(
    "exclude_type_grammar", "", [
        "", "simple_intrans", "simple_trans", "normal", "adverb",
        "relation_simple_intrans", "relation_simple_trans", "relation_normal",
        "relation_adverb"
    ],
    "The optional grammar type to exclude. It should be a subset of the type_grammar."
)
flags.DEFINE_list("intransitive_verbs", "walk",
                  "Comma-separated list of intransitive verbs.")
flags.DEFINE_list("transitive_verbs", "pull,push",
                  "Comma-separated list of transitive verbs.")
flags.DEFINE_list("adverbs",
                  "cautiously,while spinning,hesitantly,while zigzagging",
                  "Comma-separated list of adverbs.")
flags.DEFINE_list("nouns", "square,cylinder,circle",
                  "Comma-separated list of nouns.")
flags.DEFINE_list("color_adjectives", "red,green,yellow,blue",
                  "Comma-separated list of colors.")
flags.DEFINE_list("size_adjectives", "big,small",
                  "Comma-separated list of sizes.")
flags.DEFINE_list(
    "location_preps",
    "east of,north of,west of,south of,north east of,north west of,"
    "south west of,south east of,next to",
    "Comma-separated list of location prepositions.")
flags.DEFINE_enum(
    "sample_vocabulary", "default", ["default"],
    "Whether to specify own vocabulary or to sample a nonsensical one.")


def gather_dataset_statistics(grounded_scan):
  """Gather dataset statistics and save them for each split."""
  logging.info("Gathering dataset statistics...")

  splits = ["train", "test"]
  if FLAGS.split == "generalization":
    splits.extend([
        "visual", "situational_1", "situational_2", "contextual", "adverb_1",
        "adverb_2", "visual_easier"
    ])
  elif FLAGS.split == "spatial_relation":
    splits.extend([
        "visual", "relation", "referent", "relative_position_1",
        "relative_position_2"
    ])
  elif FLAGS.split == "target_lengths":
    splits.append("target_lengths")
  if FLAGS.make_dev_set:
    splits += ["dev"]
  for split in splits:
    grounded_scan.save_dataset_statistics(split=split)
    logging.info("Saved dataset staistics for split %s", split)
  if FLAGS.visualize_per_template > 0 or FLAGS.visualize_per_split > 0:
    grounded_scan.visualize_data_examples(
        visualize_per_split=FLAGS.visualize_per_split)
  if FLAGS.count_equivalent_examples:
    if FLAGS.split == "uniform":
      splits_to_count = ["test"]
    elif FLAGS.split == "generalization":
      splits_to_count = [
          "visual", "situational_1", "situational_2", "contextual"
      ]
    elif FLAGS.split == "spatial_relation":
      splits_to_count = [
          "visual", "relation", "referent", "relative_position_1",
          "relative_position_2"
      ]
    else:
      raise ValueError(f"Unknown option for flag --split: {FLAGS.split}")
    for split in splits_to_count:
      logging.info("Equivalent examples in train and %s: %s", split,
                   grounded_scan.count_equivalent_examples("train", split))


def generate_data():
  """Generate data."""
  tf.io.gfile.makedirs(FLAGS.output_directory)
  grounded_scan = dataset.RelationGroundedScan(
      intransitive_verbs=FLAGS.intransitive_verbs,
      transitive_verbs=FLAGS.transitive_verbs,
      adverbs=FLAGS.adverbs,
      nouns=FLAGS.nouns,
      color_adjectives=FLAGS.color_adjectives,
      size_adjectives=FLAGS.size_adjectives,
      location_preps=FLAGS.location_preps,
      grid_size=FLAGS.grid_size,
      min_object_size=FLAGS.min_object_size,
      max_object_size=FLAGS.max_object_size,
      type_grammar=FLAGS.type_grammar,
      sample_vocabulary=FLAGS.sample_vocabulary,
      percentage_train=FLAGS.percentage_train,
      percentage_dev=FLAGS.percentage_dev,
      save_directory=FLAGS.output_directory,
      exclude_type_grammar=FLAGS.exclude_type_grammar)

  # Generate all possible commands from the grammar and exclude_grammar,
  # and pair them with relevant situations.
  grounded_scan.get_data_pairs(
      max_examples=FLAGS.max_examples,
      num_resampling=FLAGS.num_resampling,
      other_objects_sample_percentage=FLAGS.other_objects_sample_percentage,
      split_type=FLAGS.split,
      visualize_per_template=FLAGS.visualize_per_template,
      visualize_per_split=FLAGS.visualize_per_split,
      train_percentage=FLAGS.percentage_train,
      min_other_objects=FLAGS.min_other_objects,
      max_objects=FLAGS.max_objects,
      k_shot_generalization=FLAGS.k_shot_generalization,
      make_dev_set=FLAGS.make_dev_set,
      cut_off_target_length=FLAGS.cut_off_target_length,
      max_distract_objects=FLAGS.max_distract_objects,
      max_examples_per_target=FLAGS.max_examples_per_target)
  grounded_scan.save_dataset(FLAGS.save_dataset_as)
  gather_dataset_statistics(grounded_scan)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  if FLAGS.mode == "generate":
    generate_data()
  elif FLAGS.mode == "gather_stats":
    grounded_scan = dataset.RelationGroundedScan.load_dataset_from_file(
        FLAGS.load_dataset_from, FLAGS.output_directory)
    gather_dataset_statistics(grounded_scan)
  elif FLAGS.mode == "visualize":
    assert FLAGS.visualize_per_split, ("Please specify the number of examples "
                                       "to visualize per split.")
    grounded_scan = dataset.RelationGroundedScan.load_dataset_from_file(
        FLAGS.load_dataset_from, FLAGS.output_directory)
    grounded_scan.visualize_data_examples(FLAGS.visualize_per_split)
    logging.info("Saved visualizations in directory: %s.",
                 FLAGS.output_directory)


if __name__ == "__main__":
  app.run(main)
