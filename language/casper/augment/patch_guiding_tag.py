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
r"""Add guiding tags to examples that fit the criteria."""
import os


from absl import app
from absl import flags
from absl import logging
from language.casper.utils import top_utils
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_string("src_pattern", "", "Pattern of the source dataset files.")
flags.DEFINE_string("tgt_dir", "", "Output directory.")
flags.DEFINE_enum("file_format", "tfr", ["tsv", "tfr"], "Dataset file format.")
flags.DEFINE_list(
    "affected_labels", [], "A list of labels (e.g., 'IN:SET_NAME'). "
    "If an exemplar has any of these labels, add a guiding tag.")

QUERY_EXEMPLARS_SEP = " @@ "
GUIDING_TAG_BEFORE = " @@ "
GUIDING_TAG_AFTER = " @@ PLATINUM "


def _format_label(label):
  """Converts labels such as 'IN:SET_NAME' to 'IN set name ='."""
  return top_utils.format_serialized("[" + label).lstrip("[")


def process_tsv(src_file, tgt_file,
                affected_labels):
  """Processes examples in the TSV file src_file and writes to tgt_file.

  Args:
    src_file: Source TSV file
    tgt_file: Target TSV file
    affected_labels: List of labels to detect. If an exemplar has any of these
      labels, add a guiding tag.

  Returns:
    (number of affected examples, total number of examples)
  """
  num_affected = num_total = 0
  with tf.io.gfile.GFile(src_file) as reader:
    with tf.io.gfile.GFile(tgt_file, "w") as writer:
      for line in reader:
        num_total += 1
        input_str, output_str = line.rstrip("\n").split("\t")
        exemplars = input_str.split(QUERY_EXEMPLARS_SEP, 1)[1]
        if any(x in exemplars for x in affected_labels):
          num_affected += 1
          input_str = input_str.replace(GUIDING_TAG_BEFORE, GUIDING_TAG_AFTER)
        writer.write("{}\t{}\n".format(input_str, output_str))
  return num_affected, num_total


def process_tfr(src_file, tgt_file,
                affected_labels):
  """Processes examples in the TFRecord file src_file and writes to tgt_file.

  Args:
    src_file: Source TFRecord file
    tgt_file: Target TFRecord file
    affected_labels: List of labels to detect. If an exemplar has any of these
      labels, add a guiding tag.

  Returns:
    (number of affected examples, total number of examples)
  """
  num_affected = num_total = 0
  with tf.io.TFRecordWriter(tgt_file) as writer:
    for raw_example in tf.data.TFRecordDataset(src_file):
      num_total += 1
      example = tf.train.Example.FromString(raw_example.numpy())
      inputs_feature = example.features.feature["inputs"]
      input_str = inputs_feature.bytes_list.value[0].decode()
      exemplars = input_str.split(QUERY_EXEMPLARS_SEP, 1)[1]
      if any(x in exemplars for x in affected_labels):
        num_affected += 1
        input_str = input_str.replace(GUIDING_TAG_BEFORE, GUIDING_TAG_AFTER)
        inputs_feature.bytes_list.value[0] = input_str.encode()
      writer.write(example.SerializeToString())
  return num_affected, num_total


def main(_):
  # Detect both label formats: 'IN:SET_NAME' and 'IN set name ='
  affected_labels = set(FLAGS.affected_labels +
                        [_format_label(x) for x in FLAGS.affected_labels])
  tf.io.gfile.makedirs(FLAGS.tgt_dir)
  for src_file in tf.io.gfile.glob(FLAGS.src_pattern):
    tgt_file = os.path.join(FLAGS.tgt_dir, os.path.basename(src_file))
    num_total = num_affected = 0
    if FLAGS.file_format == "tsv":
      num_affected, num_total = process_tsv(src_file, tgt_file, affected_labels)
    elif FLAGS.file_format == "tfr":
      num_affected, num_total = process_tsv(src_file, tgt_file, affected_labels)
    else:
      raise ValueError("Unknown file format: {}".format(FLAGS.file_format))

    logging.info("Changed %d / %d (%.2f) examples in %s", num_affected,
                 num_total, num_affected * 100. / (num_total + 1e-9), tgt_file)


if __name__ == "__main__":
  app.run(main)
