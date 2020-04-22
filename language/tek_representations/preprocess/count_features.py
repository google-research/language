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
"""Count all features across files."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function
import collections
import os
from absl import app
from absl import flags

import tensorflow.compat.v1 as tf

FLAGS = flags.FLAGS
flags.DEFINE_string('preprocessed_dir', None,
                    'Dir containing preprocessed files')
flags.DEFINE_string('output_file', None, 'File(s) containing counts')


def count_records(file_name):
  num_records = 0
  for _ in tf.python_io.tf_record_iterator(file_name):
    num_records += 1
  return num_records


def count_features(preprocessed_dir, output_file):
  """Loop over files and count features."""
  counts = collections.defaultdict(int)
  for fpath in tf.gfile.Glob(
      os.path.join(preprocessed_dir, '*', '*', '*tfrecord*')):
    num_records = count_records(fpath)
    parts = fpath.split('/')
    counts['/'.join(parts[-3:-1])] += num_records
  with tf.gfile.Open(output_file, 'w') as f:
    for config, count in counts.items():
      f.write("('" + config + "', " + str(count) + ')')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  count_features(FLAGS.preprocessed_dir, FLAGS.output_file)


if __name__ == '__main__':
  app.run(main)
