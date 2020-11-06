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
"""A script to convert bottom up features from TSV to HDF5 format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64
import csv
import glob
import sys

from absl import app
from absl import flags
import h5py
import numpy as np

flags.DEFINE_string("input_pattern", None, "Input file pattern.")

flags.DEFINE_string("output_file", None, "Output HDF5 file.")

FLAGS = flags.FLAGS

csv.field_size_limit(sys.maxsize)

FIELDS = ("image_id", "image_w", "image_h", "num_boxes", "boxes", "features")


def main(_):
  with h5py.File(FLAGS.output_file, "w") as hdf5_file:
    for filename in glob.glob(FLAGS.input_pattern):
      with open(filename, "r") as tsv_file:
        reader = csv.DictReader(tsv_file, delimiter="\t", fieldnames=FIELDS)
        for i, item in enumerate(reader):
          image_id = item["image_id"]
          num_boxes = int(item["num_boxes"])
          features = np.frombuffer(
              base64.decodebytes(item["features"].encode("utf-8")),
              dtype=np.float32).reshape((num_boxes, -1))
          boxes = np.frombuffer(
              base64.decodebytes(item["boxes"].encode("utf-8")),
              dtype=np.float32).reshape((num_boxes, -1))
          dims = np.array([float(item["image_h"]), float(item["image_w"])])
          hdf5_file.create_dataset("features-" + image_id, data=features)
          hdf5_file.create_dataset("bboxes-" + image_id, data=boxes)
          hdf5_file.create_dataset("dims-" + image_id, data=dims)
          if i % 1000 == 0:
            print("Processed %d items." % i)


if __name__ == "__main__":
  app.run(main)
