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
r"""Generate a retrieval-augmented dataset from cached retrievals."""
import json

from absl import app
from absl import flags
from language.casper.augment import cached_retrieval_to_dataset_lib
from language.casper.augment import casper_converters

FLAGS = flags.FLAGS

flags.DEFINE_list("train_data_paths", [],
                  "Training data JSONL files or glob patterns.")
flags.DEFINE_list("dev_data_paths", [],
                  "Development data JSONL files or glob patterns.")
flags.DEFINE_list("test_data_paths", [],
                  "Test data JSONL files or glob patterns.")
flags.DEFINE_list("index_paths", [],
                  "Cached retrieval index files or glob patterns.")

flags.DEFINE_string("output_dir", None, "Directory to output dataset files.")

flags.DEFINE_enum(
    "example_converter", "query_only",
    ["query_only", "add_top", "add_samp", "add_oracle", "add_adversarial"],
    "The converter for converting cached example to augmented example.")
flags.DEFINE_string(
    "converter_kwargs", "{}",
    "Keyword arguments for the example converter (as serialized JSON)")
flags.DEFINE_string(
    "formatter_kwargs", "{}",
    "Keyword arguments for the example formatter (as serialized JSON)")

flags.DEFINE_enum("funcall_format", "top", ["top"],
                  "Format of the output function call or logical form.")
flags.DEFINE_string("train_filename", "train", "Train data filename.")
flags.DEFINE_string("dev_filename", "dev", "Dev data filename.")
flags.DEFINE_string("test_filename", "test", "Test data filename.")
flags.DEFINE_enum("file_format", "tfr", ["tsv", "tfr"], "Output file format.")
flags.DEFINE_integer("log_every", 1000,
                     "Frequency of logging the number of generated examples.")
flags.DEFINE_integer("seed", 42, "Random seed.")


def main(_):
  retrieval_index = list(
      cached_retrieval_to_dataset_lib.read_orig_examples(FLAGS.index_paths))
  converter = casper_converters.get_converter(
      FLAGS.example_converter,
      retrieval_index,
      funcall_format=FLAGS.funcall_format,
      converter_kwargs=json.loads(FLAGS.converter_kwargs),
      formatter_kwargs=json.loads(FLAGS.formatter_kwargs))
  cached_retrieval_to_dataset_lib.generate_dataset(
      cached_retrieval_to_dataset_lib.read_orig_examples(
          FLAGS.train_data_paths),
      cached_retrieval_to_dataset_lib.read_orig_examples(FLAGS.dev_data_paths),
      cached_retrieval_to_dataset_lib.read_orig_examples(FLAGS.test_data_paths),
      converter,
      FLAGS.output_dir,
      seed=FLAGS.seed,
      log_every=FLAGS.log_every,
      train_filename=FLAGS.train_filename,
      dev_filename=FLAGS.dev_filename,
      test_filename=FLAGS.test_filename,
      file_format=FLAGS.file_format)


if __name__ == "__main__":
  app.run(main)
