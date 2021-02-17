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
"""Generate gold targets with database ID for Spider evaluation."""

from absl import app
from absl import flags

from language.nqg.tasks import tsv_utils
from language.nqg.tasks.spider import database_constants

from tensorflow.io import gfile

FLAGS = flags.FLAGS

flags.DEFINE_string("input", "",
                    "Input tsv file (e.g. output of split_dataset.py).")

flags.DEFINE_string("output", "", "Output txt file.")


def main(unused_argv):
  formatted_db_id_to_db_id = {}
  for db_id in database_constants.DATABASES:
    formatted_db_id_to_db_id[db_id.lower()] = db_id
    formatted_db_id_to_db_id[db_id] = db_id

  examples = tsv_utils.read_tsv(FLAGS.input)
  with gfile.GFile(FLAGS.output, "w") as txt_file:
    for example in examples:
      db_id = example[0].split()[0].rstrip(":")
      db_id = formatted_db_id_to_db_id[db_id]
      txt_file.write("%s\t%s\n" % (example[1], db_id))


if __name__ == "__main__":
  app.run(main)
