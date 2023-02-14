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
r"""Map examples to indexed files.

Writes three tsv files with rows:
* (query ID, query)
* (query ID, document ID)
* (query, document)

This requires that `write_doc_idx_maps.py` has been run to provide a TSV file
mapping document titles to IDs.
"""

import random

from absl import app
from absl import flags

from language.quest.common import example_utils
from language.quest.common import tsv_utils

# Setting arbitrary random seed for reproducibility.
random.seed(42)

FLAGS = flags.FLAGS

flags.DEFINE_string("examples", "", "Examples jsonl file.")

flags.DEFINE_string("doc_title_map", "", "TSV of doc ids and titles.")

flags.DEFINE_string("doc_text_map", "", "TSV of doc ids and text.")

flags.DEFINE_string("indexed_queries", "",
                    "Output filepath for indexed queries tsv file.")

flags.DEFINE_string("indexed_examples", "",
                    "Output filepath for indexed examples tsv file.")

flags.DEFINE_string("out_examples", "",
                    "Output filepath for examples tsv file passed to t5xr.")


def main(unused_argv):
  # Load mapping of doc titles to ids.
  doc_title_ids = tsv_utils.read_tsv(FLAGS.doc_title_map)
  doc_title_to_idx = {doc_title: idx for idx, doc_title in doc_title_ids}
  doc_text_ids = tsv_utils.read_tsv(FLAGS.doc_text_map, max_splits=1)
  doc_idx_to_text = {idx: doc_text for idx, doc_text in doc_text_ids}

  # Load examples and map to IDs.
  num_missing_docs = 0
  examples = example_utils.read_examples(FLAGS.examples)
  query_ids_queries = []
  query_ids_doc_ids = []
  queries_docs = []
  for query_idx, example in enumerate(examples):
    for doc_title in example.docs:
      if doc_title not in doc_title_to_idx:
        num_missing_docs += 1
        print("Missing document title: `%s`" % doc_title)
        continue
      doc_idx = doc_title_to_idx[doc_title]
      doc_text = (
          doc_idx_to_text[doc_idx]
          .replace("\t", " ")
          .replace("'''", "")
      )
      query_ids_doc_ids.append((query_idx, doc_idx))
      queries_docs.append((example.query, doc_text))
    if example.docs:
      query_ids_queries.append((query_idx, example.query))

  print("num_missing_docs: %s" % num_missing_docs)
  tsv_utils.write_tsv(query_ids_queries, FLAGS.indexed_queries)
  tsv_utils.write_tsv(query_ids_doc_ids, FLAGS.indexed_examples)
  random.shuffle(queries_docs)
  tsv_utils.write_tsv(queries_docs, FLAGS.out_examples)


if __name__ == "__main__":
  app.run(main)
