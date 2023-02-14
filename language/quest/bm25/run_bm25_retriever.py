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
"""Generate predicted candidates with BM25."""

import random

from absl import app
from absl import flags

from language.quest.bm25 import bm25_retriever
from language.quest.common import document_utils
from language.quest.common import example_utils


FLAGS = flags.FLAGS

flags.DEFINE_string("examples", "", "Path to examples jsonl file.")

flags.DEFINE_string("docs", "", "Path to document corpus jsonl file.")

flags.DEFINE_string("output", "", "Path to write predictions jsonl file.")

flags.DEFINE_integer("sample", 0, "Number of examples to sample if > 0.")

flags.DEFINE_integer("topk", 1000, "Number of documents to retrieve.")


def main(unused_argv):
  examples = example_utils.read_examples(FLAGS.examples)

  if FLAGS.sample > 0:
    random.shuffle(examples)
    examples = examples[:FLAGS.sample]

  print("Reading documents.")
  documents = document_utils.read_documents(FLAGS.docs)
  print("Finished reading documents.")
  print("Initializing BM25 retriever.")
  retriever = bm25_retriever.BM25Retriever(documents)
  print("Finished initializing BM25 retriever.")

  predictions = []
  for idx, example in enumerate(examples):
    print("Processing example %s." % idx)
    docs_scores = retriever.get_docs_and_scores(example.query, topk=FLAGS.topk)
    docs = [doc for doc, _ in docs_scores]
    scores = [score for _, score in docs_scores]
    predictions.append(
        example_utils.Example(
            original_query=example.original_query,
            query=example.query,
            docs=docs,
            scores=scores,
            metadata=example.metadata
        )
    )

  example_utils.write_examples(FLAGS.output, predictions)


if __name__ == "__main__":
  app.run(main)
