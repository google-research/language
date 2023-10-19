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
r"""Generates training data for cross-attention model.

We generate a positive example for every relevant document.

We generate negative examples from two sources:
* Randomly selected documents.
* Predicted documents (e.g. from BM25) that are not relevant.

The input to the model is a query concatenated with the first N tokens of
the document. The output is a label indicating relevant or not relevant.

This file generates a tsv file with the input and output.
"""

import random

from absl import app
from absl import flags

from language.quest.common import document_utils
from language.quest.common import example_utils
from language.quest.common import tsv_utils
from language.quest.common import vocab_utils
from language.quest.xattn import xattn_utils

FLAGS = flags.FLAGS

flags.DEFINE_string("examples", "",
                    "Examples jsonl file with gold document set.")

flags.DEFINE_string("predictions", "",
                    "Examples jsonl file with predicted documents.")

flags.DEFINE_string("output", "", "Output tsv file with training examples.")

flags.DEFINE_string("documents", "", "Document corpus jsonl file.")

flags.DEFINE_integer(
    "random_negatives", 250, "Number of random negatives per query."
)

flags.DEFINE_integer("context_size", 384,
                     "Size of document context window in wordpieces.")

flags.DEFINE_integer(
    "predicted_negatives",
    250,
    "Maximum number of predicted documents to consider.",
)

flags.DEFINE_integer(
    "replicate_positives", 50, "Number of times to replicate positive examples."
)

flags.DEFINE_bool("shuffle", True, "Whether to shuffle examples.")

flags.DEFINE_integer("sp_model", "", "Path to T5 sentencepiece model.")


def main(unused_argv):
  # Load document corpus.
  documents = document_utils.read_documents(FLAGS.documents)
  # Map of document title to text.
  doc_title_to_text = {doc.title: doc.text for doc in documents}
  doc_titles = doc_title_to_text.keys()

  # Load examples and predicted documents.
  examples = example_utils.read_examples(FLAGS.examples)
  predictions = example_utils.read_examples(FLAGS.predictions)

  # Load vocab used for truncated documents.
  spm_wrapper = vocab_utils.T5SpmWrapper(FLAGS.sp_model)

  # List of tuples of (input, output) for T5.
  outputs = []

  for example, prediction in zip(examples, predictions):
    # Add every relevant doc as a positive example.
    relevant_titles = set(example.docs)
    for doc_title in relevant_titles:
      for _ in range(FLAGS.replicate_positives):
        new_example = xattn_utils.get_example(
            example.query,
            doc_title,
            doc_title_to_text,
            spm_wrapper,
            True,
            FLAGS.context_size
        )
        outputs.append(new_example)

    # Add non-relevant predicted docs as negative example.
    num_predicted_negatives = 0
    for doc_title in prediction.docs:
      if doc_title not in relevant_titles:
        new_example = xattn_utils.get_example(
            example.query,
            doc_title,
            doc_title_to_text,
            spm_wrapper,
            False,
            FLAGS.context_size
        )
        outputs.append(new_example)
        num_predicted_negatives += 1
        if num_predicted_negatives >= FLAGS.predicted_negatives:
          break

    # Add additional random negatives.
    if FLAGS.random_negatives > 0:
      random_titles = random.sample(list(doc_titles), FLAGS.random_negatives)
      for doc_title in random_titles:
        if doc_title not in relevant_titles:
          new_example = xattn_utils.get_example(
              example.query,
              doc_title,
              doc_title_to_text,
              spm_wrapper,
              False,
              FLAGS.context_size
          )
          outputs.append(new_example)

  # Write training examples to output file.
  if FLAGS.shuffle:
    random.shuffle(outputs)
  tsv_utils.write_tsv(outputs, FLAGS.output)


if __name__ == "__main__":
  app.run(main)
