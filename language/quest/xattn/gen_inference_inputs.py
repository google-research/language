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
r"""Generates data to be used as input for inference.

We generate two files:
(1) A newline-separated txt file with string-formatted inputs.
(2) A tsv file with metadata associated with each line of (1).

We run T5 on (1). We use (1) and (2) during post-processing.
"""

from absl import app
from absl import flags

from language.quest.common import document_utils
from language.quest.common import example_utils
from language.quest.common import tsv_utils
from language.quest.common import vocab_utils
from language.quest.xattn import xattn_utils

FLAGS = flags.FLAGS

flags.DEFINE_string("predictions", "",
                    "Examples jsonl file with predicted documents.")

flags.DEFINE_string("documents", "", "Document corpus jsonl file.")

flags.DEFINE_string("output", "", "Output txt file.")

flags.DEFINE_string("metadata", "", "Output tsv file.")

flags.DEFINE_integer("context_size", 384,
                     "Size of document context window in wordpieces.")

flags.DEFINE_integer("topk", 100,
                     "Maximum number of predicted documents to consider.")

flags.DEFINE_integer("sp_model", "",
                     "Path to T5 sentencepiece model.")


def main(unused_argv):
  # Load document corpus.
  documents = document_utils.read_documents(FLAGS.documents)
  # Map of document title to text.
  doc_title_to_text = {doc.title: doc.text for doc in documents}

  # Load predictions.
  predictions = example_utils.read_examples(FLAGS.predictions)

  # Load vocab used for truncated documents.
  spm_wrapper = vocab_utils.T5SpmWrapper(FLAGS.sp_model)

  # Data to write to output files.
  input_strings = []
  # Metadata consists of (query, document title).
  metadata = []

  for prediction in predictions:
    for idx, doc_title in enumerate(prediction.docs):
      if idx >= FLAGS.topk:
        break
      # We set the target to True and generate a likelihood score.
      is_relevant = True
      new_example = xattn_utils.get_example(
          prediction.query,
          doc_title,
          doc_title_to_text,
          spm_wrapper,
          is_relevant,
          FLAGS.context_size
      )
      input_strings.append(new_example)
      # Repeat metadata entries for each chunk
      metadata += [(prediction.query, doc_title)]

  tsv_utils.write_tsv(input_strings, FLAGS.output)
  tsv_utils.write_tsv(metadata, FLAGS.metadata)


if __name__ == "__main__":
  app.run(main)
