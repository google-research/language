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
r"""Caches retrievals for retrieval-augmented dataset generation.

This script uses a retriever defined in query_retrievers.
"""
from absl import app
from absl import flags
from absl import logging
from language.casper.retrieve import query_retrievers
from language.casper.utils import top_utils

FLAGS = flags.FLAGS

flags.DEFINE_list("index_files", [], "Index JSONL files")
flags.DEFINE_list("example_files", [], "Example JSONL files")
flags.DEFINE_list("output_files", [],
                  "Output JSONL filenames (same length as example_files).")

flags.DEFINE_enum("retriever", None,
                  ["bert_pooled", "bert_cls", "bert_avg", "use"],
                  "Retriever to use.")
flags.DEFINE_string(
    "embedder_size", "base", "Size of the embedder (e.g., 'base' or 'large'). "
    "Only used for embedding-based retrievers.")

flags.DEFINE_enum("neighbor_filter", "simple", [
    "simple", "crowded", "match_domain", "match_intent", "match_frame",
    "match_domain_crowded", "match_intent_crowded"
], "Neighbor filter.")

flags.DEFINE_integer("max_neighbors", 100,
                     "Maximum number of neighbors to output.")
flags.DEFINE_integer("batch_size", 32, "Batch size for embedding queries.")
flags.DEFINE_integer("log_every", 1000, "Log every this number of examples.")


def _get_domain(ex: query_retrievers.Example) -> str:
  return ex["domain"]


def _get_frame(ex: query_retrievers.Example) -> str:
  return top_utils.get_frame_top(ex["output_str"])


def _get_intent(ex: query_retrievers.Example) -> str:
  return top_utils.get_frame_top(ex["output_str"]).split("-")[0]


def main(_):
  if len(FLAGS.example_files) != len(FLAGS.output_files):
    raise ValueError(
        "example_files and output_files must have the same length.")

  # Load the entire retrieval index
  index_exs = []
  for filename in FLAGS.index_files:
    index_exs.extend(query_retrievers.read_jsonl(filename))
  logging.info("Read %d index entries.", len(index_exs))

  # Construct a retriever and embed the index
  if FLAGS.retriever.startswith("bert_"):
    embed_method = FLAGS.retriever.split("_")[1]
    retriever = query_retrievers.BertRetriever(
        index_exs,
        bert_size=FLAGS.embedder_size,
        embed_method=embed_method,
        batch_size=FLAGS.batch_size,
        log_every=FLAGS.log_every)
  elif FLAGS.retriever == "use":
    retriever = query_retrievers.USERetriever(
        index_exs,
        use_size=FLAGS.embedder_size,
        batch_size=FLAGS.batch_size,
        log_every=FLAGS.log_every)
  else:
    raise ValueError("Unknown retriever: {}".format(FLAGS.retriever))

  # Construct a neighbor filter
  if FLAGS.neighbor_filter == "simple":
    neighbor_filter = query_retrievers.SimpleNeighborFilter()
  elif FLAGS.neighbor_filter == "crowded":
    neighbor_filter = query_retrievers.CrowdedNeighborFilter(_get_frame)
  elif FLAGS.neighbor_filter == "match_domain":
    neighbor_filter = query_retrievers.PropertyMatchNeighborFilter(_get_domain)
  elif FLAGS.neighbor_filter == "match_intent":
    neighbor_filter = query_retrievers.PropertyMatchNeighborFilter(_get_intent)
  elif FLAGS.neighbor_filter == "match_frame":
    neighbor_filter = query_retrievers.PropertyMatchNeighborFilter(_get_frame)
  elif FLAGS.neighbor_filter == "match_domain_crowded":
    neighbor_filter = query_retrievers.ChainedNeighborFilter([
        query_retrievers.PropertyMatchNeighborFilter(_get_domain),
        query_retrievers.CrowdedNeighborFilter(_get_frame)
    ])
  elif FLAGS.neighbor_filter == "match_intent_crowded":
    neighbor_filter = query_retrievers.ChainedNeighborFilter([
        query_retrievers.PropertyMatchNeighborFilter(_get_intent),
        query_retrievers.CrowdedNeighborFilter(_get_frame)
    ])
  else:
    raise ValueError("Unknown neighbor filter: {}".format(
        FLAGS.neighbor_filter))

  # Retrieval
  for example_file, output_file in zip(FLAGS.example_files, FLAGS.output_files):
    logging.info("Retrieving %s --> %s", example_file, output_file)
    examples = query_retrievers.read_jsonl(example_file)
    retriever.dump_neighbors(
        examples,
        output_file,
        neighbor_filter,
        max_neighbors=FLAGS.max_neighbors)


if __name__ == "__main__":
  app.run(main)
