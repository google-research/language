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
"""Retrieves topics for a collection of article pairs."""
import csv
import json
import time

from absl import app
from absl import flags
from absl import logging
import requests
import tensorflow as tf
from tqdm import tqdm

flags.DEFINE_string("article_pairs_jsonl", None,
                    "JSONL file containing articles to predict topics for.")
flags.DEFINE_string("output_tsv", None, "Where to write output.")
FLAGS = flags.FLAGS

ENWIKI_ENDPOINT = "https://en.wikipedia.org/w/api.php"
ORES_ENDPOINT = "https://ores.wikimedia.org/v3/scores/enwiki/"
QUERY_SIZE = 50
MAX_RETRIES = 20


def generate_article_pairs(pattern):
  """Generates article pairs from multiple jsonl files."""
  filenames = tf.io.gfile.glob(pattern)
  for filename in filenames:
    with tf.io.gfile.GFile(filename, "r") as f:
      for line in f:
        yield json.loads(line)


def request_helper(*args, **kwargs):
  """Helps rerun requests upon failure."""
  for _ in range(MAX_RETRIES):
    try:
      response = requests.get(*args, **kwargs)
    except requests.exceptions.ConnectionError:
      time.sleep(1)
    else:
      return response


def retrieve_latest_revids(pageids_chunk):
  """Gets the latest revision id for an article."""
  headers = {"Accept": "application/json"}
  params = {
      "action": "query",
      "format": "json",
      "pageids": "|".join(str(x) for x in pageids_chunk),
      "prop": "revisions",
      "rvprop": "ids",
  }
  response = request_helper(ENWIKI_ENDPOINT, headers=headers, params=params)
  response_json = response.json()
  pages = response_json["query"]["pages"]
  output = {}
  for pageid, value in pages.items():
    revisions = value.get("revisions", None)
    if revisions is None:
      continue
    revid = revisions.pop().get("revid", None)
    if revid is not None:
      output[pageid] = revid
  return output


def retrieve_topics(revids):
  """Gets the topics for a collection of revision ids."""
  headers = {"Accept": "application/json"}
  params = {
      "models": "drafttopic",
      "revids": "|".join(str(x) for x in revids.values())
  }
  response = request_helper(ORES_ENDPOINT, headers=headers, params=params)
  response_json = response.json()

  output = {}
  scores = response_json["enwiki"]["scores"]
  reverse_lookup = {revid: pageid for pageid, revid in revids.items()}
  for revid, predictions in scores.items():
    # Get top prediction
    probabilities = predictions["drafttopic"]["score"]["probability"]
    topic, _ = sorted(list(probabilities.items()), key=lambda x: x[1]).pop()

    output[reverse_lookup[int(revid)]] = topic
  return output


def main(_):
  logging.info("Extracting pageids")
  pageids = []
  for article_pair in generate_article_pairs(FLAGS.article_pairs_jsonl):
    pageids.append(article_pair["target_article"]["id"])

  pageids_chunks = [
      pageids[i:i + QUERY_SIZE] for i in range(0, len(pageids), QUERY_SIZE)
  ]
  all_topics = {}
  logging.info("Extracting topics")
  for pageids_chunk in tqdm(pageids_chunks):
    latest_revids = retrieve_latest_revids(pageids_chunk)
    if latest_revids is None:
      continue
    topics = retrieve_topics(latest_revids)
    all_topics.update(topics)
  with tf.io.gfile.GFile(FLAGS.output_tsv, "w") as g:
    writer = csv.writer(g, delimiter="\t")
    for row in all_topics.items():
      writer.writerow(row)


if __name__ == "__main__":
  app.run(main)
