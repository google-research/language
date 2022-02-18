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
"""Split dataset tsv file based on target templates.

This uses a similar method of anonymizing string literals and integers as:
https://arxiv.org/abs/1806.09029
"""

from absl import app
from absl import flags

from language.compgen.nqg.tasks import template_utils
from language.compgen.nqg.tasks import tsv_utils
from language.compgen.nqg.tasks.spider import sql_tokenizer

FLAGS = flags.FLAGS

flags.DEFINE_string("input", "", "Input tsv file.")

flags.DEFINE_string(
    "output_1", "",
    "Output tsv file containing up to `max_num_examples_1` examples.")

flags.DEFINE_string("output_2", "",
                    "Output tsv file containing the remaining examples.")

flags.DEFINE_float("max_num_examples_1", 3282,
                   "Maximum number of examples for output_1.")

flags.DEFINE_integer("seed", 1, "Seed for splitting examples.")

# Placeholder symbol for anonymized aspects of target templates.
PLACEHOLDER = "___"


def is_number(token):
  """Check if token is a SQL number literal."""
  # Note that Python's is_numeric() will return False for values like 30.3.
  try:
    float(token)
    return True
  except ValueError:
    return False


def spider_template_fn(target):
  """Anonymize quoted substrings and numbers in SQL."""
  # First, replace any numeric token.
  tokens = sql_tokenizer.tokenize_sql(target)
  template_tokens = []
  for token in tokens:
    if is_number(token):
      template_tokens.append(PLACEHOLDER)
    else:
      template_tokens.append(token)
  template = " ".join(template_tokens)

  # Second, replace any subspan surrounded by single or double quotes.
  in_quotes = False
  quote_token = None
  new_template = ""
  for char in template:
    if in_quotes:
      if char == quote_token:
        in_quotes = False
        quote_token = None
    else:
      if char in ("'", "\""):
        in_quotes = True
        quote_token = char
        new_template += PLACEHOLDER
      else:
        new_template += char
  return new_template


def main(unused_argv):
  examples = tsv_utils.read_tsv(FLAGS.input)
  examples_1, examples_2 = template_utils.split_by_template(
      examples,
      template_fn=spider_template_fn,
      max_num_examples_1=FLAGS.max_num_examples_1,
      seed=FLAGS.seed)
  tsv_utils.write_tsv(examples_1, FLAGS.output_1)
  tsv_utils.write_tsv(examples_2, FLAGS.output_2)


if __name__ == "__main__":
  app.run(main)
