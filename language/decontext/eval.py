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
"""Evaluating predictions of decontextualization task.

  It requires "annotation" and "prediction" file, both in jsonl format.

  The format of annotations file can be found in README.md

  Each line for predictions file contains:

  prediction: {example_id: 3245252,
               category: "DONE",
                   // category should be one of DONE, IMPOSSIBLE, UNNECESSARY.
               decontextualized_sentence: "Venus is one of the planets.",
                   // when category is "IMPOSSIBLE", this field is ignored.
               original_sentence: "It is one of the planets."}

"""
import functools
import re
import string

from absl import app
from absl import flags
from absl import logging
from language.decontext import decontext_util
from language.decontext import eval_util
import nltk
from sentencepiece import SentencePieceProcessor

FLAGS = flags.FLAGS

flags.DEFINE_string("annotations", None,
                    "Path to a annotations file, where each line is a JSON.")
flags.DEFINE_string("predictions", None,
                    "Path to a predictions file, where each line is a JSON.")
flags.DEFINE_enum(
    "word_tokenizer",
    "nltk_english",
    ["nltk_english", "whitespace", "character", "sentencepiece"],
    "Word tokenizer to use to compute SARI scores. The NLTK English tokenizer "
    "is the default.",
)
flags.DEFINE_string(
    "sentencepiece_model_path",
    None,
    "Path to the sentencepiece model when --word_tokenizer=sentencepiece.",
)


@functools.lru_cache()
def get_sentencepiece_tokenizer(vocab_path):
  logging.info("Loading SP model from %s", vocab_path)
  sp_tokenizer = SentencePieceProcessor()
  with open(vocab_path, "rb") as f:
    sp_tokenizer.LoadFromSerializedProto(f.read())
  return sp_tokenizer


def make_sentencepiece_tokenizer(vocab_path):
  """Splits on punctuation and whitespace, then does Sentencepiece tokenization.
  """
  sp_tokenizer = get_sentencepiece_tokenizer(vocab_path)

  def tokenizer(text):
    split_on_punctuation = re.split(
        r"\s+|\s*([%s])\s*" % re.escape(string.punctuation), text)
    spaced = " ".join(x for x in split_on_punctuation if x)
    tokens = sp_tokenizer.EncodeAsPieces(spaced)
    tokens = [t.lstrip("▁") for t in tokens if t and t != "▁"]
    return tokens

  return tokenizer


def main(argv):
  del argv
  FLAGS.logtostderr = True
  FLAGS.log_prefix = False
  if FLAGS.word_tokenizer == "whitespace":
    word_tokenizer = lambda text: text.strip().split()
  elif FLAGS.word_tokenizer == "character":
    word_tokenizer = lambda text: list(text.strip())
  elif FLAGS.word_tokenizer == "sentencepiece":
    if FLAGS.sentencepiece_model_path is None:
      raise ValueError(
          "Missing sentencepiece_model_path when word_tokenizer=sentencepiece")
    word_tokenizer = make_sentencepiece_tokenizer(
        FLAGS.sentencepiece_model_path)
    logging.info(str(word_tokenizer("Using the SP word tokenizer")))
  elif FLAGS.word_tokenizer == "nltk_english":
    nltk.download("punkt", quiet=True)
    word_tokenizer = None
  else:
    raise ValueError(f"Unknwon word_tokenizer '{FLAGS.word_tokenizer}'")
  logging.info("Using the '%s' word tokenizer", FLAGS.word_tokenizer)
  annotation_dict = decontext_util.load_annotations(FLAGS.annotations)
  logging.info("%d examples in annotation.", len(annotation_dict))

  # Evaluates generated decontextualized sentence.
  logging.info("== Reference human score ==")
  (original_sent_dict, reference_sents_dict,
   median_human_sent_dict) = eval_util.process_annotation(annotation_dict)
  eval_util.compute_sentence_generation_scores(
      original_sent_dict=original_sent_dict,
      reference_dict=reference_sents_dict,
      pred_dict=median_human_sent_dict,
      word_tokenizer=word_tokenizer,
  )
  if FLAGS.predictions is None:
    return
  prediction_dict = decontext_util.load_predictions(FLAGS.predictions)
  pred_sent_dict = eval_util.get_sent_dict(prediction_dict)
  logging.info("%d examples in predicton.", len(prediction_dict))
  logging.info("== Prediction score ==")
  eval_util.compute_sentence_generation_scores(
      original_sent_dict,
      reference_sents_dict,
      pred_sent_dict,
      word_tokenizer=word_tokenizer,
  )

  # Evaluates decontextualization category classification
  logging.info("== Feasibility classification score ==")
  eval_util.score_classification(annotation_dict, prediction_dict)


if __name__ == "__main__":
  flags.mark_flag_as_required("annotations")
  app.run(main)
