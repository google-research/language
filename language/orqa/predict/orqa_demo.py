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
"""ORQA demo."""
import os
from wsgiref import simple_server

from absl import flags
import jinja2
from language.orqa.models import orqa_model
import six
import tensorflow.compat.v1 as tf
import tornado.web
import tornado.wsgi


FLAGS = flags.FLAGS

flags.DEFINE_integer("port", 8080, "Port to listen on.")
flags.DEFINE_string("web_path", "", "Directory containing all web resources")
flags.DEFINE_string("model_dir", None, "Model directory.")


class MainHandler(tornado.web.RequestHandler):
  """Main handler."""
  _tmpl = None
  _predictor = None

  def initialize(self, env, predictor):
    self._tmpl = env.get_template("orqa.html")
    self._predictor = predictor

  def get(self):
    question = self.get_argument("question", default="")
    if question:
      predictions = self._predictor(question)
      tf.logging.info("=" * 80)
      tf.logging.info(question)
      tf.logging.info(predictions)
      tf.logging.info("=" * 80)

      orig_tokens = [
          six.ensure_text(t, errors="ignore")
          for t in predictions["orig_tokens"]
      ]
      start = predictions["orig_start"]
      end = predictions["orig_end"] + 1
      answer_left_context = " ".join(orig_tokens[:start])
      answer_text = " ".join(orig_tokens[start:end])
      answer_right_context = " ".join(orig_tokens[end:])
    else:
      answer_left_context = ""
      answer_text = ""
      answer_right_context = ""
    self.write(self._tmpl.render(
        question=question,
        retrieved_doc_key="",
        answer_left_context=answer_left_context,
        answer_text=answer_text,
        answer_right_context=answer_right_context))


def dummy_predictor(_):
  return dict(
      orig_tokens=["the", "quick", "brown", "fox"] * 1000,
      orig_start=1,
      orig_end=2)


def main(_):
  predictor = orqa_model.get_predictor(FLAGS.model_dir)

  # Run once to initialize the block records.
  predictor("")

  web_path = FLAGS.web_path
  if not web_path:
    web_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "web")

  env = jinja2.Environment(loader=jinja2.FileSystemLoader(web_path))
  application = tornado.wsgi.WSGIApplication([(r"/", MainHandler, {
      "env": env,
      "predictor": predictor,
  })])
  tf.logging.info("READY!")

  server = simple_server.make_server("", FLAGS.port, application)
  server.serve_forever()


if __name__ == "__main__":
  tf.disable_v2_behavior()
  tf.app.run()
