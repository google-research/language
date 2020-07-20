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
"""A simple web api wrapper around the wikipedia sql db.

This is helpful if trying to query the fever wikipedia dump without needing
to use directly access protobufs.
"""
from absl import app
from absl import flags
import flask
from language.serene import wiki_db

FLAGS = flags.FLAGS
flags.DEFINE_string('wiki_db_path', None, '')


def main(_):
  db = wiki_db.WikiDatabase.from_local(FLAGS.wiki_db_path)
  flask_app = flask.Flask(__name__)

  @flask_app.route('/wiki_page_sentence', methods=['POST'])
  def get_page_sentence():  # pylint: disable=unused-variable
    request = flask.request.json
    maybe_sentence = db.get_page_sentence(request['wikipedia_url'],
                                          int(request['sentence_id']))
    return flask.jsonify({'text': maybe_sentence})

  flask_app.run()


if __name__ == '__main__':
  app.run(main)
