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
# Lint as: python3
"""Sqlite wikipedia db that stores protos by fever formatted wikipedia url."""
import os


from absl import logging
from language.serene import fever_pb2
from language.serene import util
# pylint: disable=g-bad-import-order
import sqlalchemy
from sqlalchemy.ext import declarative
from sqlalchemy import schema
from sqlalchemy import orm
from sqlalchemy import types
import tensorflow.compat.v2 as tf
import tqdm


Base = declarative.declarative_base()  # pylint: disable=invalid-name


def create_session(db_path):
  engine = sqlalchemy.create_engine(f'sqlite:///{db_path}')
  Session = orm.sessionmaker(bind=engine)  # pylint: disable=invalid-name
  return engine, Session()


class WikiPage(Base):
  __tablename__ = 'wikipedia'
  id = schema.Column(types.Integer, primary_key=True)
  wikipedia_url = schema.Column(types.String, index=True)
  proto = schema.Column(types.LargeBinary)


class WikiDatabase:
  """A wrapper around sqlite wikipedia database that returns proto pages."""

  def __init__(self, db_path):
    """Constructor.

    Args:
      db_path: Path to db to read or write
    """
    self._db_path: Text = db_path
    self._engine, self.session = create_session(self._db_path)

  def create(self):
    Base.metadata.create_all(self._engine)

  def drop(self):
    Base.metadata.drop_all(self._engine)

  def close(self):
    self.session.close()

  @classmethod
  def from_local(cls, db_path):
    db = cls(db_path)
    return db


  def get_wikipedia_urls(self):
    """Return a list of all wikipedia urls in this database.

    Returns:
      A list of wikipedia url strings in db
    """
    wiki_rows: List[Tuple[Text]] = self.session.query(
        WikiPage.wikipedia_url).all()
    return [row[0] for row in wiki_rows]

  def get_page(self, wikipedia_url):
    """Get the proto for the wikipedia page by url.

    Args:
      wikipedia_url: Url to lookup

    Returns:
      Returns proto of page if it exists, otherwise None
    """
    page = (
        self.session
        .query(WikiPage)
        .filter_by(wikipedia_url=wikipedia_url)
        .first()
    )
    if page is None:
      return None
    else:
      return fever_pb2.WikipediaDump.FromString(page.proto)

  def get_page_sentence(self, wikipedia_url,
                        sentence_id):
    """Get the text for a sentence on the given wikipedia page.

    Args:
      wikipedia_url: Url to lookup
      sentence_id: The sentence to retrieve

    Returns:
      The text of the sentence if it exists, None otherwise
    """
    page = (
        self.session.query(WikiPage).filter_by(
            wikipedia_url=wikipedia_url).first())
    if page is None:
      return None
    else:
      page_proto = fever_pb2.WikipediaDump.FromString(page.proto)
      if sentence_id in page_proto.sentences:
        return page_proto.sentences[sentence_id].text
      else:
        return None


