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
"""Sqlite scrape db that stores protos by fever claim_id."""
import gzip
import os


from absl import logging
from language.serene import retrieval_pb2
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


class Scrape(Base):
  __tablename__ = 'scrape'
  id = schema.Column(types.Integer, primary_key=True)
  claim_id = schema.Column(types.Integer, index=True)
  proto = schema.Column(types.LargeBinary)  # Gzip compressed proto binary


class ScrapeDatabase:
  """A wrapper around sqlite scrapes database that returns proto pages."""

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


  def __getitem__(
      self,
      claim_id):
    """Get the proto for the wikipedia page by url.

    Args:
      claim_id: The fever claim_id to get evidence for

    Returns:
      Returns proto of page if it exists, otherwise None
    """
    claim_id = int(claim_id)
    result = (self.session.query(Scrape).filter_by(claim_id=claim_id).first())
    if result is None:
      return None
    else:
      return retrieval_pb2.GetDocumentsResponse.FromString(
          gzip.decompress(result.proto))

  def __contains__(self, claim_id):
    claim_id = int(claim_id)
    maybe_scrape = self.session.query(
        Scrape.claim_id).filter_by(claim_id=claim_id).scalar()
    return maybe_scrape is not None


