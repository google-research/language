"""Contains flags for training on TPU."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags


FLAGS = flags.FLAGS

flags.DEFINE_bool("use_tpu", False, "Whether to use a TPU for training.")

flags.DEFINE_string("primary", "", "The primary machine to use for TPU training.")

flags.DEFINE_integer("num_tpu_shards", 1, "The number of shards to use during TPU training.")
