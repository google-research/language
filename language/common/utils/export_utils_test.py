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
"""Tests for export_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl.testing import absltest
from language.common.utils import export_utils


class ExportUtilsTest(absltest.TestCase):

  def test_numeric_timestamp_and_trailing_slashes(self):
    temp_dir = self.create_tempdir().full_path
    os.makedirs(os.path.join(temp_dir, "export", "best", "2", "my_module"))
    os.makedirs(os.path.join(temp_dir, "export", "best", "10", "my_module"))
    result = export_utils.tfhub_export_path(temp_dir + "/", "best", "my_module")
    self.assertEqual(
        result, os.path.join(temp_dir, "export", "best", "10", "my_module"))

  def test_cleanup_old_dirs(self):
    temp_dir = self.create_tempdir().full_path
    for i in range(7, 12):
      os.makedirs(os.path.join(temp_dir, "export", "best", str(i), "my_module"))
    export_utils.clean_tfhub_exports(temp_dir, "best", exports_to_keep=3)
    dirnames = os.listdir(os.path.join(temp_dir, "export", "best"))
    self.assertEqual(set(dirnames), {"9", "10", "11"})


if __name__ == "__main__":
  absltest.main()
