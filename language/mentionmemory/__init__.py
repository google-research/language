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
"""Find and register unittests.

See https://docs.python.org/3/library/unittest.html#load-tests-protocol
for details or
https://github.com/python/cpython/blob/main/Lib/unittest/test/__main__.py
for sample implementation.
"""

import os


def load_tests(loader, standard_tests, unused_pattern):
  """Our tests end in `_test.py`, so need to override the test discovery."""
  this_dir = os.path.dirname(__file__)
  package_tests = loader.discover(start_dir=this_dir, pattern="*_test.py")
  standard_tests.addTests(package_tests)
  return standard_tests
