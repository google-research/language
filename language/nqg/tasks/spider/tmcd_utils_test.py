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
"""Tests for tmcd_utils."""

from language.nqg.tasks.spider import tmcd_utils

import tensorflow as tf

SQL = ("select count(*) from courses as t1 join student_course_attendance as t2"
       " on t1.course_id = t2.course_id where t1.course_name = 'english'")


class DivergenceUtilsTest(tf.test.TestCase):

  def test_get_atoms_and_compounds(self):
    atoms = tmcd_utils.get_atoms(SQL)
    compounds = tmcd_utils.get_compounds(SQL)
    print(atoms)
    print(compounds)
    self.assertEqual(
        atoms, {
            "t1", "course_id", "as", ")", "from", "join", "(", "=", "on",
            "course_name", "count", "'english'", "courses", "*", ".",
            "student_course_attendance", "where", "t2", "select"
        })
    self.assertEqual(
        compounds.keys(), {
            "___ where ___ = ___", "___ . ___ = ___", "___ . course_id = ___",
            "___ as ___ join ___", "___ on ___ . ___ = ___",
            "___ . course_name = ___", "___ = t2 . ___",
            "select count ( ___ ) from ___ where ___",
            "select count ( ___ ) from ___", "courses as ___",
            "___ . course_name", "t1 . ___ = ___", "___ where ___ . ___ = ___",
            "select ___ from ___ join ___ on ___", "courses as ___ join ___",
            "___ = 'english'", "student_course_attendance as ___",
            "___ . course_id", "___ = ___ . course_id", "___ as ___ on ___",
            "___ = ___ . ___", "student_course_attendance as ___ on ___",
            "___ join ___ on ___ = ___", "___ as t1 join ___",
            "___ join ___ on ___", "___ as t2 on ___", "___ on ___ = ___ . ___",
            "___ where ___ = 'english'",
            "select ___ from ___ join ___ where ___",
            "select ___ from ___ as ___ join ___",
            "select ___ from ___ join ___", "count ( * )", "t2 . ___",
            "___ as t2", "___ as t1", "select ___ from ___ where ___",
            "t1 . ___", "___ on ___ = ___", "select count ( * ) from ___",
            "___ join ___ as ___ on ___"
        })


if __name__ == "__main__":
  tf.test.main()
