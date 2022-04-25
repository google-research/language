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
"""Tests for abstract_sql."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest

from language.xsp.data_preprocessing import abstract_sql

# Choose a test query that exhibits:
# - A nested expression.
# - Table joins.
# - Table referenced only in FROM clause.
# - Column references outside of SELECT clause.
TEST_QUERY = ("SELECT T1.name FROM user_profiles AS T1 JOIN follows AS T2 ON "
              "T1.uid  =  T2.f1 GROUP BY T2.f1 HAVING count(*)  >  (SELECT "
              "count(*) FROM user_profiles AS T1 JOIN follows AS T2 ON T1.uid  "
              "=  T2.f1 WHERE T1.name  =  'Tyler Swift');")


class AbstractSqlTest(absltest.TestCase):

  def test_sql_to_sql_spans(self):
    sql_spans = abstract_sql.sql_to_sql_spans(TEST_QUERY)
    expected_sql = ("select user_profiles.name from user_profiles join follows "
                    "on user_profiles.uid = follows.f1 group by follows.f1 "
                    "having count ( * ) > ( select count ( * ) from "
                    "user_profiles join follows on user_profiles.uid = "
                    "follows.f1 where user_profiles.name = 'tyler swift' )")
    self.assertEqual(expected_sql, abstract_sql.sql_spans_to_string(sql_spans))

  def test_remove_from_clause(self):
    sql_spans = abstract_sql.sql_to_sql_spans(TEST_QUERY)
    replaced_spans = abstract_sql.replace_from_clause(sql_spans)
    expected_sql = ("select user_profiles.name <from_clause_placeholder> group "
                    "by follows.f1 having count ( * ) > ( select count ( * ) "
                    "<from_clause_placeholder> follows where user_profiles.name"
                    " = 'tyler swift' )")
    self.assertEqual(expected_sql,
                     abstract_sql.sql_spans_to_string(replaced_spans))

  def test_restore_from_clause(self):
    fk_relations = [
        abstract_sql.ForeignKeyRelation("user_profiles", "follows", "uid", "f1")
    ]
    sql_spans = abstract_sql.sql_to_sql_spans(TEST_QUERY)
    replaced_spans = abstract_sql.replace_from_clause(sql_spans)
    restored_spans = abstract_sql.restore_from_clause(replaced_spans,
                                                      fk_relations)
    expected_sql = ("select user_profiles.name from follows join user_profiles "
                    "on follows.f1 = user_profiles.uid group by follows.f1 "
                    "having count ( * ) > ( select count ( * ) from follows "
                    "join user_profiles on follows.f1 = user_profiles.uid where"
                    " user_profiles.name = 'tyler swift' )")
    self.assertEqual(expected_sql,
                     abstract_sql.sql_spans_to_string(restored_spans))

  def test_nested_sql_with_unqualified_column(self):
    query = ("SELECT count(*) FROM enzyme WHERE id NOT IN ( SELECT enzyme_id "
             "FROM medicine_enzyme_interaction )")
    sql_spans = abstract_sql.sql_to_sql_spans(query)
    replaced_spans = abstract_sql.replace_from_clause(sql_spans)
    restored_spans = abstract_sql.restore_from_clause(
        replaced_spans, fk_relations=[])
    expected_sql = (
        "select count ( * ) from enzyme where enzyme.id not in ( select "
        "medicine_enzyme_interaction.enzyme_id from "
        "medicine_enzyme_interaction )")
    self.assertEqual(expected_sql,
                     abstract_sql.sql_spans_to_string(restored_spans))

  def test_union_clause(self):
    query = ("SELECT student_id FROM student_course_registrations UNION SELECT "
             "student_id FROM student_course_attendance")
    sql_spans = abstract_sql.sql_to_sql_spans(query)
    replaced_spans = abstract_sql.replace_from_clause(sql_spans)
    restored_spans = abstract_sql.restore_from_clause(
        replaced_spans, fk_relations=[])
    expected_sql = (
        "select student_course_registrations.student_id from "
        "student_course_registrations union select "
        "student_course_attendance.student_id from student_course_attendance")
    self.assertEqual(expected_sql,
                     abstract_sql.sql_spans_to_string(restored_spans))

  def test_parse_order_by(self):
    query = "SELECT Total_Horses FROM farm ORDER BY Total_Horses ASC"
    sql_spans = abstract_sql.sql_to_sql_spans(query)
    sql_spans_string = abstract_sql.sql_spans_to_string(sql_spans, sep=",")
    # Ensure query is split correctly.
    expected_sql_spans_string = (
        "select,farm.total_horses,from,farm,order by,farm.total_horses,asc")
    self.assertEqual(expected_sql_spans_string, sql_spans_string)

  def test_nested_query(self):
    query = ("SELECT count(*) FROM (SELECT * FROM endowment WHERE amount  >  "
             "8.5 GROUP BY school_id HAVING count(*)  >  1)")
    sql_spans = abstract_sql.sql_to_sql_spans(query)
    replaced_spans = abstract_sql.replace_from_clause(sql_spans)

    expected_replaced_sql = (
        "select count ( * ) <from_clause_placeholder> ( select * "
        "<from_clause_placeholder> where endowment.amount > 8.5 group by "
        "endowment.school_id having count ( * ) > 1 )")
    self.assertEqual(expected_replaced_sql,
                     abstract_sql.sql_spans_to_string(replaced_spans))

    restored_spans = abstract_sql.restore_from_clause(
        replaced_spans, fk_relations=[])
    expected_restored_sql = (
        "select count ( * ) from ( select * from endowment where "
        "endowment.amount > 8.5 group by endowment.school_id having count ( * "
        ") > 1 )")
    self.assertEqual(expected_restored_sql,
                     abstract_sql.sql_spans_to_string(restored_spans))

  def test_restore_from_string(self):
    sql_spans = abstract_sql.sql_to_sql_spans(TEST_QUERY)
    replaced_spans = abstract_sql.replace_from_clause(sql_spans)
    replaced_sql = abstract_sql.sql_spans_to_string(replaced_spans)
    parsed_spans = abstract_sql.sql_to_sql_spans(replaced_sql)
    fk_relations = [
        abstract_sql.ForeignKeyRelation("user_profiles", "follows", "uid", "f1")
    ]
    restored_spans = abstract_sql.restore_from_clause(
        parsed_spans, fk_relations=fk_relations)
    restored_sql = abstract_sql.sql_spans_to_string(restored_spans)
    expected_restored_sql = (
        "select user_profiles.name from follows join user_profiles on "
        "follows.f1 = user_profiles.uid group by follows.f1 having count ( * )"
        " > ( select count ( * ) from follows join user_profiles on follows.f1"
        " = user_profiles.uid where user_profiles.name = 'tyler swift' )")
    self.assertEqual(expected_restored_sql, restored_sql)

  def test_restore_from_string_no_tables(self):
    sql_spans = abstract_sql.sql_to_sql_spans(
        "SELECT foo, count(*) FROM bar WHERE id = 5")
    replaced_spans = abstract_sql.replace_from_clause(sql_spans)
    replaced_sql = abstract_sql.sql_spans_to_string(replaced_spans)
    expected_replaced_sql = (
        "select bar.foo , count ( * ) <from_clause_placeholder> where bar.id = 5"
    )
    self.assertEqual(expected_replaced_sql, replaced_sql)
    parsed_spans = abstract_sql.sql_to_sql_spans(replaced_sql)
    fk_relations = []
    restored_spans = abstract_sql.restore_from_clause(
        parsed_spans, fk_relations=fk_relations)
    restored_sql = abstract_sql.sql_spans_to_string(restored_spans)
    expected_restored_sql = (
        "select bar.foo , count ( * ) from bar where bar.id = 5")
    self.assertEqual(expected_restored_sql, restored_sql)

  def test_parse_asql_tables(self):
    query = "SELECT table1.foo <from_clause_placeholder> table1 table2"
    sql_spans = abstract_sql.sql_to_sql_spans(query)
    sql_spans_string = abstract_sql.sql_spans_to_string(sql_spans, sep=",")
    # Ensure query is split correctly.
    expected_sql_spans_string = (
        "select,table1.foo,<from_clause_placeholder>,table1,table2")
    self.assertEqual(expected_sql_spans_string, sql_spans_string)

  def test_find_table(self):
    tables = [
        "author", "domain", "domain_author", "organization", "publication",
        "writes"
    ]
    relations = [
        abstract_sql.ForeignKeyRelation(
            child_table="publication",
            parent_table="writes",
            child_column="pid",
            parent_column="pid"),
        abstract_sql.ForeignKeyRelation(
            child_table="author",
            parent_table="writes",
            child_column="aid",
            parent_column="aid"),
        abstract_sql.ForeignKeyRelation(
            child_table="author",
            parent_table="organization",
            child_column="oid",
            parent_column="oid"),
        abstract_sql.ForeignKeyRelation(
            child_table="author",
            parent_table="domain_author",
            child_column="aid",
            parent_column="aid"),
        abstract_sql.ForeignKeyRelation(
            child_table="domain",
            parent_table="domain_author",
            child_column="did",
            parent_column="did"),
    ]
    from_clause_spans = abstract_sql._get_from_clause_for_tables(
        tables, relations)
    from_clause = abstract_sql.sql_spans_to_string(from_clause_spans, sep=" ")
    expected_from_clause = ("author join domain_author on author.aid = "
                            "domain_author.aid join domain on domain_author.did"
                            " = domain.did join organization on domain.oid = "
                            "organization.oid join writes on organization.aid ="
                            " writes.aid join publication on writes.pid = "
                            "publication.pid")
    self.assertEqual(expected_from_clause, from_clause)

  def test_order_by(self):
    original = "SELECT title FROM course ORDER BY title ,  credits"
    sql_spans = abstract_sql.sql_to_sql_spans(original)
    replaced_spans = abstract_sql.replace_from_clause(sql_spans)
    replaced = abstract_sql.sql_spans_to_string(replaced_spans, sep=" ")
    expected = ("select course.title <from_clause_placeholder> order by "
                "course.title , course.credits")
    self.assertEqual(expected, replaced)

  def test_inner_join(self):
    original = "SELECT f.name FROM foo AS f INNER JOIN bar AS b"
    sql_spans = abstract_sql.sql_to_sql_spans(original)
    replaced_spans = abstract_sql.replace_from_clause(sql_spans)
    replaced = abstract_sql.sql_spans_to_string(replaced_spans, sep=" ")
    expected = ("select foo.name <from_clause_placeholder> bar")
    self.assertEqual(expected, replaced)


if __name__ == "__main__":
  absltest.main()
