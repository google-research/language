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
#!/bin/bash
# This is a demo script to run end-to-end experiment on GeoQuery dataset.

RUN_DIR=/path/to/.../  # CHANGEME

# Download and preprocess data to TSV file.
DATA_DIR=/path/to/data/  # CHANGEME
TRAIN_TSV=${DATA_DIR}/train.tsv
TEST_TSV=${DATA_DIR}/dev.tsv

# Generate exact match rules as seed rules.
# Other tasks might use different preprocessing process, see README for details.
SEED_RULES_FILE=${DATA_DIR}/seed_rules.txt
python -m language.compgen.csl.tasks.generate_exact_match_rules \
  --input ${TRAIN_TSV} \
  --output ${SEED_RULES_FILE}

# Run grammar induction (use Apache Beam version for large datasets).
RULES=${RUN_DIR}/rules.txt
python -m language.compgen.csl.induction.search_main \
  --input ${TRAIN_TSV} \
  --output ${RULES} \
  --config language/compgen/csl/tasks/geoquery/induction_config.json \
  --seed_rules_file ${SEED_RULES_FILE} \
  --target_grammar language/compgen/csl/tasks/geoquery/target_cfg.txt

# Write training data to a TFRecord file (use Apache Beam version for large datasets).
TF_EXAMPLES=${DATA_DIR}/examples.tfrecord
python -m language.compgen.csl.model.data.write_examples \
  --input ${TRAIN_TSV} \
  --output ${TF_EXAMPLES} \
  --config language/compgen/csl/tasks/geoquery/model_config.json \
  --rules ${RULES}

# Run model training.
python -m language.compgen.csl.model.training.train_model \
  --input ${TF_EXAMPLES} \
  --model_dir ${RUN_DIR} \
  --config language/compgen/csl/tasks/geoquery/model_config.json

# Run model evaluation (use Apache Beam version for large datasets).
python -m language.compgen.csl.model.inference.eval_model \
  --input ${TEST_TSV} \
  --model_dir ${RUN_DIR} \
  --config language/compgen/csl/tasks/geoquery/model_config.json \
  --rules ${RULES} \
  --target_grammar language/compgen/csl/tasks/geoquery/target_cfg.txt

# Generate synthetic data (use Apache Beam version for large datasets).
SAMPLED_TSV=${RUN_DIR}/synthetic_examples.tsv
NUM_EXAMPLES=100000
python -m language.compgen.csl.augment.generate_synthetic_examples \
  --augment_config language/compgen/csl/tasks/geoquery/augment_config.json \
  --output ${SAMPLED_TSV} \
  --num_examples ${NUM_EXAMPLES} \
  --rules ${RULES} \
  --target_grammar language/compgen/csl/tasks/geoquery/target_cfg.txt \
  --model_dir ${RUN_DIR} \
  --model_config config language/compgen/csl/tasks/geoquery/model_config.json

# Merge synthetic data with the original training data.
# We up-sample the training data to roughly match the number of synthetic examples.
# The new TSV can be used to train any downstream model.
AUGMENT_TSV=${RUN_DIR}/train_augment.tsv
python -m language.compgen.csl.augment.merge_tsvs \
  --input_1 ${TRAIN_TSV} \
  --input_2 ${SAMPLED_TSV} \
  --output ${AUGMENT_TSV} \
  --duplicate_input_1 227

