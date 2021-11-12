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
# Generate retrieval-augmented datasets for domain bootstrapping experiments.
set -e
set -u

[[ -z "${DATA_DIR}" ]] && echo "Error: DATA_DIR must be set." && exit 1

function gen_baseline() {
  # Baseline (no augmentation)
  domain="${1}"
  suffix="${2}"
  formatter_kwargs="${3}"

  # query = O_train
  python -m language.casper.augment.cached_retrieval_to_dataset \
    --alsologtostderr \
    --train_data_paths="${DATA_DIR}/raw/train.except-${domain}.no_exemplars.jsonl" \
    --output_dir="${DATA_DIR}/t5/except-${domain}.noaug${suffix}" \
    --example_converter=query_only \
    --formatter_kwargs="${formatter_kwargs}"
  # query = O_dev
  python -m language.casper.augment.cached_retrieval_to_dataset \
    --alsologtostderr \
    --dev_data_paths="${DATA_DIR}/raw/dev.except-${domain}.no_exemplars.jsonl" \
    --output_dir="${DATA_DIR}/t5/except-${domain}.noaug${suffix}" \
    --example_converter=query_only \
    --formatter_kwargs="${formatter_kwargs}"
  # query = N_support
  python -m language.casper.augment.cached_retrieval_to_dataset \
    --alsologtostderr \
    --train_data_paths="${DATA_DIR}/raw/train.100ex-${domain}.no_exemplars.jsonl" \
    --output_dir="${DATA_DIR}/t5/100ex-${domain}.noaug${suffix}" \
    --example_converter=query_only \
    --converter_kwargs='{"n":20}' \
    --formatter_kwargs="${formatter_kwargs}"
  # query = N_dev
  python -m language.casper.augment.cached_retrieval_to_dataset \
    --alsologtostderr \
    --dev_data_paths="${DATA_DIR}/raw/dev.only-${domain}.no_exemplars.jsonl" \
    --output_dir="${DATA_DIR}/t5/only-${domain}.noaug${suffix}" \
    --example_converter=query_only \
    --formatter_kwargs="${formatter_kwargs}"
}

function gen_augmented() {
  # With augmentation
  domain="${1}"
  suffix="${2}"
  formatter_kwargs="${3}"

  # query = O_train / index = O_train + N_support
  python -m language.casper.augment.cached_retrieval_to_dataset \
    --alsologtostderr \
    --train_data_paths="${DATA_DIR}/raw/train.except-${domain}.use-large.100-shot.jsonl" \
    --index_paths="${DATA_DIR}/raw/train.no_exemplars.jsonl" \
    --output_dir="${DATA_DIR}/t5/except-${domain}.100-shot.samp5x20${suffix}" \
    --example_converter=add_samp \
    --converter_kwargs='{"k":5,"n":20}' \
    --formatter_kwargs="${formatter_kwargs}"
  # query = N_support / index = O_train + N_support
  python -m language.casper.augment.cached_retrieval_to_dataset \
    --alsologtostderr \
    --train_data_paths="${DATA_DIR}/raw/train.100ex-${domain}.use-large.100-shot.jsonl" \
    --index_paths="${DATA_DIR}/raw/train.no_exemplars.jsonl" \
    --output_dir="${DATA_DIR}/t5/100ex-${domain}.100-shot.samp5x20${suffix}" \
    --example_converter=add_samp \
    --converter_kwargs='{"k":5,"n":20}' \
    --formatter_kwargs="${formatter_kwargs}"
  # query = O_train / index = O_train
  python -m language.casper.augment.cached_retrieval_to_dataset \
    --alsologtostderr \
    --train_data_paths="${DATA_DIR}/raw/train.except-${domain}.use-large.0-shot.jsonl" \
    --index_paths="${DATA_DIR}/raw/train.no_exemplars.jsonl" \
    --output_dir="${DATA_DIR}/t5/except-${domain}.0-shot.samp5x20${suffix}" \
    --example_converter=add_samp \
    --converter_kwargs='{"k":5,"n":20}' \
    --formatter_kwargs="${formatter_kwargs}"

  # query = O_dev / index = O_train + N_support
  python -m language.casper.augment.cached_retrieval_to_dataset \
    --alsologtostderr \
    --dev_data_paths="${DATA_DIR}/raw/dev.except-${domain}.use-large.100-shot.jsonl" \
    --index_paths="${DATA_DIR}/raw/train.no_exemplars.jsonl" \
    --output_dir="${DATA_DIR}/t5/except-${domain}.100-shot.top5${suffix}" \
    --example_converter=add_top \
    --converter_kwargs='{"k":5}' \
    --formatter_kwargs="${formatter_kwargs}"
  # query = N_dev / index = O_train + N_support
  python -m language.casper.augment.cached_retrieval_to_dataset \
    --alsologtostderr \
    --dev_data_paths="${DATA_DIR}/raw/dev.only-${domain}.use-large.100-shot.jsonl" \
    --index_paths="${DATA_DIR}/raw/train.no_exemplars.jsonl" \
    --output_dir="${DATA_DIR}/t5/only-${domain}.100-shot.top5${suffix}" \
    --example_converter=add_top \
    --converter_kwargs='{"k":5}' \
    --formatter_kwargs="${formatter_kwargs}"
}

for domain in {alarm,calling,event,messaging,music}; do
  gen_baseline "${domain}" '' '{}'
  gen_augmented "${domain}" '' '{}'
  gen_augmented "${domain}" '.anon' '{"anonymize":true}'
done
