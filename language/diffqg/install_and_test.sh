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



LANG_DIR="${1?}"


cd $LANG_DIR
DIFFQG_DIR="${LANG_DIR}/language/diffqg"

mkdir -p ${DIFFQG_DIR}

GOLD_FNAME="gold_annotations.jsonl"
TEST_FNAME="sample_output.jsonl"

# Externally accessible web paths to download the final data from.
URL="https://storage.googleapis.com/gresearch/diffqg"
FINAL_DATA_GOLD="${URL}/${GOLD_FNAME}"
FINAL_DATA_TEST="${URL}/${TEST_FNAME}"


BLEURT_FNAME="BLEURT-20"
PATH_TO_BLEURT="https://storage.googleapis.com/bleurt-oss-21/${BLEURT_FNAME}.zip"
BLEURT_DIR="${DIFFQG_DIR}/bleurt"
BLEURT_ZIP="${BLEURT_DIR}/${BLEURT_FNAME}.zip"
BLEURT_MODEL_DIR="${BLEURT_DIR}/${BLEURT_FNAME}"

DATA_DIR="${DIFFQG_DIR}/data"
GOLD_PATH="${DATA_DIR}/${GOLD_FNAME}"
TEST_PATH="${DATA_DIR}/${TEST_FNAME}"

if test -d "${DIFFQG_DIR}"; then
  echo "Running DiffQG out of ${DIFFQG_DIR}"
else
  echo "No directory found at ${DIFFQG_DIR}. This should point to where you cloned the repo."
  exit
fi
if test -d "${DATA_DIR}"; then
  echo "Reusing data directory ${DATA_DIR}"
else
  echo "Creating data directory ${DATA_DIR}"
  mkdir "${DATA_DIR}" || exit
fi
if test -s "${GOLD_PATH}"; then
  echo "Found gold data downloaded into ${DATA_DIR}, proceeding."
else
  echo "Downloading gold data to ${DATA_DIR}"
  wget $FINAL_DATA_GOLD -P ${DATA_DIR} || exit
fi
if test -s "${TEST_PATH}"; then
  echo "Found test data downloaded into ${DATA_DIR}, proceeding."
else
  echo "Downloading test data to ${DATA_DIR}"
  wget $FINAL_DATA_TEST -P ${DATA_DIR} || exit
fi

if test -d "${BLEURT_DIR}"; then
  echo "Reusing existing bleurt installation dir."
else
  echo "Creating bleurt installation dir."
  mkdir ${BLEURT_DIR} || exit
fi
if test -d "${BLEURT_MODEL_DIR}"; then
  echo "Reusing existing BLEURT installation."
else
  if test -s "${BLEURT_ZIP}"; then
    echo "Reusing existing downloaded BLEURT zip file."
  else
    echo "Downloading BLEURT zip file."
    wget "${PATH_TO_BLEURT}" -P ${BLEURT_DIR} || exit
  fi
  echo "Unpacking BLEURT zip file to ${BLEURT_MODEL_DIR}"
  unzip "${BLEURT_ZIP}" -d ${BLEURT_DIR} || exit
fi

pip install -r "${DIFFQG_DIR}/requirements.txt"

python3 -m language.diffqg.run_metrics \
--gold_annotations=$GOLD_PATH \
--predicted_annotations=$TEST_PATH \
--output_scores=$DIFFQG_DIR/test_scores.txt \
--output_metrics=$DIFFQG_DIR/test_metrics.txt \
--bleurt_checkpoint=$BLEURT_MODEL_DIR \
--run_qsim=True \
--batch_size=4 \
--num_batches=4

