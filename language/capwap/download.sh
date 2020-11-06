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
#! /bin/bash
# Script to download the data files for the project.
# This will take awhile.
#
# usage:
#  ./download.sh [(optional) data_dir]

set -e

# Set root directory according to:
# 1 Command line argument.
# 2 CAPWAP environment variable.
# 3 Default `data` in current directory.
if [ -z "$1" ]; then
  if [ -z "${CAPWAP}" ]; then
    ROOT="data"
  else
    ROOT="${CAPWAP}"
  fi
else
  ROOT="${1}"
fi
mkdir -p "${ROOT}"

# gsutil download
function gsutil_download() {
  gsutil cp -r ${1} .
}
GET="gsutil_download"

RELEASE="gs://capwap"

# Helper function to download and unpack a .zip file.
function download_and_unzip() {
  local BASE_URL=${1}
  local FILENAME=${2}
  local DOWNLOAD="${3:-wget -nd -c}"
  local UNZIP="${4:-unzip -nq}"

  if [ ! -f ${FILENAME} ]; then
    echo "Downloading ${FILENAME} to $(pwd)"
    ${DOWNLOAD} "${BASE_URL}/${FILENAME}"
  else
    echo "Skipping download of ${FILENAME}"
  fi
  echo "Unzipping ${FILENAME}"
  ${UNZIP} ${FILENAME}
  rm ${FILENAME}
}

# ------------------------------------------------------------------------------
#
# COCO dataset.
#
# ------------------------------------------------------------------------------

mkdir -p "${ROOT}/COCO"
pushd "${ROOT}/COCO"

BASE_IMAGE_URL="http://msvocds.blob.core.windows.net/coco2014"

TRAIN_IMAGE_FILE="train2014.zip"
download_and_unzip ${BASE_IMAGE_URL} ${TRAIN_IMAGE_FILE}
mv train2014 images

VAL_IMAGE_FILE="val2014.zip"
download_and_unzip ${BASE_IMAGE_URL} ${VAL_IMAGE_FILE}
mv val2014/* images/
rm -r val2014

BASE_CAPTIONS_URL="http://msvocds.blob.core.windows.net/annotations-1-0-3"
CAPTIONS_FILE="captions_train-val2014.zip"
download_and_unzip ${BASE_CAPTIONS_URL} ${CAPTIONS_FILE}

BASE_DATASET_URL="https://cs.stanford.edu/people/karpathy/deepimagesent"
DATASET_FILE="caption_datasets.zip"
download_and_unzip ${BASE_DATASET_URL} ${DATASET_FILE}
mv dataset_coco.json karpathy_splits.json
rm dataset_flickr30k.json dataset_flickr8k.json

FEATURES_FILE="coco.zip"
download_and_unzip ${RELEASE}/features ${FEATURES_FILE} gsutil_download
popd

# ------------------------------------------------------------------------------
#
# VQA dataset.
#
# ------------------------------------------------------------------------------

mkdir -p "${ROOT}/VQA"
pushd "${ROOT}/VQA"

BASE_VQA_URL="https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa"

TRAIN_ANNOTATION_FILE="v2_Annotations_Train_mscoco.zip"
download_and_unzip ${BASE_VQA_URL} ${TRAIN_ANNOTATION_FILE}

TRAIN_QUESTION_FILE="v2_Questions_Train_mscoco.zip"
download_and_unzip ${BASE_VQA_URL} ${TRAIN_QUESTION_FILE}

VAL_ANNOTATION_FILE="v2_Annotations_Val_mscoco.zip"
download_and_unzip ${BASE_VQA_URL} ${VAL_ANNOTATION_FILE}

VAL_QUESTION_FILE="v2_Questions_Val_mscoco.zip"
download_and_unzip ${BASE_VQA_URL} ${VAL_QUESTION_FILE}
popd

# ------------------------------------------------------------------------------
#
# GQA dataset.
#
# ------------------------------------------------------------------------------

mkdir -p "${ROOT}/GQA"
pushd "${ROOT}/GQA"

GQA_BASE_URL="https://nlp.stanford.edu/data/gqa/"
QUESTION_FILE="questions1.2.zip"
download_and_unzip ${GQA_BASE_URL} ${QUESTION_FILE}
rm -r *_all_*
rm *.txt challenge_* test* submission*

FEATURES_FILE="gqa.zip"
download_and_unzip ${RELEASE}/features ${FEATURES_FILE} gsutil_download
gsutil_download ${RELEASE}/capwap_gqa_splits.json
mv capwap_gqa_splits.json capwap_splits.json
popd

# ------------------------------------------------------------------------------
#
# Visual7W dataset.
#
# ------------------------------------------------------------------------------

mkdir -p "${ROOT}/V7W"
pushd "${ROOT}/V7W"

V7W_BASE_URL="http://ai.stanford.edu/~yukez/papers/resources"
V7W_FILE="dataset_v7w_telling.zip"
download_and_unzip ${V7W_BASE_URL} ${V7W_FILE}

GENOME_BASE_URL="https://visualgenome.org/static/data/dataset"
GENOME_FILE="image_data.json.zip"
download_and_unzip ${GENOME_BASE_URL} ${GENOME_FILE}

FEATURES_FILE="v7w.zip"
download_and_unzip ${RELEASE}/features ${FEATURES_FILE} gsutil_download
popd

# ------------------------------------------------------------------------------
#
# VizWiz dataset.
#
# ------------------------------------------------------------------------------

mkdir -p "${ROOT}/VIZWIZ"
pushd "${ROOT}/VIZWIZ"

VIZWIZ_BASE_URL="https://qa2cap.s3.us-east-2.amazonaws.com/data"
# Uncomment to download images for visualization.
# download_and_unzip ${VIZWIZ_BASE_URL} Images.zip
download_and_unzip ${VIZWIZ_BASE_URL} Annotations.zip
gsutil_download ${RELEASE}/capwap_vizwiz_splits.json
mv capwap_vizwiz_splits.json capwap_splits.json

FEATURES_FILE="vizwiz.zip"
download_and_unzip ${RELEASE}/features ${FEATURES_FILE} gsutil_download
popd


# ------------------------------------------------------------------------------
#
# Vocabulary, RC model, and QA generation model.
#
# ------------------------------------------------------------------------------


pushd ${ROOT}
gsutil_download ${RELEASE}/uncased_vocab.txt
download_and_unzip ${RELEASE} rc_model.zip gsutil_download
download_and_unzip ${RELEASE} qgen_model.zip gsutil_download
popd
