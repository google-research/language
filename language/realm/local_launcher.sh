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
# Launcher for running REALM on a single machine.
# See README.md for instructions.

################################################
# Change the data and output directories here:

# Data directory
DATA_DIR='gs://realm-data/realm-data-small'

# Output directory
MODEL_DIR='./out/'

################################################
# Constants

# Pre-training corpus (Glob of gzip TFRecords)
PRETRAIN_CORPUS_PATH="${DATA_DIR}/pretrain_corpus_small/wikipedia_annotated_with_dates_public*.tfrecord.gz"

# Knowledge corpus (Glob of gzip TFRecords)
RETRIEVAL_CORPUS_PATH="${DATA_DIR}/retrieval_corpus_small/docs.small.tfr.gz-*-of-*"

# Initial BERT hub module for the Knowledge-Augmented Encoder p(y | z, x)
BERT_MODULE_PATH='https://tfhub.dev/google/small_bert/bert_uncased_L-2_H-128_A-2/1'

# Initial ICT hub module for the Knowledge Retriever p(z | x)
# Must contain the embeddings of the documents in the knowledge corpus.
ICT_MODULE_PATH="${DATA_DIR}/small-ict"

# BERT vocab
VOCAB_PATH="${ICT_MODULE_PATH}/assets/vocab.txt"

# Port for the training example generator
TRAIN_EX_GENERATOR_PORT=8888

# Port for the evaluation example generator
EVAL_EX_GENERATOR_PORT=8889

# Log directory
LOG_DIR="${MODEL_DIR}/log/"

mkdir -p "${MODEL_DIR}/export/tf_hub"
mkdir -p "${MODEL_DIR}/export/encoded"
mkdir -p "${LOG_DIR}"

################################################
# Invocations

launch_refresh_doc_embeds()
{
  python -m language.realm.refresh_doc_embeds \
    --model_dir="${MODEL_DIR}" \
    --retrieval_corpus_path="${RETRIEVAL_CORPUS_PATH}" \
    --start_at_step=0 \
    --nouse_tpu
}

launch_example_generator()
{
  case "$1" in
    train)
      extra_flags="--port=${TRAIN_EX_GENERATOR_PORT} --is_train"
      ;;
    eval)
      extra_flags="--port=${EVAL_EX_GENERATOR_PORT} --nois_train"
      ;;
    *)
      echo "Invalid argument for launch_example_generator: $1"
      exit 1
      ;;
  esac

  python -m language.realm.example_generator \
    --vocab_path="${VOCAB_PATH}" \
    --do_lower_case \
    --query_seq_len=96 \
    --candidate_seq_len=288 \
    --num_candidates=8 \
    --max_masks=10 \
    --num_shards_per_mips_refresh=1 \
    --pretrain_corpus_path="${PRETRAIN_CORPUS_PATH}" \
    --retrieval_corpus_path="${RETRIEVAL_CORPUS_PATH}" \
    --initial_embedder_module="${ICT_MODULE_PATH}" \
    --model_dir="${MODEL_DIR}" \
    $extra_flags
}

launch_train_realm()
{
  python -m language.realm.train_realm \
    --model_dir="${MODEL_DIR}" \
    --vocab_path="${VOCAB_PATH}" \
    --do_lower_case \
    --query_seq_len=96 \
    --candidate_seq_len=288 \
    --num_candidates=8 \
    --max_masks=10 \
    --bert_hub_module_handle="${BERT_MODULE_PATH}" \
    --embedder_hub_module_handle="${ICT_MODULE_PATH}" \
    --batch_size=4 \
    --eval_batch_size=4 \
    --save_checkpoints_steps=100 \
    --keep_checkpoint_max=10 \
    --eval_throttle_secs=10 \
    --eval_start_delay_secs=10 \
    --num_eval_steps=50 \
    --num_train_steps=1000 \
    --train_preprocessing_servers="localhost:${TRAIN_EX_GENERATOR_PORT}" \
    --eval_preprocessing_servers="localhost:${EVAL_EX_GENERATOR_PORT}" \
    --num_input_threads=4
}

################################################
# Main entry

case "$1" in
  refresh)
    # Run a server that re-encode the documents in the knowledge corpus.
    launch_refresh_doc_embeds 2>&1 | tee -a "${LOG_DIR}/refresh.log"
    ;;
  gen-train)
    # Run a server that generates training examples.
    launch_example_generator train 2>&1 | tee -a "${LOG_DIR}/gen-train.log"
    ;;
  gen-eval)
    # Run a server that generates evaluation examples.
    launch_example_generator eval 2>&1 | tee -a "${LOG_DIR}/gen-eval.log"
    ;;
  train)
    # Run the main training code.
    launch_train_realm 2>&1 | tee -a "${LOG_DIR}/train.log"
    ;;
  all)
    # Run everything in separate processes.
    echo "Starting the servers ..."
    launch_refresh_doc_embeds 2>>"${LOG_DIR}/refresh.log" &
    launch_example_generator train 2>>"${LOG_DIR}/gen-train.log" &
    launch_example_generator eval 2>>"${LOG_DIR}/gen-eval.log" &
    echo "Wait for the servers to be ready ..."
    sleep 20     # Wait for everything else to be ready
    echo "Starting the training code ..."
    launch_train_realm 2>>"${LOG_DIR}/train.log" &
    echo "Done."
    ;;
  *)
    echo "Invalid command: $1"
    echo "Available commands: refresh, gen-train, gen-eval, train, all"
    exit 1
    ;;
esac
