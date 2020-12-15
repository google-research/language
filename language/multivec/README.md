# Sparse, Dense, and Attentional Representations for Text Retrieval

Yi Luan (Google Research), Jacob Eisenstein (Google Research), Kristina Toutanova (Google Research), and Michael Collins (Google Research).

## Abstract

Dual encoders perform retrieval by encoding documents and queries into dense lowdimensional vectors, scoring each document by its inner product with the query. We investigate the capacity of this architecture relative to sparse bag-of-words models and attentional neural networks. Using both theoretical and empirical analysis, we establish connections between the encoding dimension, the margin between gold and lower-ranked documents, and the document length, suggesting limitations in the capacity of fixed-length encodings to support precise retrieval of long documents. Building on these insights, we propose a simple neural model that combines the efficiency of dual encoders with some of the expressiveness of more costly attentional architectures, and explore sparse-dense hybrids to capitalize on the precision of sparse retrieval. These models outperform strong alternatives in large-scale retrieval.

In this repostory we release the implimentation of ME-BERT, a multi-vector encoding model,
which is computationally feasible for retrieval like
the dual-encoder architecture and achieves significantly better quality.

## Installation

`ME-BERT` is compatible with Python 3, and a Unix-like system. It requires Tensorflow v1.15 to run [BERT-LARGE](https://github.com/google-research/bert), and Tensorflow v2.3.0 to run [ScaNN](https://github.com/google-research/google-research/tree/master/scann)

You must download resources for training the models (e.g., a pre-trained BERT model). Clone the [official BERT repository](https://github.com/google-research/bert) and download the [BERT-Large, uncased model with whole word masking](https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip) to ${BERT_LARGE_DIR}.

For inference and retrieval, run:

```
pip install -r requirements.txt
```

## Data Download

To download all of the datasets used in the paper, run:

```bash
DATA_DIR=<LOCAL_DIR>
./utils/download.sh
python -m language.multivec.utils.convert_tsv_to_json --data_dir=$DATA_DIR
```

Note: this requires [gsutil](https://cloud.google.com/storage/docs/gsutil).

## Instructions

### STEP-1: Data creation

Start with preparing lists of top 100 or 1000 candidates for each training set
and development set query using an initial efficient retrieval model. To use the
official BM25 Anserini baseline from MS MARCO, follow [anserini script](https://github.com/castorini/pyserini/blob/master/docs/experiments-msmarco-passage.md) to calculate top candidates and save files to DATA_DIR/neighbors_train.json and DATA_DIR/neighbors_dev.json.
Note you need to covert the BM25 output to json format, refer to DATA_DIR/neighbors_example.dev.json and language.multivec.utils.convert_tsv_to_json.convert_neighbor_to_json for json format used in this project to save neighbors file.
To create the training examples using candidates from the top 1000 and random negatives, run:

``` bash
python -m language.multivec.preprocessing.create_training_data \
  --neighbors_path=${DATA_DIR}/neighbors_train.json \
  --fraction_dev=0 \
  --fraction_test=0 \
  --max_neighbors=1000 \
  --factored_model \
  --add_random \
  --queries_path=${DATA_DIR}/queries_train.json \
  --passages_path=${DATA_DIR}/passages.json \
  --answers_path=${DATA_DIR}/answers.json \
  --examples_path=${DATA_DIR}/tfexamples
```
Each training queries is paired with 20 candidates, including positives. Half of the rest of the candidates are hard negatives randomly sampled from top 1000 candidates. The other half are randomly sampled from the whole paragraph collection.

To create the development set examples used for hyper-parameter selection using candidates from the top 100 and random negatives, run:

```bash
python -m language.multivec.preprocessing.create_training_data \
  --neighbors_path=${DATA_DIR}/neighbors_dev.json \
  --fraction_dev=1 \
  --fraction_test=0 \
  --max_neighbors=100 \
  --factored_model \
  --add_random \
  --queries_path=${DATA_DIR}/queries_dev.json \
  --passages_path=${DATA_DIR}/passages.json \
  --answers_path=${DATA_DIR}/answers.json \
  --examples_path=${DATA_DIR}/tfexamples
```

### STEP-2: Training
To exactly reproduce the results in the paper, we recommend to run training on v3-32 TPU. The code also supports CPU and GPU, as long as there is enough memory. The following offers training script for M=3. For regular dual encoder (M=1), set num_vec_passage=1

```bash
MODEL_BASE_DIR=<LOCAL_OR_GS_MODEL_DIR>
python -m language.multivec.models.ranking_model_experiment_inbatch \
       --output_base_dir=$MODEL_BASE_DIR \
       --train_batch_size=32 \
       --num_vec_passage=3 \
       --layer_norm \
       --num_train_epochs=10 \
       --use_tpu \
       --max_seq_length=260 \
       --max_seq_length_query=30 \
       --max_seq_length_query=30 \
       --predict_batch_size=32 \
       --data_dir=$DATA_DIR/tfexamples/ \
       --bert_config_file=${BERT_LARGE_DIR}/bert_config.json \
       --init_checkpoint=${BERT_LARGE_DIR}/bert_model.ckpt \
       --eval_name=eval \
       --nodo_eval \
       --do_train \
       --num_candidates=20 \
       --mode=train \
       --num_tpu_cores=$TPU_NUM_CORES \
       --tpu_name=$TPU_NAME \
       --tpu_zone=$TPU_ZONE
```
Continous eval results (with the parameter above) are in

```
${MODEL_BASE_DIR}/msl260_nl24_ah16_hs1024_lr5e-06_warmup0.10_bs32_ne10_dropout0.9_proj0_layernorm_q1_d3/eval_${GLOBAL_STEP}_metrics.tsv
```

The best checkpoint is copied to

```bash
BEST_CKPT=${MODEL_BASE_DIR}/msl260_nl24_ah16_hs1024_lr5e-06_warmup0.10_bs32_ne10_dropout0.9_proj0_layernorm_q1_d3/evalbest_checkpoint_mrr/best_checkpoint
```

### STEP-3: Convert best checkpoint to tfhub

```bash
HUB_DIR=<LOCAL_OR_GS_MODEL_DIR>
python -m language.multivec.models.export_to_tfhub \
  --bert_directory=$BERT_LARGE_DIR \
  --checkpoint_path=$BEST_CKPT \
  --export_path=$HUB_DIR \
  --layer_norm \
  --num_vec_passage=3
```

### STEP-4: Generate data for query & document encoding
The code takes PASSAGE JSON file with the format of {PASSAGE_ID:PASSAGE_TEXT} as input, and outputs a TFRecords file where each entry is the tokenized text IDs converted from the passage text.

```bash
PASSAGE_JSON_PATH=${DATA_DIR}/passages.json
PASSAGE_TFR_PATH=${DATA_DIR}/passage.tfr
python -m language.multivec.utils.data_processor \
  --bert_hub_module_path=$HUB_DIR \
  --max_seq_length=230 \
  --input_pattern=$PASSAGE_JSON_PATH \
  --output_path=$PASSAGE_TFR_PATH \
  --num_threads=12
```

```bash
QUERY_JSON_PATH=${DATA_DIR}/queries_dev.json # change to queries_train.json to generate hard negative training data.
QUERY_TFR_PATH=${DATA_DIR}/query.tfr
python -m language.multivec.utils.data_processor \
  --bert_hub_module_path=$HUB_DIR \
  --max_seq_length=30 \
  --input_pattern=$QUERY_JSON_PATH \
  --output_path=$QUERY_TFR_PATH \
  --num_threads=12
```

### STEP-5: Encode passages & queries
Works with CPU, GPU and TPU. If use TPU inference, we recommend to use v3-8. Also upload passage and query tfr files to gs directory (use gsutil cp command) since TPU can't read from local files.

```bash
INFERENCE_OUTPUT_DIR=<LOCAL_DIR>
# Need to be a local directory since h5py format is incompatible with gs directory.

python -m language.multivec.predict.encode_blocks \
  --noencode_query \
  --batch_size=32 \
  --num_threads=8 \
  --retriever_module_path=$HUB_DIR \
  --output_path=$INFERENCE_OUTPUT_DIR \
  --block_seq_len=230 \
  --examples_path=$PASSAGE_TFR_PATH \
  --num_vec_per_block=3 \
  --tpu_name=$TPU_NAME \
  --tpu_zone=$TPU_ZONE \
  --num_tpu_cores=$NUM_TPU_CORES \
  --use_tpu

# NUM_QUERIES=6980 for MSMARCO dev set
python -m language.multivec.predict.encode_blocks \
  --encode_query \
  --batch_size=32 \
  --num_threads=8 \
  --retriever_module_path=$HUB_DIR \
  --output_path=$INFERENCE_OUTPUT_DIR \
  --block_seq_len=30 \
  --num_blocks=$NUM_QUERIES \
  --examples_path=$QUERY_TFR_PATH \
  --num_vec_per_block=1 \
  --tpu_name=$TPU_NAME \
  --tpu_zone=$TPU_ZONE \
  --suffix=dev \
  --num_tpu_cores=$NUM_TPU_CORES \
  --use_tpu
```

### STEP-6: Run retrieval
Retrieval for evaluation

```bash
python -m language.multivec.predict.retrieval \
  --output_path=$INFERENCE_OUTPUT_DIR \
  --input_path=$INFERENCE_OUTPUT_DIR \
  --suffix=dev \
  --num_vec_per_passage=3 \
  --write_tsv \
  --write_json \
  --brute_force
```
To run approximate search, set brute_force=False adjust approximate search parameters (num_leaves and num_leaves_to_search) accordingly.

### STEP-7: Run MSMARCO evaluation
Download MSMARCO evaluation [script](https://github.com/spacemanidol/MSMARCO/blob/master/Ranking/Baselines/msmarco_eval.py).
Then run:

```bash
python msmarco_eval.py $GOLD_LABELS $INFERENCE_OUTPUT_DIR/neighbors_dev.tsv
```

## Hard negative mining
From our experiments, we noticed doing one round of hard negative mining can significantly improve performance on MSMARCO dev set:

|| MRR@10 | Recall@10 | Recall@100 |
|---|---|---|---|
|ME-BERT| 0.281 | 0.534 | 0.814
|ME-BERT + hard negative mining| 0.334 | 0.608 | 0.863
|Hybrid + ME-BERT + hard negative mining| 0.343 | 0.615 | 0.869

In order to run hard negative mining, change $QUERY_JSON_PATH to MSMARCO training set (${DATA_DIR}/queries_train.json) and change corresponding output paths.
Follow the instruction above,
run STEP-4 and STEP-5 to generate encodings for training set queries.
Check whether queries_train_encodings.h5py is in $INFERENCE_OUTPUT_DIR to make sure inference finished successfully.

Then run retrieval for training queries:

```bash
python -m language.multivec.predict.encode_blocks \
  --encode_query \
  --batch_size=32 \
  --num_threads=8 \
  --retriever_module_path=$HUB_DIR \
  --output_path=$INFERENCE_OUTPUT_DIR \
  --block_seq_len=30 \
  --num_blocks=$NUM_QUERIES \
  --examples_path=$QUERY_TFR_PATH \
  --num_vec_per_block=1 \
  --tpu_name=$TPU_NAME \
  --tpu_zone=$TPU_ZONE \
  --suffix=train \
  --num_tpu_cores=$NUM_TPU_CORES \
  --use_tpu

python -m language.multivec.predict.retrieval \
  --output_path=$INFERENCE_OUTPUT_DIR \
  --input_path=$INFERENCE_OUTPUT_DIR \
  --suffix=train \
  --num_vec_per_passage=3 \
  --write_json \
  --brute_force
```

Create new training data following STEP-1 using the generated neighbors in ${INFERENCE_OUTPUT_TRAIN_DIR}/neighbors.json.
Repeat the whole training process from STEP-1 to STEP-7 to get the final results.

## Trained Models
To run inference only, set $BEST_CKPT to the following and run Step 3-7.

```bash
BEST_CKPT=gs://multivec/msmarco/ckpts/q1d3/best_checkpoint
```

## Citation

If you use this in your work please cite:

```
@article{luan2020sparse,
   title={Sparse, Dense, and Attentional Representations for Text Retrieval},
   journal={Transactions of the Association for Computational Linguistics},
   publisher={MIT Press - Journals},
   author={Yi Luan and Jacob Eisenstein and Kristina Toutanova and Michael Collins},
   year={2021},
}
```
