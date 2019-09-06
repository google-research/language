# Fusion of Detected Objects in Text for Visual Question Answering

This document describes the reference code for the paper "Fusion of Detected
Objects in Text for Visual Question Answering"
(<https://arxiv.org/abs/1908.05054>). Note the following requires
downloading the uncased BERT-large model from
<https://github.com/google-research/bert>.

## Data Preparation

Once the zipped VCR dataset has been downloaded from
<https://visualcommonsense.com/> to the `data/vcr` directory run the following
commands to preprocess it:

```bash
for SHARD in `seq 0 19`; do
  python -m compute_vcr_features \
    --data_dir=data/vcr \
    --noinclude_rationales \
    --max_seq_length=64 \
    --output_tfrecord=preprocessed-data/vcr \
    --shard=$SHARD
```

## Training and Inference

Training and prediction for the dual encoder model can be run as:

```bash
python -m run_dual_encoder \
  --bert_config_file=uncased_L-24_H-1024_A-16/bert_config.json \
  --do_predict \
  --do_train \
  --init_checkpoint=uncased_L-24_H-1024_A-16/bert_model.ckpt \
  --iterations_per_loop=1000 \
  --learning_rate=2e-5 \
  --max_seq_length=64 \
  --num_train_epochs=5 \
  --output_dir=output/seed-0.lr-2e-5.epochs-5 \
  --output_pred_file=predicted-tfrecords \
  --predict_precomputed_file="preprocessed-data/vcr-val-*" \
  --random_seed=0 \
  --save_checkpoints_steps=5000 \
  --train_num_precomputed=851760 \
  --train_precomputed_file="preprocessed-data/vcr-train-*" \
  --notrainable_resnet \
  --use_bboxes \
  --warmup_proportion=0.1
```

Training and prediction for the full B2T2 model can be run as:

```bash
python -m run_b2t2 \
  --bert_config_file=uncased_L-24_H-1024_A-16/bert_config.json \
  --do_predict \
  --do_train \
  --init_checkpoint=uncased_L-24_H-1024_A-16/bert_model.ckpt \
  --iterations_per_loop=1000 \
  --learning_rate=2e-5 \
  --max_seq_length=64 \
  --num_train_epochs=5 \
  --output_dir=output/seed-0.lr-2e-5.epochs-5 \
  --output_pred_file=predicted-tfrecords \
  --predict_precomputed_file="preprocessed-data/vcr-val-*" \
  --random_seed=0 \
  --save_checkpoints_steps=5000 \
  --train_num_precomputed=851760 \
  --train_precomputed_file="preprocessed-data/vcr-train-*" \
  --notrainable_resnet \
  --warmup_proportion=0.1
```
