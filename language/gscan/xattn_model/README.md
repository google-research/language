# Systematic Generalization on gSCAN: What is Nearly Solved and What is Next?

This directory contains modeling code for the paper "[Systematic Generalization on gSCAN: What is Nearly Solved and What is Next?](https://arxiv.org/abs/2109.12243)" (Linlu Qiu, Hexiang Hu, Bowen Zhang, Peter Shaw, Fei Sha).

### Setup and Prerequisites
```
conda create --name gscan
source activate gscan
pip install tensorflow absl-py clu
```

All python scripts should be run using Python 3 while in the top-level of this repository using -m.

### Data

You need to preprocess the data to TFRecord as follows:

```
python -m language.gscan.xattn_model.dataset.preprocess --data_dir=${DATA_DIR} --split=${SPLIT}
```

The available `SPLIT` includes `compositional_splits`, `spatial_relation_splits`, and `target_length_split`. You can find more information in [TFDS](https://www.tensorflow.org/datasets/catalog/grounded_scan). The preprocessing script will generate TFRecord for the training set and all test sets to `DATA_DIR`. It will also generate `training_input_vocab.txt` and `training_target_vocab.txt` to `DATA_DIR` if they do not exist. If you want to use your own splits, you need to update the `num_examples` specified in `gscan_dataset.py`.

### Configuration

We use [ml_collections](https://github.com/google/ml_collections) for configuration. The default configs for compositional splits and spatial relation splits are in `configs`. You also need to update `data_dir` in the config to the correct `DATA_DIR` that stores your TFRecord. To enable the cross-modal attention, set `config.model.cross_attn = True`. You can also override config using `--config.seed=43` when running your experiments.

### Training

To train the model:

```
python -m language.gscan.xattn_model.main --config=${CONFIG} --workdir=${WORKDIR}
```


### Testing

To test and evaluate the model:

```
python -m language.gscan.xattn_model.main --config=${CONFIG} --workdir=${WORKDIR} --mode=predict --ckpt_path=${CKPT_PATH}
```

The latest checkpoint will be used if `CKPT_PATH` is not specified. You need to put `dataset.txt`, `training_input_vocab.txt` and `training_target_vocab.txt` to `DATA_DIR` for evaluation. The `dataset.txt` can be downloaded for [compositional splits and target lengths split](https://github.com/LauraRuis/groundedSCAN/tree/master/data) or [spatial relation splits](https://storage.googleapis.com/gresearch/gscan/spatial_relation_splits.zip). During preprocessing, the two vocab files will be saved to the `DATA_DIR` if they do not exist. All predictions will be saved to a JSON file in the `${WORK_DIR}/predictions`. The JSON file can be used directly for error analysis provided in the original [gSCAN](https://github.com/LauraRuis/groundedSCAN) repository.

