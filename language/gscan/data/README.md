# Systematic Generalization on gSCAN: What is Nearly Solved and What is Next?

This directory contains data generation code for the paper "[Systematic Generalization on gSCAN: What is Nearly Solved and What is Next?](https://arxiv.org/abs/2109.12243)"
(Linlu Qiu, Hexiang Hu, Bowen Zhang, Peter Shaw, Fei Sha).

### Setup and Prerequisites

```
conda create --name gscan
source activate gscan
pip install tensorflow absl-py
```

You also need to put the `GroundedScan` of the original original [groundedSCAN](https://github.com/LauraRuis/groundedSCAN) in the top-level of this repository and install its requirements.

```
git clone https://github.com/LauraRuis/groundedSCAN.git
pip install -r groundedSCAN/requirements
mv groundedSCAN/GroundedScan .
```

All python scripts should be run using Python 3 while in the top-level of this repository using -m.

### Data Generation

We follow the original data generation process. You can find more information from the Appendix B of our paper and the original gSCAN [documentation](https://github.com/LauraRuis/groundedSCAN#using-the-repository). The spatial relation splits we use for the paper can be loaded from [TFDS](https://www.tensorflow.org/datasets/catalog/grounded_scan). You can also download the raw file [here](https://storage.googleapis.com/gresearch/gscan/spatial_relation_splits.zip). You can reproduce our splits by

```
python -m language.gscan.data.main \
--mode=generate \
--output_directory=${OUTPUT_DIR} \
--split=spatial_relation \
--type_grammar=relation_normal \
--exclude_type_grammar=normal \
--num_resampling 1 \
--max_examples_per_target 2 \
--make_dev_set
```

You need to include `relation` in your `type_grammar` to generate data for spatial relation splits. You can specify `--exclude_type_grammar` if you want to filter out examples that do not include spatial relations between objects.

A `dataset.txt` and data statistics files will be saved to `OUTPUT_DIR`. The `dataset.txt` contains the dataset information and data examples. The format of the data example is the same as gSCAN. You can find detailed information from the [documentation](https://github.com/LauraRuis/multimodal_seq2seq_gSCAN/tree/master/read_gscan) of the original gSCAN paper.

You can also compute data statistics for an existing dataset by

```
python -m language.gscan.data.main \
--mode=gather_stats \
--output_directory=${OUTPUT_DIR} \
--load_dataset_from=${DATASET_FILE}
```

### Visualization

You can visualize the data examples in the end of data generation by specifying `--visualize_per_split` or `--visualize_per_template`. You can also visualize data examples for an existing dataset by

```
python -m language.gscan.data.main \
--mode=visualize \
--output_directory=${OUTPUT_DIR} \
--load_dataset_from=${DATASET_FILE} \
--visualize_per_split=${VISUALIZE_NUM}
```
