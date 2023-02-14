We provide data preprocessing scripts to help setup dual
encoder experiments. To run fine-tuning and inference follow
the instructions in the `t5x_retrieval` library:

https://github.com/google-research/t5x_retrieval

You can use `write_doc_idx_maps.py` and `convert_examples.py` to
convert examples and documents jsonl files to the indexed format used by the `t5x_retrieval` library.
