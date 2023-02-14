These scripts expect that systems have produced predictions
following the same jsonl format of the original examples files. Only the `query` and `docs` fields need to be populated
for predictions.

Use `run_eval.py` to compute average precision, recall, and F1.

To analyze the average recall and MRecall of a candidate set produced by a retriever prior to thresholding or classifying candidates to produce a final set, use `analyze_retriever.py`.
