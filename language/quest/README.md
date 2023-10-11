# QUEST: A Retrieval Dataset of Entity-Seeking Queries with Implicit Set Operations

Paper link: https://arxiv.org/abs/2305.11694

# Dataset

## Examples

QUEST contains 6307 training queries, 323 examples for development, and 1727
examples for testing. These examples are available at:

* https://storage.googleapis.com/gresearch/quest/train.jsonl
* https://storage.googleapis.com/gresearch/quest/train_aug.jsonl (includes augmented data used for paper experiments)
* https://storage.googleapis.com/gresearch/quest/val.jsonl
* https://storage.googleapis.com/gresearch/quest/test.jsonl

We also provide files which include the data for the books and films domains
 before filtering documents based on relevance labels:

* https://storage.googleapis.com/gresearch/quest/prefiltered_films.jsonl
* https://storage.googleapis.com/gresearch/quest/prefiltered_books.jsonl

Each examples file contains newline-separated json dictionaries with the following fields:

* `query` - Paraphrased query written by annotators.
* `docs` - List of relevant document titles.
* `original_query` - The original query which was paraphrased. Atomic queries are
  enclosed by `<mark></mark>`. Augmented queries do not have this field populated.
* `scores` - This field is not populated and only used when producing predictions to enable sharing the same data structure.
* `metadata` - A dictionary with the following fields:
    * `template` - The template used to create the query.
    * `domain` - The domain to which the query belongs.
    * `fluency` - List of fluency ratings for the query.
    * `meaning` - List of ratings for whether the paraphrased query meaning is the
      same as the original query.
    * `naturalness` - List of naturalness ratings for the query.
    * `relevance_ratings` - Dictionary mapping document titles to relevance ratings
      for the document.
    * `evidence_ratings` - Dictionary mapping document titles to evidence ratings
      for the document.
    * `attributions` - Dictionary mapping a document title to its attributions
      attributions are a list of dictionaries mapping a query substring to a
      document substring.

## Document Corpus

The document corpus is at https://storage.googleapis.com/gresearch/quest/documents.jsonl. Note that this file is quite large
(899MB). The format is newline separated json dicts containing `title` and
`text`.

# Evaluation

Scripts and documentation for running evaluations are in the `eval/` directory.

# Baselines

Code and documentation for various baseline systems are in the following sub-directories:

* `bm25/` - BM25 retriever.
* `t5xr/` - Dual encoder based on `t5x_retrieval` library.
* `xattn/` - T5-based cross-attention classifier.

