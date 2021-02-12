## Decontextualization

This repository contains a data description, evaluation code, and utilities for
[*Decontextualization: Making Sentences Stand-Alone*](https://arxiv.org/abs/2102.05169).

The training and evaluation data can be downloaded from the following links:

* [train.jsonl](https://storage.cloud.google.com/decontext_dataset/decontext_train.jsonl):  11290 labeled training examples,
* [dev.jsonl](https://storage.cloud.google.com/decontext_dataset/decontext_dev.jsonl):  1945 labeled development examples,
* [test.jsonl](https://storage.cloud.google.com/decontext_dataset/decontext_test.jsonl):  1945 labeled test examples.

The data format and instructions for use are described below.

### Dataset

The decontextualization dataset contains triplets containing (example_id,
sentence, context, decontextualized sentence).

A context consists of a Wikipedia page url (article_url), page title
(page_title), and a list of section headers, which can define a nested structure
(section_title).

The sentence to be decontextualized will be recovered from byte offsets
(sentence_start_byte_offset, sentence_end_byte_offset) indexed to paragraph text
(paragraph_text).


#### Format

The decontextualization dataset release consists of three .jsonl files

(train.jsonl, dev.jsonl, test.jsonl) where each line is a JSON dictionary.

with the following format:

```bash
  {
    example_id: 3245252,
    original_sentence: "It is named after
      the Roman goddess of love and beauty",
    page_title: "Venus (planet)",
    seciton_title: ["Construction"],
    paragraph_text: "Venus is the second planet from the Sun. It is named after
      the Roman goddess of love and beauty. As the second-brightest natural
      object in the night sky after the Moon, Venus can cast shadows and,
      rarely, is visible to the naked eye in broad daylight. Venus lies within
      Earth's orbit, and so never appears to venture far from the Sun, either
      setting in the west just after dusk or rising in the east a
      bit before dawn.",
    sentence_start_byte_offset: 10,  # offset inclusive
    sentence_end_byte_offset: 20,  # offset inclusive
    article_url: "https://en.wikipedia.org/wiki/Venus",
    annotations: [a list of annotations, as defined below]
  }
```

  A single annotation can be one of the following form:

  (1) Finely-annotated (dev.jsonl and test.jsonl):

```bash
  {"original_sentence": "It is named after the Roman goddess of love and beauty.",
   "decontextualized_sentence": "Venus is named after the Roman goddess of love and beauty.",
   "category": "DONE",
   "changes": [{"new": "Venus", "old": {"start_ind": 0,"end_ind": 2}]}
```

  (2) Coarsely-annotated (train.jsonl):
  {"decontextualized_sentence": "", "category": "DONE"}


### Paper
More details are available in our paper, which can be cited as follows.

```
@inproceedings{Choi2020:Decontext,
  title =     {Making Sentences Stand-Alone: A Task Definition, an Annotated Corpus, and an Empirical Evaluation},
  author =    {Eunsol Choi, Jennimaria Palomaki, Matthew Lamm, Dipanjan Das, Tom Kwiatkowski, Michael Collins},
  booktitle = {Transactions of the Association of Computational Linguistics},
  year =      {2021},
}
```

### Evaluation

#### Prerequisites
The code requires python 3 and a few libraries.
You can use your favorite python environment manager (e.g., virtualenv or conda)
to install the requirements listed in eval_requirements.txt.

```
pip3 install -r language/decontext/eval_requirements.txt
```

#### Running the evaluation script

After installing the pre-requisite, you can run evaluation by

```
python3 -m language.decontext.eval \
  --annotations=decontext_dev.jsonl \
  --predictions=your_model_predictions.jsonl
```

### License
This dataset is released under the Creative Commons Share-Alike 3.0 license.

### Citation
Please cite as:
```
@article{choi2021making,
  title = {Decontextualization: Making Sentences Stand-Alone},
  author = {Eunsol Choi and Jennimaria Palomaki and Matthew Lamm and Tom Kwiatkowski and Dipanjan Das and Michael Collins},
  year = {2021},
  journal = {Transactions of the Association of Computational Linguistics},
}
```
