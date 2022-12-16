# MASSIVE translations

We are releasing machine translations of the English utterances from the
training split in [MASSIVE](https://arxiv.org/abs/2204.08582), a multilingual
semantic parsing dataset. We translate such utterances to all the 50 non-English
languages in MASSIVE.

The translations were used to create silver semantic parsing data in multiple
languages starting from English annotated data with our
[Translate-and-Fill (TaF)](https://arxiv.org/abs/2109.04319) approach.

TaF has been a key part in our winning submission to the zero-shot task of the
[MMNLU-22 Multilingual Semantic Parsing competition](https://mmnlu-22.github.io/)
organized by Amazon.

## Data Location

You can download the [MASSIVE translations](https://storage.googleapis.com/massive-translations/massive_translations.zip)
with the following command:

```
wget https://storage.googleapis.com/massive-translations/massive_translations.zip
```

## Data Format

The translations are available in TSV (Tab-separated) files, each language in
its own file. The file columns are: id, english_utterance, translation.

## Citation

If you use this data please cite:

```
@inproceedings{nicosia-piccinno-22,
    title = "Evaluating Byte and Wordpiece Level Models for Massively Multilingual Semantic Parsing",
    author = "Nicosia, Massimo and Piccinno, Francesco",
    booktitle = "In Proceedings of the Massively Multilingual Natural Language Understanding Workshop (MMNLU-22 @ EMNLP 2022)",
    month = dec,
    year = "2022",
    publisher = "Association for Computational Linguistics",
}
```
