# Anchor Prediction: Automatic Refinement of Internet Links

This repository contains accompanying material for ["Anchor Prediction: Automatic Refinement of Internet Links"](https://arxiv.org/abs/2305.14337).


Internet links enable users to deepen their understanding of a topic by providing convenient access to related information.
However, the majority of links are _unanchored_---they link to a target document as a whole, and readers may expend considerable
effort localizing the specific parts of the target webpage that enrich their understanding of the link's source context.
To help readers effectively find information in linked webpages, we introduce the task of _anchor prediction_, where the goal
is to identify the specific part of the linked target webpage that is most related to the source linking context.
These anchors serve as finer-grained destinations/previews for a link that may be more informative than the typical
first-paragraph heuristic.

## Datasets

We construct two datasets for anchor prediction: (1) AuthorAnchors, which is automatically constructed from
naturally-occurring anchored hyperlinks on Wikipedia, and (2) ReaderAnchors, which contain human judgments of
the target page paragraph that is most related to the source link context.

### AuthorAnchors

AuthorAnchors contains 27186 examples for training, 3398 examples for development, and 3398
examples for testing. These examples are available at:

* [https://storage.googleapis.com/gresearch/anchors-wikipedia/authoranchors-train.jsonl](https://storage.googleapis.com/gresearch/anchors-wikipedia/authoranchors-train.jsonl)
* [https://storage.googleapis.com/gresearch/anchors-wikipedia/authoranchors-dev.jsonl](https://storage.googleapis.com/gresearch/anchors-wikipedia/authoranchors-dev.jsonl)
* [https://storage.googleapis.com/gresearch/anchors-wikipedia/authoranchors-test.jsonl](https://storage.googleapis.com/gresearch/anchors-wikipedia/authoranchors-test.jsonl)

### ReaderAnchors

ReaderAnchors contains 443 examples. These examples are available at:

* [https://storage.googleapis.com/gresearch/anchors-wikipedia/readeranchors.jsonl](https://storage.googleapis.com/gresearch/anchors-wikipedia/readeranchors.jsonl)

### Data Format

Each examples file contains newline-separated JSON dictionaries with the following fields:

* `src_title`: The source page title
* `src_oldid`: The Wikipedia revision ID of the source page
* `src_context`: The source page content
* `tgt_title`: The target page title
* `tgt_oldid`: The Wikipedia revision ID of the target page
* `candidate_anchors`: The target page anchor candidate set, which is a list of JSON dictionaries with paragraph data:
  * `text`: The text of the target paragraph paragraph
  * `heading`: The heading of the section that contains this target page paragraph.
  * `heading_level`: An integer referring to the heading level. Top-level Wikipedia section headings have `heading_level` of `1`, subsections have `heading_level` of `2`, etc.
* `src_context_link_span`: The source page mention location (an index into the source page content along with a short text context to introduce redundancy that enables error checking)
* `relevant_anchor_indices`: A list of integers with the set of correct anchors. Each integer represents is into the anchor candidate set `candidate_anchors`.
* `example_id`: A unique example ID

## Citation

More details are available in [our paper](https://arxiv.org/abs/2305.14337), which can be cited as follows.

```
@misc{liu2023anchor,
      title={Anchor Prediction: Automatic Refinement of Internet Links},
      author={Nelson F. Liu and Kenton Lee and Kristina Toutanova},
      year={2023},
      eprint={2305.14337},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## License
AuthorAnchors and ReaderAnchors are released under the [Creative Commons Share-Alike 3.0](https://creativecommons.org/licenses/by-sa/3.0/) license.
