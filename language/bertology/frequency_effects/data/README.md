# Evaluation Stimuli for Frequency Effects on Syntactic Rule-Learning in Transformers

This directory contains the nonce evaluation stimuli for our paper:
- `nouns.tsv` contains 200 nouns.
- `verbs.tsv` contains 336 verbs.
- `sentential_contexts.tsv` contains 56 "sentential contexts" (sentences with the noun and verb masked out).

Any noun and verb can be inserted into a sentential context to form a "nonce" (grammatically correct but semantically meaningless) sentence. These nouns, verbs, and sentential contexts allows us to generate a evaluation set of 7.5 million examples.

## Column header descriptions

`verbs.tsv`:

| Column Header | Description |
| ----------- | ----------- |
| `s_verb` | singular verb |
| `s_verb_freq` | frequency of singular verb in BERT's training set (log10) |
| `s_verb_nfreq` | frequency of singular verb used as a noun in BERT's training set (log10) |
| `s_verb_vfreq` | frequency of singular verb used as a verb in BERT's training set (log10) |
| `s_verb_direct_div` | number of unique nouns that this singular verb has been used in a subject-verb pair with (log10) |
| `s_verb_indirect_div` | number of unique nouns that have occurred in the same sentence as this singular verb (log10) |
| `p_verb_{x}` | same as `s_verb_{x}`, but for the plural form of the verb |
| `sp_freq_diff` | `s_verb_freq` - `p_verb_freq` |

`nouns.tsv`:

| Column Header | Description |
| ----------- | ----------- |
| `s_noun` | singular noun |
| `s_noun_freq` | frequency of singular noun in BERT's training set (log10) |
| `s_noun_nfreq` | frequency of singular noun used as a verb in BERT's training set (log10) |
| `s_noun_vfreq` | frequency of singular noun used as a noun in BERT's training set (log10) |
| `s_noun_direct_div` | number of unique verbs that this singular noun has been used in a subject-verb pair with (log10) |
| `s_noun_indirect_div` | number of unique verbs that have occurred in the same sentence as this singular verb (log10) |
| `p_noun_{x}` | same as `s_noun_{x}`, but for the plural form of the verb |
| `sp_freq_diff` | `s_noun_freq` - `p_noun_freq` |

`sentential_contexts.tsv`

| Column Header | Description |
| ----------- | ----------- |
| `template_id` | an ID number that can help identify the template |
| `original_inflection` | the inflection of the original verb used in the sentence (singular/plural) |
| `num_attractors` | the number of "distracting" clauses between the subject and verb |
| `subject` | the original subject used in the sentence |
| `correct_verb` | the original verb used in the sentence |
| `incorrect_verb` | the incorrect inflection of the original verb used in the sentence |
| `sv_distance` | number of words between the subject and verb |
| `template` | sentential context |
