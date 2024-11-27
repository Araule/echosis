# /bin/env python
# -*- coding: utf-8 -*-

"""how to create annotation files for manual annotation or prodi.gy
"""

from echosis import utils

# create an empty annotation file
utils.comments_to_annots(
    filename="./corpus/tatiana_ventose_comments.csv", n_v=10, n_c=10
)

# create an empty annotation file with the whole corpus
utils.comments_to_annots(
    filename="./corpus/tatiana_ventose_comments.csv"
)

# create prodi.gy format file for training
utils.annots_to_jsonl(
    input_file="./corpus/annotations/first-comments_laura.csv",
    output_file="./corpus/annotations/first-comments_laura.jsonl",
    corpus_type="first_comments",
    column_name="label"
)

utils.annots_to_jsonl(
    input_file="./corpus/annotations/replies_laura.csv",
    output_file="./corpus/annotations/replies_laura.jsonl",
    corpus_type="replies",
    column_name="label"
)

# create an empty annotation file for prodigy with the whole corpus
utils.annots_to_jsonl(
    input_file="./corpus/annotations_empty/first-comments_empty.csv",
    output_file="./corpus/annotations_empty/first-comments_empty.jsonl",
    corpus_type="first_comments"
)

utils.annots_to_jsonl(
    input_file="./corpus/annotations_empty/replies_empty.csv",
    output_file="./corpus/annotations_empty/replies_empty.jsonl",
    corpus_type="replies"
)

# get matrix of cohen kappa scores
utils.inter_annotators_agreement(
    "./corpus/first-comments_iaa/", "label"
)

# check the disagreement in the annotations
utils.check_labels(
    "./corpus/first-comments_iaa/", "label"
)
