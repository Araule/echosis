# /bin/env python
# -*- coding: utf-8 -*-

"""how to annotate comments sub-corpus with perspective api
"""

from echosis import toxicity_model

# preprocess corpus
toxicity_model.preprocess(
	input_file="./corpus/tatiana_ventose_comments.csv",
	output_file="./corpus/perspective/corpus.json"
)

# annotate
toxicity_model.annotate(
	api_key="key",
	corpus_file="./corpus/perspective/corpus.json",
	annot_file="./corpus/perspective/annots.jsonl",
	error_file="./corpus/perspective/errors.log"
)

# add annots to the corpus
toxicity_model.write_annots(
	input_file="./corpus/perspective/annots.jsonl",
	output_file="./corpus/tatiana_ventose_comments.csv"
)
