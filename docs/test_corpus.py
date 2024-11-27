# /bin/env python
# -*- coding: utf-8 -*-

"""how to create a corpus
"""

from echosis import corpus

# get videos sub-corpus
corpus.get_videos(
	channel_url="https://www.youtube.com/c/TatianaVentôseOfficiel",
    output_file="./corpus/tatiana_ventose_videos.csv", langs=["fr"]
)

# get comments sub-corpus
corpus.get_comments(
    input_file="./corpus/tatiana_ventose_videos.csv",
    output_file="./corpus/tatiana_ventose_comments.csv"
)

# get commentators sub-corpus
corpus.get_commenters(
    input_file="./corpus/tatiana_ventose_comments.csv",
    output_file="./corpus/tatiana_ventose_commenters.csv"
)
