# /bin/env python
# -*- coding: utf-8 -*-

"""how to annotate videos sub-corpus with gensim lda model
"""

from echosis import lda_model

# gensim config file
conf_path = "./gensim.json"

# preprocess corpus
corpus, docs, id2word, dictionary = lda_model.preprocess(conf_path)
print("unique lemmas:", len(dictionary))
print("number of documents:", len(docs))
print("number of tokens in each documents:", ", ".join([str(len(d)) for d in docs]))

# train model
model = lda_model.train_model(conf_path, corpus, docs, id2word, dictionary)

# save model
lda_model.save_model(conf_path, model, corpus, dictionary)

# get model infos by loading model
lda_model.write_infos(conf_path)

# get model infos by giving model
lda_model.write_infos(conf_path, model, corpus, dictionary)

# add annots to the corpus by loading model
lda_model.write_annots(conf_path, 'gensim_topics')

# add annots to the corpus by giving model
lda_model.write_annots(conf_path, 'gensim_topics', model, corpus)
