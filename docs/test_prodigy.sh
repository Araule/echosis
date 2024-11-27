# ====================================================
# ECHOSIS CLASSIFICATION AND ANNOTATION with PRODI.GY
# ====================================================
#
# to annotate corpus and train classification model
# not used for now

# ====================================================
# database
# ====================================================

# import database (use prodi.gy jsonl format to import annotations)
prodigy db-in agree_disagree ./data.jsonl

# export database
prodigy db-out agree_disagree > ./data.jsonl

# delete database
prodigy drop agree_disagree

# check all databases
prodigy stats -l

# ====================================================
# annotation
# ====================================================
# if you do not have enough data annotated, you wan
# increase your corpus by annotating with prodi.gy

# manually annotate categories
prodigy textcat.manual agree_disagree ./data.jsonl --label accord,desaccord,ambigue,hs --exclusive

# use active learning and a model in a loop
# can be useful with unbalanced classes
prodigy textcat.teach agree_disagree ./annotations/prodigy/model-best ./data.jsonl

# ====================================================
# train model for text classification
# ====================================================
#
# if training on gpu, see here to install spacy with cuda: https://spacy.io/usage
#
# you need prodigy evaluate plugin to evaluate the models
# pip install "prodigy-evaluate @ git+https://github.com/explosion/prodigy-evaluate"

# import annotated data if not already done
prodigy db-in agree_disagree_train ./train.jsonl
prodigy db-in agree_disagree_dev ./dev.jsonl
prodigy db-in agree_disagree_test ./test.jsonl

# training
mkdir -p ./models/prodigy/
# train on CPU
prodigy train ./models/prodigy/ --textcat agree_disagree_train,eval:agree_disagree_dev --lang fr --label-stats
# train model while checking the quality of collected annotations
prodigy train-curve --textcat agree_disagree_train,eval:agree_disagree_dev --lang fr --show-plot
# evaluate model
prodigy evaluate.evaluate ./models/prodigy/model-best --textcat agree_disagree_test --label-stats --confusion-matrix

# delete databases
prodigy drop agree_disagree_train
prodigy drop agree_disagree_dev
prodigy drop agree_disagree_test
