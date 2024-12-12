from echosis import spacy_model as sm
from echosis import utils

spacy_model = "fr_core_news_lg"   # do not use transformer model for data augmentation

# split corpus in k sub-corpus for k-fold cross-validation
utils.k_fold(
    input_file="./corpus/annotations/replies_laura.jsonl",
    output_dir="./corpus/k_fold/",
    k=5,
    dev=True
)

# train models
print("\n\n# ====================================================")
print("# k-fold cross validation")
print("# ====================================================\n\n")

for k in [1, 2, 3, 4, 5]:
    sm.preprocess(
        train_path=f"./corpus/k_fold/train_{k}.jsonl",
        dev_path=f"./corpus/k_fold/dev_{k}.jsonl",
        spacy_model=spacy_model,
        n_sentence=0,
    )

    sm.train_model(
        config_path="./docs/cpu_config.cfg",
        model_path=f"./models/k_fold/{k}/",
        train_path=f"./corpus/k_fold/train_{k}.spacy",
        dev_path=f"./corpus/k_fold/dev_{k}.spacy"
    ) # do not forget to init spacy config on your machine

# get scores
sm.cross_validation_scores("./corpus/k_fold/test*.jsonl", "./models/k_fold/")