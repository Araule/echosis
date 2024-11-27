from echosis import spacy_model as sm
from echosis import utils

spacy_model = "fr_core_news_lg"   # do not use transformer model for data augmentation


utils.k_fold(
    input_file="./corpus/annotations/annotated_corpus.jsonl",
    output_dir="./corpus/k_fold/",
    k=5,
    dev=True
)

print("\n\n# ====================================================")
print("# k-fold cross validation")
print("# ====================================================\n\n")

for k in [1, 2, 3, 4, 5]:
    sm.preprocess(
        train_path=f"./corpus/k_fold/train_{k}.jsonl",
        dev_path=f"./corpus/k_fold/dev_{k}.jsonl",
        spacy_model=spacy_model,
        n_sentence=16,
    ) # n_sentence is how many times you want to augment each sentence

    sm.train_model(
        config_path="./docs/gpu_config.cfg",
        model_path=f"./models/{k}/",
        train_path=f"./corpus/k_fold/train_{k}.spacy",
        dev_path=f"./corpus/k_fold/dev_{k}.spacy"
    ) # do not forget to init spacy config on your machine

    sm.scores(f"./corpus/k_fold/test_{k}.jsonl",
              f"./models/{k}/model-best"
    )
