# ====================================================
# ECHOSIS CLASSIFICATION MODELs with SPACY
# ====================================================
#
# to train classification models
# for agreement annotation
#
# do not forget to init base_config
# python -m spacy init fill-config ./docs/gpu_base_config.cfg ./docs/gpu_config.cfg
# python -m spacy init fill-config ./docs/cpu_base_config.cfg ./docs/cpu_config.cfg
#

import os
import spacy
import glob as glb
from spacy.tokens import DocBin
from spacy.cli.train import train
from rich.progress import track
import augmenty
import polars as pl
from echosis.utils import load_file, save_file
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from typing import Optional
import matplotlib as plt


def create_spacy_file(dataset: list[tuple[str]], output_file: str) -> None:
    """to transform dataset in SPACY file for training.

    Args:
        dataset (list[tuple[str]]): dataset of texts and labels.
        output_file (str): path to output file.
    """
    db = DocBin()
    nlp = spacy.blank("fr")
    for text,label in dataset:
        doc = nlp(text)
        doc.cats = {label: 1.0}
        db.add(doc)
    db.to_disk(output_file)


def preprocess(train_path: str, dev_path: str, spacy_model: str, n_sentence: Optional[int] = 4) -> None:
    """to perform data augmentation on train dataset and save dev and train datasets in SPACY files.

    Args:
        train_path (str): path to train dataset.
        dev_path (str): path to dev dataset.
        spacy_model (str): name or path of the spacy model.
        n_sentence (int): number of sentences created for each train text.
    """
    train = load_file(train_path)
    text = train["text"].to_list()
    label = train["label"].to_list()

    augmented_train = data_augmentation(text, label, spacy_model, n_sentence)

    path, _ = os.path.splitext(train_path)
    create_spacy_file(augmented_train, path+".spacy")

    dev = load_file(dev_path)
    dev = [(text, label) for text, label in zip(dev["text"].to_list(), dev["label"].to_list())]
    path, _ = os.path.splitext(dev_path)
    create_spacy_file(dev, path+".spacy")


def data_augmentation(texts: list[str], labels: list[str], spacy_model: str, n_sentence: int) -> list[tuple[str]]:
    """to perform data augmentation on train dataset

    Args:
        texts (str): list of texts.
        labels (str): list of labels associated with texts.
        spacy_model (str): name or path of the spacy model.
        n_sentence (int): number of sentences created for each train text.
    """
    nlp = spacy.load(spacy_model)

    keystroke_error_augmenter = augmenty.load("keystroke_error_v1", level=0.1, keyboard="fr_azerty_v1")
    random_casing_augmenter = augmenty.load("random_casing_v1", level=0.1)
    token_swap_augmenter = augmenty.load("token_swap_v1", level=0.1)
    word_embedding_augmenter = augmenty.load("word_embedding_v1", level=0.1)

    zipped = list(zip(nlp.pipe(texts), labels))

    dataset = []
    for doc, label in track(zipped, description="data augmentation", total=len(zipped)):
        for i in range(round(n_sentence)):
            dataset.append([list(augmenty.docs(doc, augmenter=keystroke_error_augmenter, nlp=nlp))[0], label])
            dataset.append([list(augmenty.docs(doc, augmenter=random_casing_augmenter, nlp=nlp))[0], label])
            dataset.append([list(augmenty.docs(doc, augmenter=token_swap_augmenter, nlp=nlp))[0], label])
            dataset.append([list(augmenty.docs(doc, augmenter=word_embedding_augmenter, nlp=nlp))[0], label])
    dataset.extend([[t,l] for t, l in zip(texts, labels)])
    return dataset


def train_model(config_path: str, model_path: str, train_path: str, dev_path: str) -> None:
    """to train a spacy model.

    Args:
        config_path (str): path to spacy config file.
        model_path (str): path to save model file.
        train_path (str): path to train dataset.
        dev_path (str): path to dev dataset.
    """
    train(config_path=config_path, output_path=model_path, overrides={"paths.train": train_path, "paths.dev": dev_path})


def scores(test_path, model_path) -> None:
    """to test model and get score.

    Args:
        test_path (str): path to test dataset.
        model_path (str): path to spacy model.
    """
    test = load_file(test_path)
    texts = test["text"].to_list()

    y_pred = get_annots(texts, model_path)
    y_true = test["label"].to_list()

    print(classification_report(y_true, y_pred))
    get_confusion_matrix(y_true, y_pred)


def cross_validation_scores(test_pattern_path: str, models_path: str) -> None:
    """to test models obtain through k-fold cross-validation and get scores.
        still working on it.

    Args:
        test_pattern_path (str): path to test datasets.
            exemple: "./corpus/test*.jsonl".
        models_path (str): path to save model file.
    """
    files = [[load_file(file).get_column("text").to_list(), load_file(file).get_column("label").to_list()] for file in glb.glob(test_pattern_path)]

    results = []
    for i, texts, y_true in enumerate(files):
        y_pred = get_annots(texts, models_path+f"{i+1}/model-best/")
        results.append(classification_report(y_true, y_pred, output_dict=True))
        # print(f"confusion matrix saved at ./cm_{i+1}.png")
        # get_confusion_matrix(y_true, y_pred).savefig(f"./cm_{i+1}.png")
    print(results)


def get_annots(texts: list[str], model_path: str) -> list[str]:
    """to get labels with a spacy model.

    Args:
        texts (list[str]): list of texts.
        model_path (str): trained spacy model.

    Returns:
        list[str]: list of predicted labels
    """
    nlp = spacy.load(model_path)
    docs = nlp.pipe(texts)
    labels = [max(doc.cats, key=doc.cats.get) for doc in docs]
    return labels


def get_confusion_matrix(y_true: list[str], y_pred: list[str], k: Optional[int] = None) -> None:
    """to create and save confusion matrix.

    Args:
        y_true (list[str]): true labels.
        y_pred (list[str]): predicted labels.
        k (int, optional): id of the fold if confusion matrix is needed for k-fold cross-validation.
            it really is optional.
            it's for the title...
            defaults to None.
    """
    cm = confusion_matrix(y_true, y_pred)

    _, _ = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".1f", cmap="Blues")

    if k:
        plt.title(f"Confusion Matrix for k={k}", fontsize=20, fontweight="bold")
    else:
        plt.title("Confusion Matrix", fontsize=20, fontweight="bold")

    plt.xlabel("Predicted labels", fontsize=14)
    plt.ylabel("True labels", fontsize=14)

    return plt


def write_annots(input_file: str, model_path: str) -> None:
    """to annotate comments with a spacy model.

    Arguments:
        input_file (str): path to a comments file with 'text' column.
        model_path (str): path to the spacy classification model.
    """
    comments = load_file(input_file)

    nlp = spacy.load(model_path)
    docs = nlp.pipe(comments["text"].to_list())
    textcat = nlp.get_pipe("textcat")

    for label in textcat.labels:
        comments = comments.with_columns(
            [doc.cats[label] for doc in docs].alias(label)
        )

    save_file(comments, input_file)
