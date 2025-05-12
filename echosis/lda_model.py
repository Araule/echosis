# ====================================================
# LDA TOPIC MODEL with GENSIM
# ====================================================
#
# to annotate videos with gensim topic model
#

from typing import Optional

from echosis.utils import (
    create_dir,
    load_file,
    save_file,
)
from echosis.config import GensimConfig
import spacy
from spacy.language import Language
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.models.callbacks import (
    PerplexityMetric,
    CoherenceMetric,
    ConvergenceMetric,
)
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import requests
from requests.exceptions import ConnectionError, HTTPError
import pickle
import polars as pl
import os
import numpy as np


ProcessedCorpus = tuple[list[list[tuple[int, int]]], list[list[str]], dict, Dictionary]
LoadedModel = tuple[LdaModel, list[list[tuple[int, int]]], Dictionary]

def preprocess(config_file: str) -> ProcessedCorpus:
    """to preprocess corpus for Gensim model

    Args:
        config_file (str): path to gensim config JSON file.
    """
    conf = GensimConfig.read_file(config_file)

    nlp = spacy.load(conf.spacy_model, exclude=["tok2vec", "morphologizer", "parser", "senter", "ner"])

    filtered_videos = load_file(conf.input_file).filter(pl.col("captions").ne(""))
    docs = [element[0] for element in filtered_videos.select("captions").iter_rows()]
    docs = [spacy_filter(doc, nlp, conf.more_stop) for doc in docs]

    dictionary = Dictionary(docs)
    dictionary.filter_extremes(no_below=conf.no_below, no_above=conf.no_above)
    temp = dictionary[0]
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    id2word = dictionary.id2token

    return corpus, docs, id2word, dictionary


def spacy_filter(doc: str, nlp: Language, stopwords: Optional[list[str]] = None)-> list[str]:
    """to filter stopwords and punctuation with spacy

    Args:
        doc (str): captions of one YouTube video.
        nlp (Language): loaded spacy model.
        stopwords (list[str], optional): stopwords from the config file.
            default to None.
    """
    if stopwords is None:
        stopwords = []

    return [
        tok.lemma_ for tok in nlp(doc) if not tok.is_stop and not tok.is_punct and tok.text not in stopwords
    ]


def train_model(config_file: str, corpus: list[list[tuple[int, int]]], docs: list[list[str]], id2word: dict, dictionary: Dictionary) -> LdaModel:
    """to train a lda model and get scores to evaluate the quality of clustering results.

    Arguments:
        config_file (str): path to Gensim config JSON file.
        corpus (list[list[tuple[int, int]]]): stream of document vectors or sparse matrix of shape (num_documents, num_terms).
        docs (list[list[str]]): list of tokenized captions.
        id2word (dict): mapping from word IDs to words.
        dictionary (Dictionary): Gensim dictionary mapping of id word to create corpus.
    """
    conf = GensimConfig.read_file(config_file)
    if conf.visdom_flag:
        check_visdom_server()
        perplexity = PerplexityMetric(
            corpus=corpus,
            logger="visdom",
            title="perplexity"
        )
        convergence = ConvergenceMetric(
            distance=conf.convergence_distance,
            logger="visdom",
            title="convergence"
        )
        coherence = CoherenceMetric(
            texts=docs,
            dictionary=dictionary,
            coherence=conf.coherence_metric,
            logger="visdom",
            title="coherence",
        )
        model = LdaModel(
            corpus=corpus,
            id2word=id2word,
            passes=conf.passes,
            iterations=conf.iterations,
            chunksize=conf.chunksize,
            num_topics=conf.num_topics,
            alpha=conf.alpha,
            eta=conf.eta,
            callbacks=[perplexity, convergence, coherence],
        )
    else:
        model = LdaModel(
            corpus=corpus,
            id2word=id2word,
            passes=conf.passes,
            iterations=conf.iterations,
            chunksize=conf.chunksize,
            num_topics=conf.num_topics,
            alpha=conf.alpha,
            eta=conf.eta,
        )

    return model


def check_visdom_server() -> None:
    """to check if visdom server is running before training lda model
    """
    try:
        response = requests.get('http://localhost:8097')
        response.raise_for_status()
        print("\nvisdom server is running and can be accessed at: http://localhost:8097")
    except ConnectionError:
        raise ConnectionError("Connection error. Visdom is not running. Please retry after launching visdom server: ~$ visdom")
    except HTTPError as http_err:
        raise HTTPError(f"HTTP error : {http_err}")
    except Exception as e:
        raise Exception(f"\nAn error happened : {e}")


def save_model(config_file: str, model: LdaModel, corpus: list[list[tuple[int, int]]], dictionary: Dictionary) -> None:
    """to save a Gensim lda model, corpus and dictionary to disk.

    Args:
        config_file (str): path to Gensim config JSON file.
        model (LdaModel): Gensim lda model.
        corpus (list[list[tuple[int, int]]]): stream of document vectors or sparse matrix of shape
            (num_documents, num_terms).
        dictionary (Dictionary): Gensim dictionary mapping of id word to create corpus.
        docs (int): list of documents, tokenized or not
    """
    conf = GensimConfig.read_file(config_file)
    create_dir(conf.model_output, False)

    model.save(conf.model_output+"model")
    with open(conf.model_output+"corpus.pkl", "wb") as f:
        pickle.dump(corpus, f)
    dictionary.save(conf.model_output+"corpus.dict")


def load_model(config_file: str) -> LoadedModel:
    """to load a Gensim lda model, corpus and dictionary from disk.

    Args:
        config_file (str): path to Gensim config JSON file.
    """
    conf = GensimConfig.read_file(config_file)

    if not os.path.isfile(conf.model_output + "model"):
        raise FileNotFoundError(f"'{conf.model_output}model' does not exist.")
    elif not os.path.isfile(conf.model_output + "corpus.dict"):
        raise FileNotFoundError(f"'{conf.model_output}corpus.dict' does not exist.")
    elif not os.path.isfile(conf.model_output + "corpus.pkl"):
        raise FileNotFoundError(f"'{conf.model_output}corpus.pkl' does not exist.")

    dictionary = Dictionary.load(conf.model_output+"corpus.dict")
    model = LdaModel.load(conf.model_output+"model")
    with open(conf.model_output+"corpus.pkl", "rb") as f:
        corpus = pickle.load(f)

    return model, corpus, dictionary


def write_annots(config_file: str, column_name: str, model: Optional[LdaModel] = None, corpus: Optional[list[list[tuple[int, int]]]] = None) -> None:
    """to write annotations in corpus.
        if model AND corpus are not given, they will be loaded.
        if a column with the same name as `column_name` exists, it will be overwritten.

    Args:
        config_file (str): path to Gensim config JSON file.
        column_name (str): name of the new column for topics.
        model (LdaModel, optional): Gensim lda model.
            default to None.
        corpus (list[list[tuple[int, int]]], optional): stream of document vectors or sparse matrix of shape
            (num_documents, num_terms).
            default to None.
    """
    conf = GensimConfig.read_file(config_file)
    if not model or not corpus:
        model, corpus, _ = load_model(config_file)

    videos = load_file(conf.input_file)
    filtered_videos = videos.filter(pl.col("captions").ne(""))

    # delete old annotations if they exist
    if column_name in videos.columns:
        videos = videos.drop(column_name)

    videos = videos.join(
        pl.DataFrame(
            {
                "video_id": filtered_videos["video_id"],
                column_name: [
                    [str(topic+1) for topic, proba in element] for element in model.get_document_topics(corpus, minimum_probability=conf.minimum_probability)
                ]
            }
        ),
        how="left", 
        on="video_id",
    ).with_columns(
        pl.col("gensim_topics").list.join("|").fill_null("0").str.replace_all("^$", "0")
    )
    save_file(videos, conf.input_file)


def write_infos(config_file: str, model: Optional[LdaModel] = None, corpus: Optional[list[list[tuple[int, int]]]] = None, dictionary: Optional[Dictionary] = None) -> None:
    """to save lda model infos.
        if model AND corpus AND dictionary are not given, they will be loaded.

    Args:
        config_file (str): path to Gensim config JSON file.
        model (LdaModel, optional): Gensim lda model.
            default to None.
        corpus (list[list[tuple[int, int]]], optional): stream of document vectors or sparse matrix of shape
            (num_documents, num_terms).
            default to None.
        dictionary (Dictionary, optional): Gensim dictionary mapping of id word to create corpus.
            default to None.
    """
    conf = GensimConfig.read_file(config_file)
    create_dir(conf.model_infos, False)
    if not model or not dictionary or not corpus:
        model, corpus, dictionary = load_model(config_file)

    # save docs
    filtered_videos = load_file(conf.input_file).filter(pl.col("captions").ne(""))
    annotations = model.get_document_topics(corpus, minimum_probability=conf.minimum_probability)
    model_docs = pl.DataFrame({
        "video_id": filtered_videos["video_id"].to_list(),
        "gensim_topics": [
            "|".join([str(topic+1)+"-"+str(proba) for topic, proba in element]) for element in annotations
        ]
    })
    save_file(model_docs, conf.model_infos+"model_docs.csv")

    # save topics
    model_topics = {"gensim_topics": [], "topic_terms": []}
    for topic in range(model.num_topics):
        model_topics["gensim_topics"].append(str(topic+1))
        model_topics["topic_terms"].append("|".join([dictionary[tok] for tok, prob in model.get_topic_terms(topic, topn=20)]))
    save_file(pl.DataFrame(model_topics), conf.model_infos+"model_topics.csv")

    # generate the ldaviz visualisation
    vis_data = gensimvis.prepare(model, corpus, dictionary)
    with open(conf.model_infos + "model_ldaviz.html", "w", encoding="UTF-8") as wf:
        pyLDAvis.save_html(vis_data, wf)