# ====================================================
# TOXICITY MODEL with perspective API
# ====================================================
#
# to annotate toxicity, and other indicators of hate
#

import polars as pl
from echosis.utils import load_file, save_file
from googleapiclient import discovery
from googleapiclient.errors import HttpError
import time
import json
from typing import Optional
from rich.progress import track


def run_perspective_client(api_key: str) -> discovery.Resource:
    """to run perspective api client

    Args:
        api_key (str): perspective api key.
    """
    try:
        return discovery.build(
            "commentanalyzer",
            "v1alpha1",
            developerKey=api_key,
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
            static_discovery=False
        )
    except HttpError as e:
        raise HttpError(f"Http error occurred: {e.resp.status} - {e.reason}")
    except Exception as e:
        raise Exception(f"An error occurred: {e}")


def preprocess(input_file: str, output_file: str) -> None:
    """to preprocess the input file into JSON or JSONL file

    Args:
        input_file (str): path to comments file with 'text' and 'comment_id' columns.
        output_file (str): {'any_filename.json', 'any_filename.jsonl'}.
            path to preprocessed corpus.
    """
    print("preprocessing file...")
    corpus = load_file(
        input_file
    ).with_columns(
        pl.col("text").str.split(by="\n").list.join(" ")
    ).with_columns(
        pl.col("text").map_elements(request, return_dtype=pl.Struct).alias("request")
    ).select(
        ["comment_id", "request"]
    )
    save_file(corpus, output_file, valid_extensions=[".json", ".jsonl"])


def request(text: str) -> dict:
    """to get empty perspective request

    Args:
        text (str): one dataframe row.
    """
    return {
        "comment": 
            {"text": text}, 
            "languages": ["fr"],
            "requestedAttributes": {
                "TOXICITY": {},
                "SEVERE_TOXICITY": {},
                "IDENTITY_ATTACK": {},
                "INSULT": {},
                "PROFANITY": {},
            },
    }


def annotate(api_key: str, corpus_file: str, annot_file: str, error_file: Optional[str] = None, quota: Optional[float] = 1.0):
    """to get and save perspective api scores in a JSONL file

    Args:
        api_key (str): perspective api key.
        corpus_file (str): path to the preprocessed corpus.
        annot_file (str): path to the file where scores are saved.
        error_file (str, optional): if not None, errors are saved in file.
            default to None.
        quota (float, optional): number of api requests per second as 1/number (should be less than 1.0 unless
            quota error happens too often).
            default to 1.0.
    """
    if (quota < 0.0) or (quota > 1.0):
        raise ValueError("quota must be between 0.0 and 1.0")

    try:
        annot = load_file(annot_file, valid_extensions=[".jsonl"])
        print(f"Annotation file has been found. Starting from {annot.height}e comment...")
        corpus = load_file(corpus_file).filter(pl.col("comment_id").is_in(annot["comment_id"]).not_())
    except FileNotFoundError:
        print("No annotation file found. Starting from first comment...")
        corpus = load_file(corpus_file)

    client = run_perspective_client(api_key)

    with open(annot_file, "a", encoding="UTF-8") as file:
        for comment_id, body in track(corpus.iter_rows(), total=corpus.height, description="getting annotations"):
            while True: # to catch some errors like quota reach
                try:
                    time.sleep(quota)
                    response = client.comments().analyze(body=body).execute()
                    json.dump(scores(comment_id, response["attributeScores"]), file)
                    file.write("\n")
                    break
                except HttpError as e:
                    if e.resp.status == 429:
                        print("Quota has been reached. If error happens to often, augmenter quota might be needed.")
                        print("Retrying...")
                        continue
                    else:
                        raise HttpError(f"Http error occurred: {e.resp.status} - {e.reason}")
                except Exception as e:
                    print("An error occurred:", str(e))
                    if error_file:
                        with open(error_file, "a", encoding="UTF-8") as f:
                            f.write(f"{comment_id}, {e}\n")
                    break


def scores(comment_id: str, res: dict) -> dict:
    """to get all perspective scores attached to the comment id

    Args:
        comment_id (str): YouTube comment id.
        res (dict): attributeScores value of perspective api response.
    """
    return {
        "comment_id": comment_id,
        "toxicity": res.get('TOXICITY', {'summaryScore': {'value': ''}})['summaryScore']['value'],
        "severe_toxicity": res.get('SEVERE_TOXICITY', {'summaryScore': {'value': ''}})['summaryScore']['value'],
        "identity_attack": res.get('IDENTITY_ATTACK', {'summaryScore': {'value': ''}})['summaryScore']['value'],
        "insult": res.get('INSULT', {'summaryScore': {'value': ''}})['summaryScore']['value'],
        "profanity": res.get('PROFANITY', {'summaryScore': {'value': ''}})['summaryScore']['value'],
    }


def write_annots(input_file: str, output_file: str):
    """to write scores in the input file

    Args:
        input_file (str): file where perspective score has been saved.
        output_file (str): path to the input file with 'text' and 'comment_id' columns.

    """
    annots = load_file(input_file)
    comments = load_file(output_file).join(annots, on="comment_id", how="left")
    save_file(comments, output_file)
    