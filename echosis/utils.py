# ====================================================
# ECHOSIS UTILS
# ====================================================
#
# small functions used to load and save files,
# create annotation files,
# and calculate inter-annotator agreement
#

import os
import polars as pl
from polars.exceptions import ComputeError
from echosis.exceptions import (
    InvalidFileExtensionError,
    InsufficientAnnotatorsError,
    UnequalLabelsError,
)
from typing import Optional
from pathlib import Path
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np


# ====================================================
# load and save files
# ====================================================


def load_file(filename: str, valid_extensions: Optional[list[str]] = None) -> pl.DataFrame:
    """to load any file with polars

    Args:
        filename (str): path to the file.
        valid_extensions (list[str], optional): list of valid file extensions.
            default to ['.csv', '.tsv', '.json', '.jsonl', .'txt'].

    Returns:
        pl.DataFrame: file content
    """
    if valid_extensions is None:
        valid_extensions = [".csv", ".tsv", ".json", ".jsonl", ".txt"]
    elif isinstance(valid_extensions, str):
        valid_extensions = [valid_extensions]

    if not os.path.isfile(filename):
        raise FileNotFoundError(f"{filename} not found.")

    ext = verify_format(filename, valid_extensions)

    if ext == ".csv":
        return pl.read_csv(filename, separator=",")
    elif ext == ".tsv" or ext == ".txt":
        return pl.read_csv(filename, separator="\t")
    else:
        try:
            if ext == ".json":
                return pl.read_json(filename)
            else:
                return pl.read_ndjson(filename)
        except ComputeError:
            raise ComputeError(f"{filename} is empty or in wrong format. It cannot be loaded as a dataframe.")
        except Exception as e:
            raise Exception(f"An error occurred: {e}")


def save_file(df: pl.DataFrame, filename: str, valid_extensions: Optional[list[str]] = None) -> None:
    """to save a dataframe

    Args:
        df (pl.DataFrame): dataframe to save.
        filename (str): path to the file.
        valid_extensions (list[str], optional): list of valid file extensions.
            default to ['.csv', '.tsv', '.json', '.jsonl', '.txt'].
    """
    create_dir(filename, True)

    if valid_extensions is None:
        valid_extensions = [".csv", ".tsv", ".json", ".jsonl", ".txt"]
    elif isinstance(valid_extensions, str):
        valid_extensions = [valid_extensions]

    ext = verify_format(filename, valid_extensions)

    if ext == ".csv":
        df.write_csv(filename, separator=",")
    elif ext == ".tsv" or ext == ".txt":
        df.write_csv(filename, separator="\t")
    elif ext == ".json":
        df.write_json(filename)
    elif ext == ".jsonl":
        df.write_ndjson(filename)
    print(f"file saved in {filename}")


def create_dir(path: str, is_file: Optional[bool] = False) -> None:
    """to create directories present in the path

    Args:
        path (str): path to the directory.
        is_file (bool, optional): if true, filename is removed from path.
            default to False.
    """
    if is_file:
        path = os.path.dirname(path)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def verify_format(filename: str, valid_extensions: list[str]) -> str:
    """to verify the file format

    Args:
        filename (str): path to the file.
        valid_extensions (list[str]): {'.any_ext'}.
            list of valid file extensions.

    Returns:
        str: file suffix in lower case
    """
    _, ext = os.path.splitext(filename)

    if ext.lower() not in valid_extensions:
        valid_exts = ", ".join("`{}`".format(x) for x in valid_extensions)
        raise InvalidFileExtensionError("Unsupported file type, valid only {}".format(valid_exts))

    return ext.lower()


# ====================================================
# from comments sub-corpus, create annotation file
# ====================================================


def comments_to_annots(filename: str, n_v: Optional[int] = None, n_c: Optional[int] = None) -> None:
    """to get empty annotation files for agreement annotation: one for first comments, one for replies.

    Args:
        filename (str): path to comment sub-corpus.
        n_v (int, optional): max number of videos.
            if None, no filter is applied.
            default to None.
        n_c (int, optional): max number of comments.
            if None, no filter is applied.
            default to None.
    """
    comments = load_file(filename)
    video_ids = sampled_videos(comments, n_v) # sampled videos
    comment_ids = sampled_comments(comments, video_ids, n_c) # sampled comments

    first_comments = comments.filter(
        pl.col("comment_id").is_in(comment_ids)
    ).select(
        ["comment_id", "video_title", "text"]
    )

    print("number of comments sampled:", first_comments["comment_id"].len())
    save_file(first_comments, "./first-comments_empty.csv")

    first_comments = comments.select(
        ["comment_id", "video_title", "author_name", "text"]
    ).rename(
        {"comment_id": "first_comment_id", "author_name": "first_commenter_name", "text": "first_comment"}
    )

    replies = comments.filter(
        (pl.col("first_comment_id").is_in(comment_ids)) & (pl.col("position") > 1)
    ).join(
        first_comments, on="first_comment_id"
    ).rename(
        {"author_name": "replier_name", "text": "reply"}
    )

    replies = replies.select(
        ["comment_id", "video_title", "first_commenter_name", "first_comment", "replier_name", "reply", "position"]
    )

    print("number of replies sampled:", replies["comment_id"].len())
    save_file(replies, "./replies_empty.csv")


def sampled_videos(df: pl.DataFrame, n_v: int) -> list[str]:
    """to select x random videos by year.

    Args:
        df (pl.DataFrame): dataframe with 'published_at' and 'video_id' columns.
        n_v (int): max number of videos to be selected each year.

    Return:
        list[str]: list of video ids
    """
    if n_v is None: # no filter applied
        return df["video_id"].unique().to_list()
    else:
        df = df.with_columns(
            year=pl.col("published_at").str.slice(0, length=4)
        ).group_by("year").agg(
            pl.col("video_id")
        ).with_columns(
            pl.col("video_id").list.unique()
        )
        return pl.concat([
            df.filter(pl.col("video_id").list.len() <= n_v),
            df.filter(
                pl.col("video_id").list.len() > n_v
            ).with_columns(
                pl.col("video_id").list.sample(n=n_v)
            )
        ]).get_column("video_id").explode().to_list()


def sampled_comments(df: pl.DataFrame, video_ids: list, n_c: int) -> list[str]:
    """to select x random first comments by videos

    Args:
        df (pl.DataFrame): dataframe with 'comment_id' and 'video_id' columns.
        video_ids (list): list of selected video ids.
        n_c (int): max number of comments to be selected each year.

    Return:
        list[str]: list of comment ids
    """
    df = df.filter(
        (pl.col("video_id").is_in(video_ids)) & (pl.col("position") == 1)
    ).group_by("video_id").agg(
        pl.col("comment_id")
    )
    if n_c is None: # no filter applied
        return df["comment_id"].explode().unique().to_list()
    else:
        return pl.concat([
            df.filter(pl.col("comment_id").list.len() <= n_c),
            df.filter(
                pl.col("comment_id").list.len() > n_c
            ).with_columns(
                pl.col("comment_id").list.sample(n=n_c)
            )
        ]).get_column("comment_id").explode().to_list()


# ====================================================
# from annotation file, create JSONL file for prodi.gy
# ====================================================


def annots_to_jsonl(input_file: str, output_file: str, corpus_type: str, column_name: Optional[str] = None, label_filter: Optional[bool] = False) -> None:
    """to transform an annotation file or corpus file in a JSONL file for prodi.gy.

    Arguments:
        input_file(str): path to annotation file with first comments.
        output_file(str): path to JSONL file.
        corpus_type(str): {first_comments, replies}.
            type of the corpus.
        column_name(str, optional): column name of the column containing the labels (if there is any).
            default to None.
        label_filter(bool, optional): if True, remove comments without a label.
            default to False.
    """
    verify_format(output_file, [".jsonl"])

    if corpus_type == "first_comments":
        corpus = first_comments_to_jsonl(input_file, column_name)
    elif corpus_type == "replies":
        corpus = replies_to_jsonl(input_file, column_name)
    else:
        raise ValueError("Unsupported corpus type, valid only ['first_comments', 'replies']")

    if label_filter:
        corpus = corpus.filter(pl.col("label").is_not_null())

    save_file(corpus, output_file)


def first_comments_to_jsonl(input_file: str, column_name: str) -> pl.DataFrame:
    """to get the first comments in prodi.gy input format

    Args:
        input_file(str): path to annotation file with first comments.
        column_name(str): column name of the column containing the labels (if there is any).

    Returns:
        pl.DataFrame: transformed corpus
    """
    corpus = load_file(
        input_file
    ).with_columns(
        meta=pl.struct([
            pl.col("comment_id"),
            pl.col("video_title"),
        ]),
        html=pl.col("video_title").str.replace(r"^(.+)$", "<h4>Ce premier commentaire provient de la vidéo YouTube: ${1} </h4>")
    )

    if column_name is None:
        return corpus.select(["html", "meta", "text"])
    elif column_name == "label":
        return corpus.select(["html", "meta", "text", "label"])
    else:
        return corpus.rename(
            {column_name: "label"}
        ).select(
            ["html", "meta", "text", "label"]
        )


def replies_to_jsonl(input_file: str, column_name: str) -> pl.DataFrame:
    """to get replies in prodi.gy input format

    Args:
        input_file(str): path to annotation file with first comments.
        column_name(str): column name of the column containing the labels (if there is any).

    Returns:
        pl.DataFrame: transformed corpus
    """
    replies = load_file(input_file)
    discussions = replies.with_columns(
    pl.col("position").cast(pl.String).str.replace_all(r"^(.+)$", "<h4>Réponse, position ${1} </h4>"),
    pl.col("replier_name").str.replace_all(r"^(.+)$", "<p>de: ${1} </p>"),
    pl.col("reply").str.replace_all(r"(\r?\n)+", " ").str.replace_all(r"^(.+)$", "<p>${1} </p>"),
    ).with_columns(
        discussion=pl.col("position")+pl.col("replier_name")+pl.col("reply")
    ).group_by(
        "first_comment"
    ).agg(
        pl.col("discussion")
    ).with_columns(
        pl.col("discussion").list.join("<br/>")
    )

    corpus = replies.join(
        discussions, on="first_comment"
    ).with_columns(
        pl.col("first_commenter_name").str.replace_all(r"^(.+)$", "<p>de: ${1} </p>"),
        pl.col("first_comment").str.replace_all(r"(\r?\n)+", " ").str.replace_all(r"^(.+)$", "<p>${1} </p>")
    ).with_columns(

        html="<h4>Premier commentaire, position 1</h4>"
            +pl.col("first_commenter_name")
            +pl.col("first_comment")
            +"<br/>"
            +pl.col("discussion")
            +pl.col("position").cast(pl.String).str.replace("(.+)", "<h4>Il s'agit d'annoter la réponse à la position ${1} </h4>"),
        meta=pl.struct([
            pl.col("comment_id"),
            pl.col("video_title"),
        ]),
    ).rename(
        {"reply": "text"}
    )

    if column_name is None:
        return corpus.select(["html", "meta", "text"])
    elif column_name == "label":
        return corpus.select(["html", "meta", "text", "label"])
    else:
        return corpus.rename(
            {column_name: "label"}
        ).select(
            ["html", "meta", "text", "label"]
        )


# ====================================================
# split corpus for k-fold cross-validation
# ====================================================


def k_fold(input_file: str, output_dir: str, k: Optional[int] = 5, dev: Optional[bool] = False, shuffle: Optional[bool] = False) -> None:
    """to create multiple corpus for k fold cross-validation and saving them in csv format.
    1/k is given to test.

    Args:
        input_file (str): {'smt.csv', 'smt.tsv', 'smt.json', 'smt.jsonl'}.
            path to the input file.
        output_dir (str): path to the output directory.
        k (int, optional): number of batches.
            1/k is given to test, the rest to train.
            default to 5.
        dev (bool, optional): if dev is true, 1/k is given to dev.
            default to False.
        shuffle (bool, optional): if shuffle is true, data is shuffled before slicing.
            default to False.
    """
    if output_dir[-1] != "/" and "." in output_dir:
        raise NotADirectoryError(f"{output_dir} must be a path to a directory.")

    _, ext = os.path.splitext(input_file)

    df = load_file(input_file).select(
        ["text", "label"]
    ).with_columns(
        pl.col("label").str.replace_all("hs", "ambigue")
    )

    if shuffle:
        df = df.sample(frac=1.0, shuffle=True)

    df_length = df.height
    length = round(df_length/k)

    print("splitting corpus...")
    split_df = []
    for i in range(k):
        offset = round(length) * i
        if offset + length <= df_length:
            split_df.append(df.slice(offset, length))
        else:
            split_df.append(df.slice(offset))


    print("saving corpora...")
    if dev:
        for i in range(k):
            train = split_df[i:] + split_df[:i]
            test = train[-2]
            dev = train[-1]
            train.pop()
            train.pop()
            save_file(pl.concat(train), f"{output_dir}train_{i+1}{ext}")
            save_file(dev, f"{output_dir}dev_{i+1}{ext}")
            save_file(test, f"{output_dir}test_{i+1}{ext}")
    else:
        for i in range(k):
            train = split_df[i:] + split_df[:i]
            test = train[-1]
            train.pop()
            save_file(pl.concat(train), f"./train_{i+1}{ext}")
            save_file(test, f"./test_{i+1}{ext}")


# ====================================================
# calculate inter-annotators agreement
# ====================================================


def inter_annotators_agreement(input_dir: str, column_name: str) -> None:
    """to evaluate annotations produced by annotators
    by getting matrix of Cohen(1960)'s kappa coefficient scores

     Args:
        input_dir (str): path to directory containing all annotation files.
        column_name (str): name of the label column in loaded dataframe.
     """
    labels = [load_file(file).select(column_name) for file in Path(input_dir).iterdir()]
    annotators = [Path(file).stem for file in Path(input_dir).iterdir()]
    num_annotators = len(annotators)

    if num_annotators < 2:
        raise InsufficientAnnotatorsError("There needs to be at least two annotators.")

    coefficients = np.ones((num_annotators, num_annotators))
    obs_agreement = np.ones((num_annotators, num_annotators))
    exp_agreement = np.ones((num_annotators, num_annotators))

    for i in range(num_annotators):
        for j in range(num_annotators):
            if i != j:
                if labels[i].height != labels[j].height:
                    raise UnequalLabelsError(f"{annotators[i]} - {annotators[j]}: There needs to be the same number of labels in each annotation file.")
                coefficients[i, j], obs_agreement[i, j], exp_agreement[i, j] = kappa_cohen_score(labels[i], labels[j], column_name)

    plt.subplots(figsize=(8, 6))
    sns.heatmap(coefficients, annot=True, fmt=".2f", cmap="Blues", xticklabels=annotators, yticklabels=annotators)
    plt.title("Cohen(1960)'s kappa coefficient", fontsize=20, fontweight="bold")
    plt.savefig('./kappa_coefficient.png')
    plt.show()

    plt.subplots(figsize=(8, 6))
    sns.heatmap(obs_agreement, annot=True, fmt=".2f", cmap="Blues", xticklabels=annotators, yticklabels=annotators)
    plt.title("Observed agreement", fontsize=20, fontweight="bold")
    plt.savefig('./observed_agreement.png')
    plt.show()

    plt.subplots(figsize=(8, 6))
    sns.heatmap(exp_agreement, annot=True, fmt=".2f", cmap="Blues", xticklabels=annotators, yticklabels=annotators)
    plt.title("Expected agreement", fontsize=20, fontweight="bold")
    plt.savefig('./expected_agreement.png')
    plt.show()


def kappa_cohen_score(df_i: pl.DataFrame, df_j: pl.DataFrame, column_name: str) -> tuple[float, float, float]:
    """to get Cohen (1960)'s kappa score.

    Args:
        df_i (pl.DataFrame): dataframe containing annotator i's labels.
        df_j (pl.DataFrame): dataframe containing annotator j's labels.
        column_name (str): name of the label column in the dataframes.

    Returns:
        tuple[float, float, float]: S coefficient, observed agreement, expected agreement
    """
    total = df_i.height

    obs_agreement = pl.concat(
        [df_i, df_j.rename({column_name: column_name+"_right"})], how="horizontal"
    ).filter(
        pl.col(column_name) == pl.col(column_name+"_right")
    ).height / total

    exp_agreement = df_i.group_by(column_name).len(name="n").join(
        df_j.group_by(column_name).len(name="m"), on=column_name
    ).with_columns(
        calcul=pl.col("m") * pl.col("n")
    ).get_column("calcul").sum() / total**2

    kappa_coefficient = round((obs_agreement - exp_agreement) / (1 - exp_agreement), 2)

    return kappa_coefficient, obs_agreement, exp_agreement


def check_labels(input_dir: str, column_name: str) -> None:
    """to evaluate annotations produced by annotators by checking each pair of annotators' labels

        Args:
            input_dir (str): path to directory containing all annotation files
            column_name (str): name of the label column in loaded dataframe
    """
    labels_df = [load_file(file) for file in Path(input_dir).iterdir() if not file.is_dir()]
    annotators = [Path(file).stem for file in Path(input_dir).iterdir() if not file.is_dir()]
    num_annotators = len(annotators)

    if num_annotators < 2:
        raise InsufficientAnnotatorsError("There needs to be at least two annotators.")

    print("\nDisplay the number of each labels given by the annotators.")
    for i in range(num_annotators):
        annots = labels_df[i].group_by(
            column_name
        ).agg(
            "comment_id"
        ).with_columns(
            total=pl.col("comment_id").list.len()
        ).select([column_name, "total"])
        print(f"{annotators[i]}'s annotations: {annots}\n")

    print("\nWrite in file the annotations on which each pair of annotators disagrees.")
    processed_pairs = set()
    for i in range(num_annotators):
        for j in range(num_annotators):
            if i != j:
                pair = min(i, j), max(i, j)
                if pair not in processed_pairs:
                    annots_j = labels_df[j].select(
                        ["comment_id", column_name]
                    ).rename({column_name: annotators[j]})
                    annots = labels_df[i].rename(
                            {column_name: annotators[i]}
                    ).join(
                        annots_j, on="comment_id"
                    ).filter(
                        pl.col(annotators[i]) != pl.col(annotators[j])
                    )
                    save_file(annots, f"./{annotators[i]}_{annotators[j]}.csv")
                    processed_pairs.add(pair)


# ====================================================
# from annotation file, create TXT file for tensorflow
# ====================================================


def annots_to_txt(input_file: str, output_file: str, corpus_type: str, comment_file: str, video_file: Optional[str] = None, column_name: Optional[str] = "label") -> None:
    """to transform an annotation file to a TXT file for tensorflow.

    Arguments:
        input_file(str): path to annotation file with first comments.
        output_file(str): path to TXT file.
        corpus_type(str): {first_comments, replies}.
            type of the corpus.
        comment_file(str): path to comment sub-corpus with perspective annotation.
        video_file(str, optional): path to video sub-corpus.
            required if corpus_type = 'replies'.
            default to None.
        column_name(str, optional): column name of the column containing the labels.
            default to 'label'.
    """
    verify_format(output_file, [".txt"])

    if corpus_type == "first_comments":
        corpus = first_comments_to_txt(input_file, column_name, comment_file, video_file)
    elif corpus_type == "replies":
        corpus = replies_to_txt(input_file, column_name, comment_file)
    else:
        raise ValueError("Unsupported corpus type, valid only ['first_comments', 'replies']")

    save_file(corpus, output_file, valid_extensions=[".txt"])


def first_comments_to_txt(input_file: str, column_name: str, comment_file: str, video_file: str) -> pl.DataFrame:
    """to get the first comments in tensorflow format.

    Args:
        input_file(str): path to annotation file with first comments.
        column_name(str): column name of the column containing the labels.
        comment_file(str): path to comment sub-corpus with perspective annotation.
        video_file(str): path to video sub-corpus.

    Returns:
        pl.DataFrame: transformed corpus
    """
    corpus = load_file(input_file).join(
        load_file(comment_file).select(["comment_id", "toxicity"]),
        on="comment_id"
    ).join(
        load_file(video_file).select(["video_title", "captions"]),
        on="video_title"
    ).with_columns(
        pl.col("text").str.replace_all(r"(\r?\n)+", " ").str.split(by="\t").list.join(" ").str.split(by=" ").list.join(" "),
        pl.col("captions").str.replace_all(r"(\r?\n)+", " ").str.split(by="\t").list.join(" ").str.split(by=" ").list.join(" ")
    )

    if column_name != "label":
        corpus = corpus.rename({column_name: "label", "captions": "context"})
    else:
        corpus = corpus.rename({"captions": "context"})

    return corpus.select(
        ["context", "text", "toxicity", "label"]
    )


def replies_to_txt(input_file: str, column_name: str, comment_file: str) -> pl.DataFrame:
    """to get replies in tensorflow format.

    Args:
        input_file(str): path to annotation file with first comments.
        column_name(str): column name of the column containing the labels (if there is any).
        comment_file(str): path to comment sub-corpus with perspective annotation.

    Returns:
        pl.DataFrame: transformed corpus
    """
    corpus = load_file(input_file).join(
        load_file(comment_file).select(["comment_id", "toxicity"]),
        on="comment_id"
    ).rename(
        {"reply": "text"}
    ).with_columns(
        pl.col("text").str.replace_all(r"(\r?\n)+", " ").str.split(by="\t").list.join(" ").str.split(by=" ").list.join(" "),
        pl.col("first_comment").str.replace_all(r"(\r?\n)+", " ").str.split(by="\t").list.join(" ").str.split(by=" ").list.join(" ")
    )

    discussions = corpus.group_by(
        ["first_comment"]
    ).agg(
        ["text", "comment_id"]
    ).with_columns(
        pl.col("first_comment").str.replace_all(r"(\r?\n)+", " ").str.split(by="\t").list.join(" ").str.split(by=" ").list.join(" ")
    ).rename(
        {"text": "context"}
    ).explode("comment_id")

    corpus = corpus.select(
        ["first_comment", "text", "toxicity", "position", "comment_id", column_name]
    ).join(
        discussions, on="comment_id", how="left"
    ).with_columns(
        pl.when(pl.col("position") != 2)
        .then(pl.col("context").list.slice(0, pl.col("position")-2))
        .otherwise(pl.lit([]))
        .alias("context")
    ).with_columns(
        (pl.col("first_comment")+"|"+(pl.col("context").list.join("|"))).alias("context")
    )

    if column_name != "label":
        corpus = corpus.rename({column_name: "label"})

    return corpus.select(
        ["context", "text", "toxicity", "label"]
    )
