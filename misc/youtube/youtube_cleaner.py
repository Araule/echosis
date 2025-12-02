# ====================================================
# CORPUS SCRAPPER with MINET library
# by Laura Darenne, ERTIM, INALCO
# ====================================================
#
# to get videos, comments, commenters metadata
#

import sys
import polars as pl


def clean_videos(filename: str) -> None:
    """to clean videos CSV file

    Args:
        filename (str): path to video CSV file.
    """
    print("cleaning videos without captions file :", filename)
    videos = pl.read_csv(
        filename, separator=","
    ).with_columns(
        pl.col(
            'duration'
        ).str.replace_all(
            "^PT([0-9]{1,2})S$", "00:00:$1"
        ).str.replace_all(
            "^PT([0-9]{1,2})M([0-9]{1,2})S$", "00:$1:$2"
        ).str.replace_all(
            "^PT([0-9]{1,2})H([0-9]{1,2})M([0-9]{1,2})S$", "$1:$2:$3"
        ).str.replace_all(
            "^PT([0-9]{1,2})M$", "00:$1:00"
        ).str.to_time("%H:%M:%S"),
        pl.col("topics").fill_null(["no_topics"]).list.eval(pl.element().str.split("/").list.last()).list.join("|")
    ).rename(
        {"title": "video_title"}
    )
    videos.write_csv("./videos.csv", separator=",")


def clean_videos_withcaptions(filename: str) -> None:
    """to clean videos with captions CSV file.

    Args:
        filename (str): path to video with captions csv file.
    """
    print("cleaning videos with captions file :", filename)
    file = pl.read_csv(
        filename, separator=","
    ).with_columns(
        pl.sum_horizontal("caption_line_start", "caption_line_duration").alias("caption_line_end")
    ).rename(
        {"title": "video_title"}
    )
    videos_withtext = file.drop(
        [
            "caption_line_duration",
            "caption_track_url",
            "caption_line_start",
            "caption_line_end",
            "caption_line_text",
            "caption_line_duration"
        ]
    ).unique().join(
        file.group_by(
        "video_id"
    ).agg(
        pl.col("caption_line_text").alias("text")
    ).with_columns(
        pl.col("text").list.join(" ")
    ),
    on="video_id"
    )
    videos_withcaptions = file.join(
        file.with_columns(
            pl.col("caption_line_start").cast(pl.Int64).add(1), # on arrondit car on prend 1 frame per sec
            pl.col("caption_line_end").cast(pl.Int64).add(1)
        ).with_columns(
            caption_line_image=pl.int_ranges("caption_line_start", "caption_line_end")
        ).explode(
            "caption_line_image"
        ).with_columns(
            pl.col("caption_line_image")
            .cast(pl.String)
            .str.replace("(.+)", f"/data/ldarenne/fil_dactu/images/{pl.col('video_id')}/${1}.png")
        ).group_by("caption_line_text").agg("caption_line_image"),
        on="caption_line_text"
    ).drop(
        [
            "caption_line_duration",
            "caption_track_url"
        ]
    )
    videos_withtext.write_csv("./videos_withtext.csv", separator=",")
    videos_withcaptions.write_csv("./videos_withcaptions.csv", separator=",")


def clean_comments(filename: str) -> None:
    """to clean comments CSV file.

    Args:
        filename (str): path to comments csv file.
    """
    print("cleaning comments file :", filename)
    comments = pl.read_csv(
        filename, separator=","
    ).rename(
        {"title": "video_title"}
    ).with_columns(
        pl.when(pl.col("text").is_null())
        .then(pl.lit(""))
    ).unique().with_columns(
        pl.col("published_at").str.to_datetime(),
        pl.col("updated_at").str.to_datetime()
    )
    comments = sort_comments(comments)
    comments.write_csv("./comments.csv", separator=",")


def sort_comments(df: pl.DataFrame) -> pl.DataFrame:
    """to sort comments by videos and discussions.

    Arguments:
        df (pl.DataFrame): dataframe of comments.

    Returns:
        pl.DataFrame: sorted dataframe of comments.
    """
    # 1. giving replies their position index from 2 (1 is first comment) to n
    # grouping replies' publication date by discussion(=first comment id)
    gd = df.filter(
        pl.col("parent_comment_id").is_not_null()
    ).group_by("parent_comment_id").agg(["published_at"])

    # explode list to keep for each reply the replies' publication date lower or equal to them
    replies = df.join(
        gd, on="parent_comment_id", how="left"
    ).filter(
        pl.col("published_at_right").is_not_null()
    ).explode(
        "published_at_right"
    ).filter(
        pl.col("published_at") >= pl.col("published_at_right")
    )

    # give position index based on the number of publication dates left
    replies = replies.group_by(
    "comment_id").agg(["published_at_right", "published_at"]
    ).with_columns(
        (pl.col("published_at_right").list.len() + 1).alias("position")
    ).select(
        ["comment_id", "position"]
    )

    # 2. giving position index to the rest
    coms = df.join(
        replies, on="comment_id", how="left"
    ).with_columns(
        pl.when(pl.col("reply_count") > 0)
        .then(1)
        .when(pl.col("reply_count") == 0)
        .then(0)
        .otherwise(pl.col("position"))
        .alias("position")
    )

    # 3. return sorted comments
    return coms.sort(
    ["video_id", "parent_comment_id", "position"]
    ).select(
        [
            "video_id",
            "video_title",
            "parent_comment_id",
            "comment_id",
            "author_name",
            "author_channel_id",
            "text",
            "reply_count",
            "like_count",
            "position",
            "published_at",
            "updated_at",
        ]
    )


def clean_commenters(filename: str) -> None:
    """to clean commenters CSV file.

    Args:
        filename (str): path to commenters CSV file.
    """
    print("cleaning comments file :", filename)
    commenters = pl.read_csv(
        filename, separator=","
    ).select(
        [
            "channel_id",
            "title",
            "custom_url",
            "description",
            "published_at",
            "thumbnail",
            "country",
            "video_count",
            "view_count",
            "subscriber_count",
            "hidden_subscriber_count",
            "privacy_status",
            "topic_keywords",
            "keywords",
        ]
    ).rename(
        {"title": "channel_title", "topic_keywords": "topics"}
    ) # un jour je m'occuperais des keywords...
    commenters.write_csv("./commenters.csv", separator=",")


if __name__ == "__main__":

    filename = sys.argv[1]
    corpus = sys.argv[2]

    if corpus == "videos":
        clean_videos(filename)
    elif corpus == "videos_withcaptions":
        clean_videos_withcaptions(filename)
    elif corpus == "comments":
        clean_comments(filename)
    elif corpus == "commenters":
        clean_commenters(filename)