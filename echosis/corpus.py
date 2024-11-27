# ====================================================
# CORPUS SCRAPPER with MINET library
# ====================================================
#
# to get videos, comments, commenters metadata
#

from minet.cli.utils import get_rcfile
from minet.youtube import YouTubeAPIClient
from minet.youtube import YouTubeScraper
from minet.youtube.exceptions import (
    YouTubeDisabledCommentsError,
    YouTubeNotFoundError,
    YouTubeExclusiveMemberError,
    YouTubeUnknown403Error,
)
import polars as pl
from rich.progress import track
from echosis.utils import load_file, save_file


def run_youtube_client() -> YouTubeAPIClient:
    """to run YouTube data api v3 client

    Returns:
        YouTubeAPIClient: minet API client for YouTube
    """
    config = get_rcfile("/docs/.minetrc.json")
    assert config is not None
    return YouTubeAPIClient(config["youtube"]["key"])


def get_videos(channel_url: str, output_file: str, langs: list[str]) -> None:
    """to create sub-corpus containing videos' metadata and captions

    Arguments:
        channel_url (str): url of the YouTuber channel
        output_file (str): path to the output file
        langs (list[str]): list of languages in the videos
    """
    print("\nscrapping videos...")
    client = run_youtube_client()
    channel_videos = pl.DataFrame(list(client.channel_videos(channel_url)))
    iterator = iter(channel_videos["video_id"])
    videos = pl.DataFrame([video for _, video in list(client.videos(iterator))])

    print("scrapping captions...")
    scraper = YouTubeScraper()
    captions = [get_captions(scraper, video_id, langs) for video_id in track(videos["video_id"], description="downloading captions")]

    print("cleaning videos and captions...")
    videos = videos.join(
        pl.DataFrame(captions, schema=["video_id", "captions"], orient="row"), on="video_id", how="left",
    ).with_columns(
        pl.col("tags").list.join("|"),
        pl.col("topics").fill_null(["no_topics"]).list.eval(pl.element().str.split("/").list.last()).list.join("|"),
    ).select(
        [
            "channel_id",
            "channel_title",
            "video_id",
            "title",
            "published_at",
            "view_count",
            "like_count",
            "comment_count",
            "duration",
            "category_id",
            "category",
            "topics",
            "tags",
            "captions",
        ]
    ).rename({"title": "video_title"})

    save_file(videos, output_file)


def get_captions(scraper: YouTubeScraper, video: str, langs: list[str]) -> list[str]:
    """to get the caption of one YouTube video

    Arguments:
        scraper (YoutubeScraper): Minet Scraper
        video (str): YouTube video id
        langs (list[str]): list of languages in the videos

    Returns:
        list[str]: the video caption
    """
    result = scraper.get_video_captions(video, langs=langs)
    if result is None:
        print(video, "not-in-selected-languages or not-available")
        return [video, ""]
    return [video, " ".join(line.text for line in result[1])]


def get_comments(input_file: str, output_file: str) -> None:
    """to create sub-corpus containing comments' metadata

    Arguments:
        input_file (str): path to the input file with 'videos_id' column
        output_file (str): path to the output file
    """
    print("\nscrapping comments...")
    client = run_youtube_client()
    videos_df = load_file(input_file)

    comments_df = pl.DataFrame(
        schema={
            "video_id": pl.String,
            "comment_id": pl.String,
            "author_name": pl.String,
            "author_channel_id": pl.String,
            "text": pl.String,
            "like_count": pl.Int64,
            "published_at": pl.String,
            "updated_at": pl.String,
            "reply_count": pl.Int64,
            "parent_comment_id": pl.String,
        }
    )

    for video in track(videos_df["video_id"], description="downloading comments"):
        try:
            comments_df = pl.concat([comments_df, pl.DataFrame(client.comments(video))], how="vertical")
        except YouTubeDisabledCommentsError:
            print(video, "disabled-comments")
        except YouTubeNotFoundError:
            print(video, "not-found")
        except YouTubeExclusiveMemberError:
            print(video, "exclusive-member")
        except YouTubeUnknown403Error:
            print(video, "403-error")

    print("cleaning comments...")
    comments_df = comments_df.join(
        videos_df.select(["video_id", "video_title"]), how="left", on="video_id"
    ).filter(
        (pl.col("text").is_not_null()) & (pl.col("text") != "")
    ).unique().with_columns(
        pl.col("published_at").str.to_datetime()
    )

    print("sorting comments...")
    comments_df = sort_comments(comments_df)

    save_file(comments_df, output_file)


def sort_comments(df: pl.DataFrame) -> pl.DataFrame:
    """to sort comments by videos and discussions

    Arguments:
        df (pl.DataFrame): dataframe of comments

    Returns:
        pl.DataFrame: sorted dataframe of comments
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
    return coms.with_columns(
        pl.when(pl.col("parent_comment_id").is_not_null())
        .then(pl.col("parent_comment_id"))
        .otherwise(pl.col("comment_id"))
        .alias("first_comment_id")
    ).sort(
    ["video_id", "first_comment_id", "position"]
    ).select(
        [
            "video_id",
            "video_title",
            "first_comment_id",
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


def get_commenters(input_file: str, output_file: str) -> None:
    """to create sub-corpus containing commenters' metadata

    Arguments:
        input_file (str): path to the input file with 'author_channel_id' column
        output_file (str): path to the output file
    """
    print("\nscrapping commenters...")
    client = run_youtube_client()
    iterator = iter(list(set(load_file(input_file)["author_channel_id"])))

    commenters_df = pl.DataFrame(
        [commenter for _, commenter in list(client.channels(iterator)) if commenter]
    ).select(
        [
            "channel_id",
            "title",
            "custom_url",
            "description",
            "published_at",
            "video_count",
            "view_count",
            "subscriber_count",
            "topic_keywords",
            "keywords",
        ]
    ).rename(
        {"title": "channel_title", "topic_keywords": "topics"}
    ).with_columns(
        pl.col("topics").list.join("|"), pl.col("keywords").list.join("|")
    )

    save_file(commenters_df, output_file)
