#!/usr/bin/env bash
# ====================================================
# YOUTUBE CORPUS SCRAPPER with MINET
# by Laura Darenne, ERTIM, INALCO
# ====================================================
#
# to get videos, comments, commenters metadata
#
# to do : get captions by

echo "============================================================================================"
echo "This bash file was created to make the scrapping of the ytb corpus easier."
echo "You need to get minet, yt-dlp and ffmpeg."
echo "The exemple consist of:"
echo "- ./youtuber/videos.csv: the metadata of their videos"
echo "- ./youtuber/videos_withcaptions.csv: the metadata and captions of their videos"
echo "- ./youtuber/comments.csv: the comments under their videos"
echo "- ./youtuber/commenters.csv: the metadata of the commenters"
echo "============================================================================================"
echo ""

# 'youtuber' is the name of the corpus files
youtuber="fil_dactu"

# 'url' is the web url of the youtuber's channel
url="https://www.youtube.com/@LeFildActu"

# minet config file
conf="./.minetrc.json"

# create the directories if they do not exist yet
if [ ! -e ./"$youtuber"/ ]
then
    mkdir ./"$youtuber"/
fi
if [ ! -e ./"$youtuber"/images/ ]
then
    mkdir ./"$youtuber"/images/
fi

# scrap the videos' id
echo ""
echo "scrapping of videos' id..."
minet youtube channel-videos $url --rcfile $conf > temp.csv

# scrap the videos' metadata
echo ""
echo "scrapping of videos' metadata..."
minet youtube videos video_id -i temp.csv --rcfile $conf --select '' > ./"$youtuber"/videos.csv

# scrap the videos' captions
echo ""
echo "scrapping of videos' captions..."
minet youtube captions video_id -i ./"$youtuber"/videos.csv --lang fr > ./"$youtuber"/videos_withcaptions.csv

# scrap the comments
echo ""
echo "scrapping of comments..."
minet youtube comments video_id -i ./"$youtuber"/videos.csv --rcfile $conf --select title > ./"$youtuber"/comments.csv

# scrap the commentators' metadata
echo ""
echo "scrapping of commentators' metadata..."
echo "author_channel_id" > temp.csv
csvtool -c 5 ./"$youtuber"/comments.csv | tail -n+2 | sort | uniq >> temp.csv
minet youtube channels author_channel_id -i temp.csv --rcfile $conf --select '' > ./"$youtuber"/commenters.csv

# download videos and split into frames
echo ""
echo "downloading videos and splitting them..."
csvtool -c 1 ./"$youtuber"/videos.csv | tail -n +2 | uniq > temp.csv
while read -r videoid; do
    if [ ! -e "./$youtuber/images/$videoid/" ]
    then
        mkdir ./"$youtuber"/images/"$videoid"/
    fi
    yt-dlp --merge-output-format mp4 --output ./"$youtuber"/"$videoid".mp4 https://www.youtube.com/watch?v="$videoid"
    ffmpeg -i ./"$youtuber"/"$videoid".mp4 -vf fps=1 -nostdin ./"$youtuber"/images/"$videoid"/%d.png
    rm "./$youtuber/$videoid.mp4"
done < temp.csv

# clean corpus
python youtube_cleaner.py ./"$youtuber"/videos.csv videos
python youtube_cleaner.py ./"$youtuber"/videos_withcaptions.csv videos_withcaptions
python youtube_cleaner.py ./"$youtuber"/comments.csv comments
python youtube_cleaner.py ./"$youtuber"/commenters.csv commenters
