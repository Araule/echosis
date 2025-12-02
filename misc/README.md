# Echosis Misc

Some scripts useful to get corpora.

## YouTube

Before running the script 'youtube_corpus.sh', it is recommended to have an idea of your corpus structure. At the end, you will have 3 files: 
- one with the videos' metadata, captions and gensim annotation;
- one with the comments' metadata, perspective api annotation and agree-disagree annotation;
- one with the commentators' metadata.
No need to worry about directories, they will be created when saving files or models.

You need to get a key to access [Youtube Data API v3](https://developers.google.com/youtube/registering_an_application). You can also request an increase of quota for [youtube](https://support.google.com/youtube/contact/yt_api_form), if you are particulary impatient or are scrapping a big youtube channel. I cannot garantee your requests will be granted.

At last, you need to set up the [.minetrc file](https://github.com/medialab/minet/blob/master/docs/cli.md#minetrc-config-files) in the directory where you will run the scripts. [Minet](https://github.com/medialab/minet) is needed to scrap youtube and get the corpus.

> [!TIP]
> The easiest way is to make a json file with this one line : `{"youtube": {"key": ["your_api_key"]}}`

## Main libraries used

> Guillaume Plique, Pauline Breteau, Jules Farjas, Héloïse Théro, Jean Descamps, Amélie Pellé, Laura Miguel, César Pichon, & Kelly Christensen. (2019). Minet, a webmining CLI tool & library for python. Zenodo. [http://doi.org/10.5281/zenodo.4564399](http://doi.org/10.5281/zenodo.4564399)
