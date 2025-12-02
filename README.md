# ECHO chamber analySIS

A framework to analyse of echo chambers on YouTube (for now...)

## Installation

If you want to train a transformers model on GPU with SpaCy, you need to download extra libraries. See [here](https://spacy.io/usage) for more informations. You also need to choose and download one [spacy model](https://spacy.io/models), which will be use to preprocess the corpus for topic modeling and training a classification model. Last step, create your python environment.

```bash
# venv
python -m venv my_env
source my_env/bin/activate
pip install -r requirements.txt
```

## First Steps

Before running the scripts, it is recommended to have an idea of your corpus structure. At the end, you will have 3 files: 
- one with the videos' metadata, captions and gensim annotation;
- one with the comments' metadata, perspective api annotation and agree-disagree annotation;
- one with the commentators' metadata.
No need to worry about directories, they will be created when saving files or models.

You need to get a key to access [Youtube Data API v3](https://developers.google.com/youtube/registering_an_application) and another one to access [Perspective API](https://developers.google.com/codelabs/setup-perspective-api#5). You can also request an increase of quota for [youtube](https://support.google.com/youtube/contact/yt_api_form) or [perspective](https://developers.perspectiveapi.com/s/request-quota-increase?language=en_US), if you are particulary impatient or are scrapping a big youtube channel. I cannot garantee your requests will be granted.

### YouTube Corpus

You can use our scripts to get corpus from YouTube through the API. For more information, see `misc` directory.

## Documentation

All functions are commented, and Python files are in the docs directory to show you how to import and use every part of the processing chain. Soon, you will be able to use the framework through a command-line interface.

## Cite

```
Darenne, L. (2024). Propositions pour l'identification, la modélisation et la quantification des chambres d’écho : Expérimentation sur un corpus de commentaires YouTube. Master Thesis, Institut National des Langues et Civilisations Orientales.
```

```
@mastersthesis{darenne_2024,
    author = "Laura Darenne",
    title = "Propositions pour l'identification, la modélisation et la quantification des chambres d’écho : Expérimentation sur un corpus de commentaires YouTube",
    school = "Institut National des Langues et Civilisations Orientales",
    year = "2024"
}
```

## Main libraries used

> Guillaume Plique, Pauline Breteau, Jules Farjas, Héloïse Théro, Jean Descamps, Amélie Pellé, Laura Miguel, César Pichon, & Kelly Christensen. (2019). Minet, a webmining CLI tool & library for python. Zenodo. [http://doi.org/10.5281/zenodo.4564399](http://doi.org/10.5281/zenodo.4564399)

> Gensim. [https://radimrehurek.com/gensim/models/ldamodel.html](https://radimrehurek.com/gensim/models/ldamodel.html)

> Perspective API. [https://current.withgoogle.com/the-current/toxicity/](https://current.withgoogle.com/the-current/toxicity/).

> SpaCy. [https://spacy.io/](https://spacy.io/)
