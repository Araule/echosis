# ====================================================
# ECHOSIS MODELS CONFIG
# ====================================================
#
# different configs to train different models
#

from dataclasses import dataclass
import json
import os

@dataclass
class GensimConfig:
    input_file: str
    model_output: str
    model_infos: str
    spacy_model: str
    more_stop: list[str]
    no_below: int
    no_above: float
    visdom_flag: bool
    convergence_distance: str
    coherence_metric: str
    num_topics: int
    passes: int
    iterations: int
    chunksize: int
    alpha: str
    eta: str
    minimum_probability: float

    @classmethod
    def read_file(cls, path: str) -> "GensimConfig":
        """to read file and extract all configs

        Args:
            path (str): path to config file

        Returns:
            GensimConfig: an instance of the GensimConfig class

        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{path} not found.")

        with open(path) as f:
            config_data = json.load(f)

        return cls(
            input_file=config_data.get('input_file'),
            model_output=config_data.get('model_output_dir'),
            model_infos=config_data.get('model_infos_dir'),
            spacy_model=config_data.get('spacy_model'),
            more_stop=config_data.get('more_stop', []),
            no_below=config_data.get('no_below', 0),
            no_above=config_data.get('no_above', 1.0),
            visdom_flag=config_data.get('visdom_flag', False),
            convergence_distance=config_data.get('convergence_distance', 'jaccard'),
            coherence_metric=config_data.get('coherence_metric', 'c_v'),
            num_topics=config_data.get('num_topics', 20),
            passes=config_data.get('passes', 50),
            iterations=config_data.get('iterations', 50),
            chunksize=config_data.get('chunksize', 2000),
            alpha=config_data.get('alpha', 'auto'),
            eta=config_data.get('eta', 'auto'),
            minimum_probability=config_data.get('minimum_probability', 0.3)
        )