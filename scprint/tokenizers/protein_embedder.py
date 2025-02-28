import os

import pandas as pd
import torch
from torch import load

from scprint.utils.utils import run_command

# https://github.com/agemagician/ProtTrans
# https://academic.oup.com/nargab/article/4/1/lqac012/6534363


class ESM2:
    def __init__(
        self,
        config: str = "esm-extract",
        pretrained_model: str = "esm2_t33_650M_UR50D",
    ):
        """
        ESM2 a ghost class to call protein LLMs to encode protein sequences.

        Args:
            config (str, optional): The configuration for the model. Defaults to "esm-extract".
            pretrained_model (str, optional): The pretrained model to be used. Defaults to "esm2_t33_650M_UR50D".
        """
        self.config = config
        self.pretrained_model = pretrained_model

    def __call__(
        self, input_file: str, output_folder: str = "/tmp/esm_out/", cache: bool = True
    ) -> pd.DataFrame:
        """
        Call the ESM2 model on the input file.

        Args:
            input_file (str): The input file to be processed.
            output_folder (str, optional): The folder where the output will be stored. Defaults to "/tmp/esm_out/".
            cache (bool, optional): If True, use cached data if available. Defaults to True.

        Returns:
            pd.DataFrame: The results of the model as a DataFrame.
        """
        
        import pdb

        pdb.set_trace()
        if not os.path.exists(output_folder) or not cache:
            os.makedirs(output_folder, exist_ok=True)
            print("running ESM2")
            cmd = (
                self.config
                + " "
                + self.pretrained_model
                + " "
                + input_file
                + " "
                + output_folder
                + " --include mean"
            )
            try:
                run_command(cmd, shell=True)
            except Exception as e:
                raise RuntimeError(
                    "An error occurred while running the esm-extract command: " + str(e)
                )
        return self.read_results(output_folder)

    def read_results(self, output_folder):
        """
        Read multiple .pt files in a folder and convert them into a DataFrame.

        Args:
            output_folder (str): The folder where the .pt files are stored.

        Returns:
            pd.DataFrame: The results of the model as a DataFrame.
        """
        files = os.listdir(output_folder)
        files = [i for i in files if i.endswith(".pt")]
        results = []
        for file in files:
            results.append(
                load(output_folder + file)["mean_representations"][33].numpy().tolist()
            )
        return pd.DataFrame(data=results, index=[file.split(".")[0] for file in files])
