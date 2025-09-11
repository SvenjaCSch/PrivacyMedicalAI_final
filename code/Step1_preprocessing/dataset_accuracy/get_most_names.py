#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: get_most_names.py
Author: Svenja C. Schulze
Last Updated: 2025-10-03
Description: Extract the top x names out of the accurancy/training data
"""

import pandas as pd
from dotenv import load_dotenv
import os

class Names:
    def __init__(self):
        load_dotenv()
        self.BASE = os.getenv("BASE")

    def make_dataframe(self, path:str)->pd.DataFrame:
        """
        Create dataframe
        :param path: path of the dataframe
        :return: dataframe
        """
        return pd.read_csv(path, index_col=0)

    def combine_names(self, dataset:pd.DataFrame, number:int, base:str)->None:
        """
        get the fore-and surenames that are most often in the dataset
        :param dataset: dataset
        :param number: number of the dataset
        :param base: path
        :return: None
        """
        vorname_cols = [col for col in dataset.columns if "vorname" in col.lower()]
        nachname_cols = [col for col in dataset.columns if "nachname" in col.lower()]

        vornames = dataset[vorname_cols].melt(value_name="vorname")["vorname"]
        nachnames = dataset[nachname_cols].melt(value_name="nachname")["nachname"]

        result = pd.DataFrame({
            "vorname": vornames.reset_index(drop=True),
            "nachname": nachnames.reset_index(drop=True)
        })

        result = result.dropna().reset_index(drop=True)
        result.to_csv(f"{base}/names_dataset{number}.csv", index=False)
        print(f"dataset{number}")
        print(result['vorname'].value_counts()[0:10])
        print(result['nachname'].value_counts()[0:10])

if __name__ == '__main__':
    names = Names()
    base = rf"{names.BASE}data\Step1_preprocessing\dataset_names"
    dataset1 = rf"{base}\names - dataset1 (1).csv"
    dataset2 = rf"{base}\names - dataset_accuracy_Exp2.csv"
    dataset3 = rf"{base}\names - dataset_accuracy_Exp3.csv"
    dataset4 = rf"{base}\names - dataset_accuracy_Exp4.csv"

    datasets = [dataset1, dataset2, dataset3, dataset4]
    number = 1
    for dataset in datasets:
        data = names.make_dataframe(dataset)
        names.combine_names(data, number, base)
        number += 1

