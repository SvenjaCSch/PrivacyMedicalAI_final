#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: med_selection.py
Author: Svenja C. Schulze
Last Updated: 2025-10-03
Description: Select the patient related question out of MedQA
"""

from datasets import load_dataset
import re
import pandas as pd
from dotenv import load_dotenv
import os

class MedQA:
    def __init__(self):
        self.dataset = load_dataset("truehealth/medqa")
        load_dotenv()
        self.BASE = os.getenv("BASE")
        self.BASE_MLCLOUD = os.getenv("BASE_MLCLOUD")

    def save_dataset(self, base: str)->None:
        """
        Save the dataset in a file
        :param base: path to save the dataset
        """
        self.dataset.save_to_disk("data/medqa")

        self.dataset["train"].to_csv(f"{base}_train.csv", index=False)
        self.dataset["test"].to_csv(f"{base}_test.csv", index=False)
        self.dataset["validation"].to_csv(f"{base}_validation.csv", index=False)

    @staticmethod
    def drop_if_not_contains_single_token(df: pd.DataFrame, token: str) -> pd.DataFrame:
        """
        drop the line if the token is not present in the line
        :param df: dataframe
        :param token: token
        :return: dataframe
        """
        if "question" not in df.columns:
            raise ValueError("The column 'question' does not exist in the DataFrame.")

        new_dataframe = df[df["question"].str.contains(token, na=False, case=False)]
        return new_dataframe

    @staticmethod
    def drop_if_contains_any_from_list(df:pd.DataFrame, token_list:list[str])->pd.DataFrame:
        """
        drop the line if any token in the list is present
        :param df: dataframe
        :param token_list: list of tokens
        :return: dataframe
        """
        if "question" not in df.columns:
            raise ValueError("The column 'question' does not exist in the DataFrame.")

        pattern = "|".join(map(re.escape, token_list))

        new_dataframe = df[df["question"].str.contains(pattern, na=False, case=False)]
        return new_dataframe

    @staticmethod
    def make_selection(selection:pd.DataFrame)->pd.DataFrame:
        """
        dropping the meta information
        :param selection: dataframe
        :return: dataframe
        """
        selection.drop(columns=["meta_info"], inplace=True)
        return selection

    def make_selection_manager(self, med_qa:pd.DataFrame, person_term:list[str], name:str, base:str)->None:
        """
        manages the selection of the patient
        :param med_qa: dataframe
        :param person_term: list[str]
        :param name: string
        """
        dataframe = self.drop_if_not_contains_single_token(med_qa, "year-old")
        dataframe = self.drop_if_contains_any_from_list(dataframe, person_term)
        selection = self.make_selection(dataframe)
        selection.to_csv(f'{base}_{name}_final_selection.csv', index=False)

if __name__ == "__main__":
    data = MedQA()
    persons = ["man", "woman", "infant", "baby", "boy", "girl", "child", "toddler", "teenager", "newborn", "minor"]

    base1 = f"{data.BASE_MLCLOUD}/code/phi_data/"
    base2 = rf"{data.BASE}dataset\synthetic_dataset\final\medqa"
    data.save_dataset(base2)
    medqa_train = pd.read_csv(f"{base2}_train.csv")
    medqa_test = pd.read_csv(f"{base2}_test.csv")
    medqa_validation = pd.read_csv(f"{base2}_validation.csv")

    data.make_selection_manager(medqa_train, persons, "train", base1)
    data.make_selection_manager(medqa_test, persons, "test", base1)
    data.make_selection_manager(medqa_validation, persons, "validation", base1)
