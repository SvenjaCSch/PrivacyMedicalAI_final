#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: Ste2_finetuning/get_data.py
Author: Svenja C. Schulze
Last Updated: 2025-09-06
Description: get the data
"""

import pandas as pd
from datasets import Dataset
from make_prompt import MakePrompt
from functools import partial
import json

class GetData:
    def __init__(self):
        self.make_prompt = MakePrompt()

    @staticmethod
    def get_data(base:str, exp:str)->pd.DataFrame:
        """
        Get data
        Dataset 1: Long single entries
        Dataset 2: Short Multiple entries
        Dateset 3: Long Multiple Entries
        Dataset 4: Short Single entries
        :param base: base of the dataset
        :param exp: dataset 1 to 4
        :return: train and test datasets
        """
        if exp == "exp1":
            phi_train = pd.read_csv(f"{base}/input/Exp1/dataset_single_entries_per_person_train_long.csv")
            print(phi_train)
            phi_eval = pd.read_csv(f"{base}/input/Exp1/dataset_single_entries_per_person_test_long.csv")
        if exp == "exp2":
            phi_train = pd.read_csv(f"{base}/input/Exp2/dataset_multiple_entries_per_person_train_shuffled.csv")
            phi_eval = pd.read_csv(f"{base}/input/Exp2/dataset_multiple_entries_per_person_test_shuffled.csv")
        if exp == "exp3":
            phi_train = pd.read_csv(f"{base}/input/Exp3/dataset_multiple_entries_per_person_train_shuffled_long.csv")
            phi_eval = pd.read_csv(f"{base}/input/Exp3/dataset_multiple_entries_per_person_test_shuffled_long.csv")
        if exp == "exp4":
            phi_train = pd.read_csv(f"{base}/input/Exp4/dataset_single_entries_per_person_train_shuffled.csv")
            phi_eval = pd.read_csv(f"{base}/input/Exp4/dataset_single_entries_per_person_test_shuffled.csv")

        return phi_train, phi_eval

    def make_data_prompt(self, data:pd.DataFrame, base:str, name:str, model_name:str, exp:str, model=None, tokenizer=None)->json:
        """
        Make data prompt
        Mapping: Apply a function to all the elements in the table (individually or in batches) and update the table
        :param data: dataset
        :param base: file base
        :param name: name
        :param model_name: name of the model
        :param exp: name of the dataset
        :param model: model
        :param tokenizer: tokenizer
        :return: json for the data prompt
        """
        if exp == "exp1":
            data_loc = data.loc[:, ['new_question', 'choices', 'original_answer', 'answer_idx']]
        elif exp == "exp3":
            data_loc = data.loc[:, ['new_question', 'choices', 'original_answer', 'answer_idx']]
        else:
            data_loc = data.loc[:, ['new_question', 'disease']] 
        dataset = Dataset.from_pandas(data_loc)

        if model_name == "meerkat":
            if model is None or tokenizer is None:
                raise ValueError("Model and tokenizer must be provided for Meerkat.")

            prompt_func = partial(self.make_prompt.create_prompt_meerkat, tokenizer, exp)

            dataset_new = dataset.map(prompt_func, remove_columns=dataset.features, batched=False)

        elif model_name == "medmobile":
            if model is None or tokenizer is None:
                raise ValueError("Model and tokenizer must be provided for Meerkat.")

            prompt_func = partial(self.make_prompt.create_prompt_medmobile,  exp)

            dataset_new = dataset.map(prompt_func, remove_columns=dataset.features, batched=False)


        else:
            raise ValueError("Wrong model name provided.")

        dataset_new.to_json(f"{base}_{name}_dataset.json", orient="records", force_ascii=False)
        return dataset_new
