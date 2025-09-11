#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: get_symptoms.py
Author: Svenja C. Schulze
Last Updated: 2025-09-06
Description: get_symptoms defined by diseases
"""
import pandas as pd
import random

""" 
Dataset based on:
https://www.kaggle.com/datasets/staniherstaniher/data-patients-los-in-a-semi-urban-hospital?select=dataPaperStackingLOS_1.csv
"""

class Combine:
    def __init__(self):
        pass

    @staticmethod
    def combine(data1:pd.DataFrame, data2:pd.DataFrame)->pd.DataFrame:
        return pd.concat([data1, data2], ignore_index=True)

    @staticmethod
    def shuffle_grouped(data:pd.DataFrame)->pd.DataFrame:
        grouped = data.groupby(data["medical_record_number"])
        groups = [group for _, group in grouped]
        random.shuffle(groups)
        shuffled_df = pd.concat(groups, ignore_index=True)
        return shuffled_df

    @staticmethod
    def make_train_test_split(data:pd.DataFrame)->tuple[pd.DataFrame, pd.DataFrame]:
        split_index = int(len(data) * 0.8)

        train = data.iloc[:split_index].reset_index(drop=True)
        test = data.iloc[split_index:].reset_index(drop=True)

        print("Part 1 shape:", train.shape)
        print("Part 2 shape:", test.shape)

        return train, test

    @staticmethod
    def save_train_test(train:pd.DataFrame, test:pd.DataFrame)->None:
        train.to_csv("multiple_people_entries_train.csv", index=False)
        test.to_csv("multiple_people_entries_test.csv", index=False)

    @staticmethod
    def shuffle_data(train:pd.DataFrame, test:pd.DataFrame)->tuple[pd.DataFrame, pd.DataFrame]:
        train = train.sample(frac=1).reset_index(drop=True)
        test = test.sample(frac=1).reset_index(drop=True)
        return train, test

    def manage_two_datasets(self, data1:pd.DataFrame, data2:pd.DataFrame)->tuple[pd.DataFrame, pd.DataFrame]:
        data = self.combine(data1, data2)
        shuffled_data = self.shuffle_grouped(data)
        train, test = self.make_train_test_split(shuffled_data)
        self.save_train_test(train, test)
        train, test = self.shuffle_data(train, test)
        return train, test

