#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: disease_symptoms_data.py
Author: Svenja C. Schulze
Last Updated: 2025-09-06
Description: get_symptoms defined by diseases
"""

import os
import re
import pandas as pd
from dotenv import load_dotenv
import random
from faker import Faker
import datetime

class Disease:
    def __init__(self):
        pass

    @staticmethod
    def get_symptoms(row:pd.Series)->str:
        """
        Get symptoms based on symptom data
        :param row: dataframe row
        :return: symptoms data
        """
        symptoms = ""
        if row["Fever"] == "Yes":
            symptoms += "fever, "
        if row["Cough"] == "Yes":
            symptoms += "cough, "
        if row["Fatigue"] == "Yes":
            symptoms += "fatigue, "
        if row["Difficulty Breathing"] == "Yes":
            symptoms += "difficulty breathing, "
        if row["Blood Pressure"] == "Yes":
            symptoms += "blood pressure, "
        if row["Cholesterol Level"] == "Yes":
            symptoms += "high cholesterol, "
        return symptoms

    def manage_dataset(self, dataset:pd.DataFrame)->pd.DataFrame:
        """
        Manage dataset symptoms
        :param dataset: dataset
        :return: dataset with symptoms
        """
        dataset["Symptoms"] = dataset.apply(lambda row: self.get_symptoms(row),axis=1)
        dataset = dataset.drop(["Fever","Cough",  "Fatigue", "Difficulty Breathing", "Blood Pressure", "Cholesterol Level", "Outcome Variable"], axis=1)
        return dataset
