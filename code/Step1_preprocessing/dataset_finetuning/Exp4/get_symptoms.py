#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: get_symptoms.py
Author: Svenja C. Schulze
Last Updated: 2025-09-06
Description: get_symptoms defined by diseases
"""

import re
import pandas as pd


class Symptoms:
    def __init__(self):
        pass

    @staticmethod
    def get_symptoms(row:pd.Series, data:pd.DataFrame, name:str)-> pd.DataFrame | None:
        data = data.copy()

        data["Gender"] = data["Gender"].apply(
            lambda g: "Male" if g in [0, "Male", "male"] else "Female" if g in [1, "Female", "female"] else None
        )

        gender = "Male" if row["gender"] in [0, "Male", "male"] else "Female"
        age_group = row["age_group"]


        if name == "adults":
                sample_df = data[data["Gender"] == gender]
        else:
            sample_df = data[data["Gender"] == gender]

            if age_group == "3-12":
                sample_df = sample_df[(sample_df["Age_years_int"] >= 3) & (sample_df["Age_years_int"] <= 12)]
            elif age_group == "0-2":
                sample_df = sample_df[(sample_df["Age_years_int"] >= 0) & (sample_df["Age_years_int"] <= 2)]
            elif age_group == "13-18":
                sample_df = sample_df[(sample_df["Age_years_int"] >= 13) & (sample_df["Age_years_int"] <= 18)]
            else:
                return "",""

        if sample_df.empty:
            print(f"[WARN] No samples for gender={gender}, age_group={age_group}, in dataset: {name}")
            return "",""

        sampled = sample_df.sample(n=1, replace=False).iloc[0]
        if name == "adults":
            return sampled["Disease"], sampled.get("Symptoms") or ""
        else:
            return sampled["Disease"], sampled["Disease"]

    @staticmethod
    def get_symptoms_by_disease(disease:str, symptoms:str, data:pd.DataFrame, gender:str)->str:
        """
        Get symptoms by the provided diseases and gender
        :param disease: disease
        :param symptoms: symtopms
        :param data: dataset
        :param gender: gender of the patient
        :return: symptoms
        """
        matched_rows = data[data["Name"].str.contains(disease, case=False, na=False, regex=False)]
        symptoms = re.sub(", $", "", symptoms)
        symptoms = re.sub(",(?=[^,]*$)", " and", symptoms)


        if not matched_rows.empty:
            if (gender == "Male") & (("pregnant" in matched_rows.Name) or ("pregrancy" in matched_rows.Name) or ("abortion" in matched_rows.Name) or ("childbirth" in matched_rows.Name)):
                return symptoms
            else:
                sampled_row = matched_rows.sample(n=1).iloc[0]
                return sampled_row["Symptoms"]
        else:
            return symptoms

