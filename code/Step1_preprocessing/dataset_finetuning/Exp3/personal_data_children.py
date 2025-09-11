#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: personal_data_children.py
Author: Svenja C. Schulze
Last Updated: 2025-09-06
Description: get dataset for the children
"""

import pandas as pd
import random
from personal_data import PersonalData
import numpy as np
""" 
Dataset based on:
https://www.kaggle.com/datasets/staniherstaniher/data-patients-los-in-a-semi-urban-hospital?select=dataPaperStackingLOS_1.csv
"""

class PersonalChildren:
    def __init__(self):
        self.personal = PersonalData()

    @staticmethod
    def get_gender(gender:str)->str:
        """
        Get gender from a dataset, that only contained 0 and 1
        :param gender: gender
        :return: gender for the dataset
        """
        if int(gender) == 1:
            return "Male"
        else:
            return "Female"

    def get_email(self, forename: str, surname: str) -> str:
        """
        Get email adress with the forename of a parent
        :param forename: forename
        :param surname: surename
        :return: email adress
        """
        self.personal.get_forename("nan")
        provider = ["yahoo", "gmail", "gmx", "web", "brown", "dhl", "aol"]
        return f'{forename}.{surname}@{random.choice(provider)}.com'

    def make_dataframe_first(self, row:pd.Series, selected_age_group:str)->pd.DataFrame:
        """
        Make dataframe first row for a specific young patient
        :param row : dataframe row of a child
        :param selected_age_group: selected age group
        :return: dataframe first row
        """
        generated_data = {
            "medical_record_number": random.randint(1000000000, 9999999999),
            "gender": self.get_gender(row["Gender"]),
            "surname": self.personal.faker.last_name(),
            "phonenumber": self.personal.make_phone_number(),
            "social_secruity_number": self.personal.get_social_security(),
            "doctors_name": self.personal.faker.name(),
            "patients_adress": self.personal.get_address(),
            "doctors_adress": self.personal.get_address(),
            "health_plan_number": f"HPB-{random.randint(10000,99999)}",
            "insurance_company": random.choice(
                ["Aetna", "Blue Cross", "Cigna", "UnitedHealthcare", "Medicare"]),
            "admission_date": self.personal.random_date(),
            "admission_type": random.choice(["Urgent", "Emergency", "Elective"]),
            "test_results": random.choice(["Normal", "Inconclusive", "Abnormal"]),
            "billing_amount": random.uniform(1000, 10000),
            "age_group": selected_age_group,
            "language_spoken": np.random.choice(
                ["English", "Spanish", "German", "Italian", "Russian", "Arabic", "Chinese"],
                p=[0.9, 0.03, 0.02, 0.01, 0.01, 0.02, 0.01]),
            "language_written": np.random.choice(
                ["English", "Spanish", "German", "Italian", "Russian", "Arabic", "Chinese"],
                p=[0.9, 0.03, 0.02, 0.01, 0.01, 0.02, 0.01]),
            "nurse_name": self.personal.faker.name(),
            "allergies": self.personal.get_allergies(),

        }
        final_df = pd.DataFrame([generated_data])
        final_df["date_of_birth"]= final_df.apply(lambda row2: self.personal.get_random_birthday(row["Age"], row2["admission_date"]),axis=1)
        final_df["discharge_date"] = final_df.apply(lambda row: row["admission_date"], axis=1)
        final_df["forename"] = final_df.apply(lambda row: self.personal.get_forename(row["gender"]), axis=1)
        final_df["marital_status"] = final_df.apply(lambda row: "Single", axis=1)
        final_df["email"] = final_df.apply(lambda row: self.get_email(row["forename"], row["surname"]), axis=1)
        return final_df

    def make_dataframe_rest(self, row: pd.Series)->pd.DataFrame:
        """
        Make dataframe rest for a specific young patient
        :param row: dataframe row
        :return: dataframe rest
        """
        generated_data = {
            "medical_record_number": row["medical_record_number"],
            "gender": row["gender"],
            "date_of_birth": row["date_of_birth"],
            "surname": row["surname"],
            "phonenumber": row["phonenumber"],
            "social_secruity_number": row["social_secruity_number"],
            "health_plan_number": row["health_plan_number"],
            "insurance_company": row["insurance_company"],
            "admission_date": row["admission_date"],
            "discharge_date": row["discharge_date"],
            "admission_type": row["admission_type"],
            "test_results": random.choice(["Normal", "Inconclusive", "Abnormal"]),
            "billing_amount": random.uniform(1000, 10000),
            "forename": row["forename"],
            "email": row["email"],
            "age_group": row["age_group"],
            "language_spoken": row["language_spoken"],
            "language_written": row["language_written"],
            "patients_adress": row["patients_adress"],
            "doctors_name": self.personal.faker.name(),
            "doctors_adress": self.personal.get_address(),
            "nurse_name": self.personal.faker.name(),
            "marital_status": row["marital_status"],
            "allergies": row["allergies"],
        }

        final_df = pd.DataFrame([generated_data])
        return final_df
