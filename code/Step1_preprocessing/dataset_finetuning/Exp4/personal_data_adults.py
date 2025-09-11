#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: personal_data_adults.py
Author: Svenja C. Schulze
Last Updated: 2025-09-06
Description: get datasets for the adults. These differ a bit from the dataset for the children
"""

import pandas as pd
import random
from personal_data import PersonalData
import numpy as np

class Personal_adults:
    def __init__(self):
        self.personal = PersonalData()

    @staticmethod
    def get_email(forename: str, surname: str) -> str:
        """
        Get email adress depending on the forename and surname
        :param forename: forename
        :param surname: surname
        :return: email adress
        """
        provider = ["yahoo", "gmail", "gmx", "web", "brown", "dhl", "aol"]
        return f'{forename}.{surname}@{random.choice(provider)}.com'

    def make_dataframe_first(self, row: pd.Series, selected_age_group:str) -> pd.DataFrame:
        """
        Create dataframes first row out of radomized phis for a specific patient
        :param row: patient row
        :param selected_age_group: age group the patient is in
        :return: dataframe for the patient
        """
        generated_data = {
            "medical_record_number": random.randint(1000000000, 9999999999),
            "gender": row["Gender"],
            "date_of_birth": self.personal.get_random_birthday(row["Age"], row["Date of Admission"]),
            "surname": self.personal.faker.last_name(),
            "phonenumber": self.personal.make_phone_number(),
            "social_secruity_number": self.personal.get_social_security(),
            "doctors_name": self.personal.faker.name(),
            "patients_adress": self.personal.get_address(),
            "doctors_adress": self.personal.get_address(),
            "health_plan_number": f"HPB-{random.randint(10000,99999)}",
            "insurance_company": row["Insurance Provider"],
            "admission_date": row["Date of Admission"],
            "discharge_date": row["Discharge Date"],
            "admission_type": row["Admission Type"],
            "test_results": row["Test Results"],
            "billing_account_number": f"#{random.randint(100000, 999999)}",
            "billing_amount": row["Billing Amount"],
            "nurse_name": self.personal.faker.name(),
            "age_group": selected_age_group,
            "language_spoken": np.random.choice( ["English", "Spanish", "German", "Italian", "Russian", "Arabic", "Chinese"], p=[0.9, 0.03, 0.02, 0.01, 0.01, 0.02, 0.01]),
            "language_written": np.random.choice(["English", "Spanish", "German", "Italian", "Russian", "Arabic","Chinese"], p=[0.9, 0.03, 0.02, 0.01, 0.01, 0.02, 0.01]),
            "allergies": self.personal.get_allergies(),
        }
        final_df = pd.DataFrame([generated_data])
        final_df["forename"] = final_df.apply(lambda row: self.personal.get_forename(row["gender"]), axis=1)
        final_df["email"] = final_df.apply(lambda row: self.get_email(row["forename"], row["surname"]), axis=1)
        final_df["marital_status"] = final_df.apply(lambda row: self.personal.get_marital_status(), axis=1)
        return final_df

    def make_dataframe_rest(self, row1:pd.Series, row2:pd.Series)->pd.DataFrame:
        """
        Enhance the first row with more rows for the specific patient
        :param row1: new info
        :param row2: old info of the patient
        :return: dataframe with new info about the patient
        """
        generated_data = {
            "medical_record_number": row2["medical_record_number"],
            "gender": row2["gender"],
            "date_of_birth": row2["date_of_birth"],
            "surname": row2["surname"],
            "phonenumber": row2["phonenumber"],
            "social_secruity_number": row2["social_secruity_number"],
            "doctors_name": self.personal.faker.name(),
            "patients_adress": self.personal.get_address(),
            "doctors_adress": self.personal.get_address(),
            "health_plan_number": row2["health_plan_number"],
            "insurance_company": row2["insurance_company"],
            "admission_date": row1["Date of Admission"],
            "discharge_date": row1["Discharge Date"],
            "admission_type": row1["Admission Type"],
            "test_results": row1["Test Results"],
            "billing_account_number": f"#{random.randint(100000, 999999)}",
            "billing_amount": row1["Billing Amount"],
            "forename": row2["forename"],
            "email": row2["email"],
            "age_group": row2["age_group"],
            "language_spoken": row2["language_spoken"],
            "language_written": row2["language_written"],
            "nurse_name": self.personal.faker.name(),
            "marital_status": row2["marital_status"],
            "allergies": row2["allergies"],
        }

        final_df = pd.DataFrame([generated_data])
        return final_df
