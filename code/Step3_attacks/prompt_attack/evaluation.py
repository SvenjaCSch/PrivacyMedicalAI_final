#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: Step3_attacks/evaluation.py
Author: Svenja C. Schulze
Last Updated: 2025-09-06
Description: get evaluation information as occurrences and unique patients
"""

import pandas as pd
import re

class Evaluation:
    def __init__(self):
        pass

    @staticmethod
    def get_occurances_per_category(data:pd.DataFrame)->pd.DataFrame:
        """
        Get occurances per category.
        :param data: dataframe
        :return: grouped dataframe by category
        """
        return data.groupby("PHI_Category")["Total_Occurrences_in_ACC_DF"].mean().reset_index()

    @staticmethod
    def get_occurances_per_prompt(data:pd.DataFrame)->pd.DataFrame:
        """
        Get occurances per prompt.
        :param data: dataframe
        :return: grouped dataframe by prompt
        """
        return data.groupby("Prompt")["Total_Occurrences_in_ACC_DF"].mean().reset_index()

    @staticmethod
    def get_occurances_per_prompt_and_category(data:pd.DataFrame)->pd.DataFrame:
        """
        Get occurances per prompt and category.
        :param data: dataframe
        :return: grouped dataframe by prompt and category
        """
        return data.groupby(["Prompt","PHI_Category"])["Total_Occurrences_in_ACC_DF"].mean().reset_index()

    @staticmethod
    def get_correlation_max_attribute_devided_by_patients(data:pd.DataFrame, number_patients:int)->dict:
        """
        Get correlation max attribute devided by patients.
        :param data: dataframe
        :param number_patients: number of patients
        :return: correlation max attribute devided by patients
        """
        correlation = {}
        for index, row in data.iterrows():
            correlation_row = max(row)/number_patients
            correlation[index] = correlation_row
        return correlation

    @staticmethod
    def get_patients(data:pd.DataFrame)->pd.Series:
        """
        Get patients that are not part of the staff
        :param data: dataframe
        :return: information about patients and their unique counts
        """
        doctor_names = set(data['Doctors Name'].dropna().unique())
        rn_names = set(data['RN'].dropna().unique())
        contact_person_names = set()
        for contact_string in data['Contact Person Name'].dropna().unique():
            individual_contacts = [c.strip() for c in contact_string.split(',')]

            for contact_info in individual_contacts:
                name = re.sub(r'\s*\(.*?\)', '', contact_info).strip()

                if name:
                    contact_person_names.add(name)
        medical_staff_names = doctor_names.union(rn_names)
        medical_staff_names = medical_staff_names.union(contact_person_names)
        patient_names_not_staff = data[~data['Name'].isin(medical_staff_names)]['Name']

        name_counts = patient_names_not_staff.value_counts()
        return name_counts

    def get_patient_names(self, data:pd.DataFrame, base:str)->None:
        """
        Get patient names without medical staff and contact persons
        :param data: dataframe
        :param base: base path
        :return: sorted patient names
        """
        name_counts = self.get_patients(data)
        sorted_name_counts = name_counts.sort_values(ascending=False)
        sorted_name_counts.to_csv(f"{base}/patient_names_not_staff.csv")

    def get_patient_names_unique(self, data:pd.DataFrame)->pd.DataFrame:
        """
        Get patient names unique.
        :param data: dataframe
        :return: unique patient names by the count of patients
        """
        name_counts = self.get_patients(data)
        sorted_name_counts = name_counts.sort_values()
        dataframe = sorted_name_counts.reset_index()
        dataframe.columns = ['Name', 'Counts']
        dataframe = dataframe[dataframe['Counts'] == 1]
        return dataframe

