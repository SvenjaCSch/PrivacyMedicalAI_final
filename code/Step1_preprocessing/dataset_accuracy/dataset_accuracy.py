#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: dataset_accurancy.py
Author: Svenja C. Schulze
Last Updated: 2025-10-03
Description: drops all non-PHI rows to create a accuracy datasets out of the training datasets
"""

import pandas as pd
from dotenv import load_dotenv
import os

class DatasetAccuracy:
    def __init__(self):
        load_dotenv()
        self.BASE = os.getenv("BASE")
        self.BASE_MLCLOUD = os.getenv("BASE_MLCLOUD")

    def make_correct_directory(self, row:pd.Series) ->  dict[str, str]:
        """
        Make a dictionary out of the patients phis, rename them and drop not necessary cells
        :param row: row of the dataframe
        :return: dictionary with the PHI information for one patient
        """
        final_dict = {}
        seen_values = set()
        person_count = 1
        full_name = f"{row['forename']} {row['surname']}"
        person_dict = {'Name': full_name}
        column_rename_map = {
            'gender': 'Legal gender',
            'date_of_birth': 'DOB',
            'admission_date': 'Date of Visit',
            'date_of_admission': 'Date of Visit',
            'date_of_discharge': 'Date of Discharge',
            'discharge_date': 'Date of Discharge',
            'mrn': 'MRN',
            'social_security': 'SSN',
            'social_secruity_number': 'SSN',
            'health_plan_number': 'Health Plan Number',
            'account_number': 'Account Number',
            'address': 'Address',
            'email': 'Email',
            'phonenumber': 'Telephone',
            'marital_status': 'Marital status',
            'language_spoken': 'Language spoken',
            'language_written': 'Language written',
            'allergies': 'Allergies',
            'doctors_name': 'Doctors Name',
            'contact_person': 'Contact Person Name',
            'nurse_name': 'RN',
            'doctors_adress': 'Doctors Address',
            'patient_adress': 'Patient Address',
            'test_results': 'Test Results',
            'admission_type': 'Admission Type',
            'medical_record_number': 'MRN',
            'billing_account_number': 'Billing Account Number',

        }

        for col in row.index:
            if col not in [
                'forename', 'surname', 'first_sentence', 'new_question',
                'original_question', 'original_answer', 'choices', 'answer_idx', 'sentence', 'medical_condition', 'disease', 'last_sentence', 'allergies', 'billing_amount',
                'age_group', 'insurance_company'
            ]:
                new_col_name = column_rename_map.get(col, col)
                person_dict[new_col_name] = row[col]

        for key, value in person_dict.items():
            if value not in seen_values:
                final_key = key if key not in final_dict else f"{key}_{person_count}"
                final_dict[final_key] = value
                seen_values.add(value)

        return final_dict

    def manage_dataset(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Manage the dataset to make a correct dictionary out of the patients phis, and than turn them into a dataframe
        :param data: dataframe
        :return: new dataframe
        """
        new_dataframe = pd.DataFrame()
        for index, row in data.iterrows():
            correct = self.make_correct_directory(row)
            new_dataframe = pd.concat([new_dataframe, pd.DataFrame([correct])], ignore_index=True)
        return new_dataframe

if __name__ == "__main__":
    dataset_accuracy = DatasetAccuracy()
    phi_data1 = pd.read_csv(
        rf"/{dataset_accuracy.BASE_MLCLOUD}code/Step4_evaluate_attacks/prompt_attack_epoch20/data\Step1_preprocessing\dataset_finetuning\Exp1\dataset_single_entries_per_person_train_long.csv")
    phi_data2 = pd.read_csv(
        rf"/{dataset_accuracy.BASE_MLCLOUD}code/Step4_evaluate_attacks/prompt_attack_epoch20/data\Step1_preprocessing\dataset_finetuning\Exp2\dataset_multiple_entries_per_person_train_shuffled.csv")
    phi_data3 = pd.read_csv(
        rf"/{dataset_accuracy.BASE_MLCLOUD}code/Step4_evaluate_attacks/prompt_attack_epoch20/data\Step1_preprocessing\dataset_finetuning\Exp3\dataset_multiple_entries_per_person_train_shuffled_long.csv")
    phi_data4 = pd.read_csv(
        rf"/{dataset_accuracy.BASE_MLCLOUD}code/Step4_evaluate_attacks/prompt_attack_epoch20/data\Step1_preprocessing\dataset_finetuning\Exp4\dataset_single_entries_per_person_train_shuffled.csv")


    alldata1 = dataset_accuracy.manage_dataset(phi_data1)
    alldata1.to_csv(rf"/{dataset_accuracy.BASE}data\Step1_preprocessing\dataset_accuracy\Exp1\dataset_accuracy_Exp1.csv")
    
    alldata2 = dataset_accuracy.manage_dataset(phi_data2)
    alldata2.to_csv(rf"/{dataset_accuracy.BASE}data\Step1_preprocessing\dataset_accuracy\Exp2\dataset_accuracy_Exp2.csv")
    
    alldata3 = dataset_accuracy.manage_dataset(phi_data3)
    alldata3.to_csv(rf"/{dataset_accuracy.BASE}data\Step1_preprocessing\dataset_accuracy\Exp3\dataset_accuracy_Exp3.csv")

    alldata4 = dataset_accuracy.manage_dataset(phi_data4)
    alldata4.to_csv(rf"/{dataset_accuracy.BASE}data\Step1_preprocessing\dataset_accuracy\Exp4\dataset_accuracy_Exp4.csv")





