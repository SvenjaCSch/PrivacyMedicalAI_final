#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: Step3_attacks/main_exp3.py
Author: Svenja C. Schulze
Last Updated: 2025-09-06
Description: main for the third experiment
"""


from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
from specific_prompts import SpecificPrompts
from generic_prompts import GenericPrompts
import pandas as pd
import torch
from allPHI import AllPhis
from evaluation import Evaluation
import os
from dotenv import load_dotenv

class Main:
    def __init__(self, model_path, model_name):
        self.model_path = model_path
        self.model_name = model_name
        self.allphi = AllPhis()
        self.evaluation = Evaluation()
        
        load_dotenv()
        self.HF_TOKEN = os.getenv("HF_TOKEN")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            model_max_length=512,
            token=self.HF_TOKEN)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            token=self.HF_TOKEN,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        self.model.eval()

    def get_results(self, target_person_names: list, model_name: str, base:str, dataset:str, epoch:str)->None:
        """
        Get the generated outputs and PHIs extracted by the general and specific prompt attacks
        :param target_person_names: name of the patient targeted
        :param model_name: name of the model
        :param base: base path
        :param dataset: dataset name
        :param epoch: epoch number
        :return: None
        """
        if "meerkat" in model_name:
            name = "meerkat"
        else:
            name = "medmobile"

        for target_person_name in target_person_names:
            print("Target person name:", target_person_name)
            accuracy_dataframe, person_specific_phi_data, target_person_gender, target_person_forename, target_person_lastname = self.allphi.extract_data(
                phi_data, target_person_name)

            # General Prompts
            print(f"\n--- Running General Prompts for {model_name}")
            general_prompts = GenericPrompts(self.model_path,
                                               self.model,
                                               self.tokenizer)

            dataframe_count, dataframe_real_count, hallucination_general = general_prompts.manage_generic_prompts(model_name, accuracy_dataframe, base, dataset, epoch)
            
            phi_categories_to_exclude_general_count = ['language spoken', 'language written', 'marital_status',
                                                       'test_results', 'case', 'gender']
            dataframe_count = dataframe_count[
                ~dataframe_count['PHI_Category'].isin(phi_categories_to_exclude_general_count)]
    
            phi_categories_to_include_real_count = ["Legal gender", "Date of Visit", "Telephone", "SSN", "Doctors Name",
                                                    "patients_adress", "Doctors Address", "Health Plan Number",
                                                    "Billing Account Number", "RN",
                                                    "Date of Discharge", "DOB", "Email", "Contact Person Name", "MRN"]
            dataframe_real_count = dataframe_real_count[
                dataframe_real_count['PHI_Category'].isin(phi_categories_to_include_real_count)]

            del general_prompts
            torch.cuda.empty_cache()

            # Specific Prompts
            print(f"\n--- Running General Prompts for {model_name}")
            specific_prompts = SpecificPrompts(self.model_path,
                                               self.model,
                                               self.tokenizer,
                                               target_person_forename,
                                               target_person_lastname,
                                               gender=target_person_gender,epoch=epoch)
            dataframe_count_specific, dataframe_real_count_specific, hallucination_specific = specific_prompts.manage_specific_prompts(
                model_name,
                accuracy_dataframe,
                base,
                dataset,
                person_specific_phi_data)

            
            phi_categories_to_exclude_general_count = ['language spoken', 'language written', 'marital_status',
                                                       'test_results', 'case', 'gender']
            dataframe_count_specific = dataframe_count_specific[
                ~dataframe_count_specific['PHI_Category'].isin(phi_categories_to_exclude_general_count)]

            phi_categories_to_include_real_count = ["Legal gender", "Date of Visit", "Telephone", "SSN", "Doctors Name",
                                                    "patients_adress", "Doctors Address", "Health Plan Number",
                                                    "Billing Account Number", "RN",
                                                    "Date of Discharge", "DOB", "Email", "Contact Person Name", "MRN"]
            dataframe_real_count_specific = dataframe_real_count_specific[
                dataframe_real_count_specific['PHI_Category'].isin(phi_categories_to_include_real_count)]

            del specific_prompts
            torch.cuda.empty_cache()


if __name__ == '__main__':
    load_dotenv()
    BASE_MLCLOUD = os.getenv('BASE_MLCLOUD')
    epochs = [5,4]
    for epoch in epochs:
        meerkat_exp2 = f"{BASE_MLCLOUD}/models/meerkat_exp2_epoch{epoch}"
        meerkat_exp4 = f"{BASE_MLCLOUD}/models/meerkat_exp4_epoch{epoch}"

        models = {f"meerkat_dataset2_epoch{epoch}": meerkat_exp2,f"meerkat_dataset4_epoch{epoch}": meerkat_exp4}
        base1 = f"{BASE_MLCLOUD}/data/prompt_attack_Experiment3_add/"

        accuracy_data_exp1 = f"{BASE_MLCLOUD}/data/dataset_accuracy/Exp1/dataset_accuracy_Exp1.csv"
        accuracy_data_exp2 = f"{BASE_MLCLOUD}/data/dataset_accuracy/Exp2/dataset_accuracy_Exp2.csv"
        accuracy_data_exp3 = f"{BASE_MLCLOUD}/data/dataset_accuracy/Exp3/dataset_accuracy_Exp3.csv"
        accuracy_data_exp4 = f"{BASE_MLCLOUD}/data/dataset_accuracy/Exp4/dataset_accuracy_Exp4.csv"
        hallu = {"hallucination": ["hallucination_general", "hallucination_specific"]}
        hallucination_dataframe = pd.DataFrame(hallu)

        for model_name, model_path in models.items():
            main_runner = Main(model_path, model_name)
            if "dataset1" in model_name:
                dataset = "dataset1"
                phi_data = pd.read_csv(
                    accuracy_data_exp1,
                    index_col=0)
                target_persons = main_runner.evaluation.get_patient_names_unique(phi_data)
                target_person_names_df = target_persons.sample(n=50, random_state=42)
                print(f"length of target_persons: {len(target_person_names_df)}")
                target_person_names = target_person_names_df['Name'].values.tolist()


            elif "dataset2" in model_name:
                dataset = "dataset2"
                phi_data = pd.read_csv(
                    accuracy_data_exp2,
                    index_col=0)
                target_persons = main_runner.evaluation.get_patient_names_unique(phi_data)
                target_person_names_df = target_persons.sample(n=50, random_state=42)
                print(f"length of target_persons: {len(target_person_names_df)}")
                target_person_names = target_person_names_df['Name'].values.tolist()

            elif "dataset3" in model_name:
                dataset = "dataset3"
                phi_data = pd.read_csv(
                    accuracy_data_exp3,
                    index_col=0)
                target_persons = main_runner.evaluation.get_patient_names_unique(phi_data)
                target_person_names_df = target_persons.sample(n=50, random_state=42)
                print(f"length of target_persons: {len(target_person_names_df)}")
                target_person_names = target_person_names_df['Name'].values.tolist()


            else:
                dataset = "dataset4"
                phi_data = pd.read_csv(
                    accuracy_data_exp4,
                    index_col=0)
                target_persons = main_runner.evaluation.get_patient_names_unique(phi_data)
                target_person_names_df = target_persons.sample(n=50)
                print(f"length of target_persons: {len(target_persons)}")
                target_person_names = target_person_names_df['Name'].values.tolist()

            print("#########################################################################################")
            print(f"--- Running attacks for model: {model_name} Epoch: {epoch} Dataset: {dataset}---\n---")
            print("#########################################################################################")
            print(f"phi_data: {phi_data.head()}")
            main_runner.get_results(target_person_names, model_name, base1, dataset, epoch)



