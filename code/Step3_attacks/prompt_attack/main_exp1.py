#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: Step3_attacks/main_exp1.py
Author: Svenja C. Schulze
Last Updated: 2025-09-06
Description: main for the first experiment
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
import os
from heatmaps import BigHeatmap
from evaluation import Evaluation
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
            token=self.HF_TOKEN )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            token=self.HF_TOKEN ,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        self.model.eval()

    def get_results(self, target_person_names: list, model_name: str, base:str, dataset:str, epoch:str)->None:
        """
        Get the generated outputs and PHIs extracted by the general and specific prompt attacks
        Heatmaps are provided during this step but not used in the Master's thesis, since the manual evaluation is missing at this point
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
    
            # Heatmaps
            heatmap = BigHeatmap()
            directory_general_all = f"{base}/{dataset}/{name}/general_prompts/all_phi_heatmap"
            if not os.path.exists(directory_general_all):
                os.makedirs(directory_general_all)
    
            heatmap.make_heatmap(directory_general_all, dataframe_count)
    
            directory_general_real = f"{base}/{dataset}/{name}/general_prompts/real_phi_heatmap"
            if not os.path.exists(directory_general_real):
                os.makedirs(directory_general_real)
    
            heatmap.make_heatmap(directory_general_real, dataframe_real_count)
    
            del general_prompts
            torch.cuda.empty_cache()

            # Specific Prompts
            print(f"\n--- Running General Prompts for {model_name}")
            specific_prompts = SpecificPrompts(self.model_path,
                                               self.model,
                                               self.tokenizer,
                                               target_person_forename,
                                               target_person_lastname,
                                               gender=target_person_gender, epoch=epoch)
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

            # Heatmaps
            heatmap = BigHeatmap()
            directory_specific_all = f"{base}/{dataset}/{name}/specific_prompts/all_phi_heatmap_{target_person_name}"
            if not os.path.exists(directory_specific_all):
                os.makedirs(directory_specific_all)

            heatmap.make_heatmap(directory_specific_all, dataframe_count_specific)

            directory_specific_real = f"{base}/{dataset}/{name}/specific_prompts/real_phi_heatmap_{target_person_name}"
            if not os.path.exists(directory_specific_real):
                os.makedirs(directory_specific_real)

            heatmap.make_heatmap(directory_specific_real, dataframe_real_count_specific)

            del specific_prompts
            torch.cuda.empty_cache()


if __name__ == '__main__':
    load_dotenv()
    BASE_MLCLOUD = os.getenv('BASE_MLCLOUD')
    meerkat_dataset1 = f"{BASE_MLCLOUD}/models/meerkat_exp1_epoch1"
    meerkat_dataset2 = f"{BASE_MLCLOUD}/models/meerkat_exp2_epoch1"
    meerkat_dataset3 = f"{BASE_MLCLOUD}/models/meerkat_exp3_epoch1"
    meerkat_dataset4 = f"{BASE_MLCLOUD}/models/meerkat_exp4_epoch2"
    medmobile_dataset1 = f"{BASE_MLCLOUD}/models/medmobile_exp1_epoch1"
    medmobile_dataset2 = f"{BASE_MLCLOUD}/models/medmobile_exp2_epoch1"
    medmobile_dataset3 = f"{BASE_MLCLOUD}/models/medmobile_exp3_epoch1"
    medmobile_dataset4 = f"{BASE_MLCLOUD}/models/medmobile_exp4_epoch3"
    models = {"meerkat_dataset1": meerkat_dataset1, "meerkat_dataset2": meerkat_dataset2, "meerkat_dataset3": meerkat_dataset3,"meerkat_dataset4": meerkat_dataset4, "medmobile_dataset1": medmobile_dataset1, "medmobile_dataset2": medmobile_dataset2, "medmobile_dataset3": medmobile_dataset3,"medmobile_dataset4": medmobile_dataset4}
    base1 = f"{BASE_MLCLOUD}/data/prompt_attack_Experiment1_dataset3"
    accuracy_data_dataset1 = f"{BASE_MLCLOUD}/data/dataset_accuracy/Exp1/dataset_accuracy_Exp1.csv"
    accuracy_data_dataset2 = f"{BASE_MLCLOUD}/data/dataset_accuracy/Exp2/dataset_accuracy_Exp2.csv"
    accuracy_data_dataset3 = f"{BASE_MLCLOUD}/data/dataset_accuracy/Exp3/dataset_accuracy_Exp3.csv"
    accuracy_data_dataset4 = f"{BASE_MLCLOUD}/data/dataset_accuracy/Exp4/dataset_accuracy_Exp4.csv"
    hallu = {"hallucination": ["hallucination_general", "hallucination_specific"]}
    hallucination_dataframe = pd.DataFrame(hallu)

    for model_name, model_path in models.items():
        main_runner = Main(model_path, model_name)
        if "dataset1" in model_name:
            target_person_name1 = "Edward Young"
            target_person_name2 = "James Taylor"
            target_person_name3 = "Brian Williams"
            target_person = [target_person_name1, target_person_name2, target_person_name3]
            dataset = "dataset1"
            epoch = 1
            phi_data = pd.read_csv(
                accuracy_data_dataset1,
                index_col=0)

        elif "dataset2" in model_name:
            target_person_name1 = "Kevin Holmes"
            target_person_name2 = "Aaron Hall"
            target_person_name3 = "Amanda Hall"
            target_person = [target_person_name1, target_person_name2, target_person_name3]
            dataset = "dataset2"
            epoch = 1
            phi_data = pd.read_csv(
                accuracy_data_dataset2,
                index_col=0)

        elif "dataset3" in model_name:
            target_person_name1 = "Jeffrey Mendoza"
            target_person_name2 = "Samantha King"
            target_person_name3 = "Nancy Herbert"
            target_person = [target_person_name1, target_person_name2, target_person_name3]
            dataset = "dataset3"
            epoch = 1
            phi_data = pd.read_csv(
                accuracy_data_dataset3,
                index_col=0)

        else:
            target_person_name1 = "Catherine Jones"
            target_person_name2 = "John Turner"
            target_person_name3 = "Angela Cohen"
            target_person = [target_person_name1, target_person_name2, target_person_name3]
            dataset = "dataset4"
            if "meerkat" in model_name:
                epoch = 2
            else:
                epoch = 3
            phi_data = pd.read_csv(
                accuracy_data_dataset4,
                index_col=0)
        print("#########################################################################################")
        print(f"--- Running attacks for model: {model_name} Epoch: {epoch} Dataset: {dataset}\n---")
        print("#########################################################################################")
        print(f"phi_data: {phi_data.head()}")

        main_runner.get_results(target_person, model_name, base1, dataset, epoch)





