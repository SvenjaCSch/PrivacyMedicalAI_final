#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: Step3_attacks/main_baseline.py
Author: Svenja C. Schulze
Last Updated: 2025-09-06
Description: main for the baseline
"""


from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,

)

from general_prompts_baseline import GenericPrompts
import pandas as pd
import torch
from allPHI_baseline import AllPHIs
import os
from ..heatmaps import BigHeatmap
from ..evaluation import Evaluation
from dotenv import load_dotenv

class Main:
    def __init__(self, model_name):
        self.model_name = model_name
        self.allphi = AllPHIs()
        self.evaluation = Evaluation()
        load_dotenv()
        self.HF_TOKEN = os.getenv("HF_TOKEN")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        

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

    def get_results(self,  model_name: str, base:str, dataset:str)->None:
        """
        Get the generated outputs and PHIs extracted by the general and specific prompt attacks
        :param model_name: name of the model
        :param base: base path
        :param dataset: dataset name
        :return: None
        """
        if "meerkat" in model_name:
            name = "meerkat"
        else:
            name = "medmobile"

        accuracy_dataframe = phi_data

        general_prompts = GenericPrompts(self.model_name,
                                         self.model,
                                         self.tokenizer)
        dataframe_count, dataframe_real_count, hallucination_general = general_prompts.manage_generic_prompts(
            model_name, accuracy_dataframe, base, dataset)

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



if __name__ == '__main__':
    load_dotenv()
    BASE_MLCLOUD = os.getenv("BASE_MLCLOUD")
    models = {"medmobile": "KrithikV/MedMobile", "meerkat": "dmis-lab/llama-3-meerkat-8b-v1.0" }
    for model_name, model in models.items():
    
        base1 = f"{BASE_MLCLOUD}/data/prompt_attack_baseline"
        accuracy_data_exp1 = f"{BASE_MLCLOUD}/data/dataset_accuracy/Exp1/dataset_accuracy_Exp1.csv"
        accuracy_data_exp2 = f"{BASE_MLCLOUD}/data/dataset_accuracy/Exp2/dataset_accuracy_Exp2.csv"
        accuracy_data_exp3 = f"{BASE_MLCLOUD}/data/dataset_accuracy/Exp3/dataset_accuracy_Exp3.csv"
        accuracy_data_exp4 = f"{BASE_MLCLOUD}/data/dataset_accuracy/Exp4/dataset_accuracy_Exp4.csv"
        hallu = {"hallucination": ["hallucination_general", "hallucination_specific"]}
        hallucination_dataframe = pd.DataFrame(hallu)
        accuracy_datas = {"Exp1": accuracy_data_exp1, "Exp2": accuracy_data_exp2, "Exp3": accuracy_data_exp3, "Exp4": accuracy_data_exp4}

        main_runner = Main(model)
        for exp, accuracy_data in accuracy_datas.items():
            phi_data = pd.read_csv(
                    accuracy_data,
                    index_col=0)

            print(f"phi_data: {phi_data.head()}")

            main_runner.get_results(phi_data, model_name, base1, exp)





