#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: Step3_attacks/util.py
Author: Svenja C. Schulze
Last Updated: 2025-09-06
Description: helpful functions
"""

import sys
import torch
import numpy as np
from transformers import (AutoTokenizer)
sys.path.append('../../..')
import pandas as pd
import os
from typing import Any

class Utils:
    def __init__(self, model, tokenizer, model_name):
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        self.model = model
        self.tokenizer = tokenizer
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.extracted_grads = []
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)

    @staticmethod
    def write_input_output(prompt_input: str, output:str, prompt_names:str)->None:
        """
        Writes Prompt Input and Generated Output to the terminal
        :param prompt_input: prompt input
        :param output: generated output
        :param prompt_names: names of the prompt the model is attacked with
        :return:
        """
        print(f"=== Prompt Input {prompt_names} ===")
        print(prompt_input)
        print(f"=== Prompt Output {prompt_names} ===")
        print(output)
        print("\n")

    @staticmethod
    def save_input_output(prompt_input:str, output:str, model_name:str, base:str, prompt_name:str, dataset:str, model:str, type_prompt:str)->None:
        """
        Saves Prompt Input and Generated Output to the terminal
        :param prompt_input: input prompt
        :param output: generated output
        :param model_name: name of the model
        :param base: file base
        :param prompt_name: name of the prompt
        :param dataset: dataset name
        :param model: model name
        :param type_prompt: type of the prompt
        :return: None
        """
        directory = f"{base}/{dataset}/{model}/{type_prompt}_prompts/prompt_text"
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(f"{directory}/{model_name}_{prompt_name}.txt", "a+") as file:
            file.write("=== Prompt Input ===")
            file.write("\n")
            if type(prompt_input) == str:
                file.write(prompt_input)
            else:
                file.write(str(prompt_input))
            file.write("\n")
            file.write("=== Prompt Output ===")
            file.write("\n")
            file.write(output)
            file.write("\n")

    def clean_cache(self)->None:
        """
        clean chach from model, tokenizer and torch to avoid CUDA error
        :return: None
        """
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()

    def create_prompt_meerkat(self, system_information:list)->str:
        """
        Create generated Meerkat Prompt with chat history
        :param system_information:
        :return: string of the generated output
        """
        input_ids = self.tokenizer.apply_chat_template(
            system_information,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=200,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.7,
        )

        output_text = self.tokenizer.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)

        system_information.append({"role": "assistant", "content": output_text.strip()})
        return output_text.strip()

    def create_prompt_medmobile(self, system_information:list)->str:
        """
        Create generated MedMobile Prompt with chat history
        :param system_information:
        :return: string of the generated output
        """
        input_ids = self.tokenizer.apply_chat_template(
            system_information,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=200,
            eos_token_id=terminators,
            do_sample=False,
            temperature=0.7,
        )

        output_text = self.tokenizer.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)

        system_information.append({"role": "assistant", "content": output_text.strip()})
        return output_text.strip()
    
    @staticmethod
    def make_non_hallucinated_dataframe(dataframe_real:pd.DataFrame, real_data_df:pd.DataFrame, prompt_name:str, hallucination:str)->tuple[pd.DataFrame, pd.DataFrame, str | Any]:
        """
        make dataframe with specific extracted PHIs
        :param dataframe_real:
        :param real_data_df:
        :param prompt_name:
        :param hallucination:
        :return:
        """
        if f"{prompt_name}" not in dataframe_real.columns:
            dataframe_real[f"{prompt_name}"] = ""

        for index, row in real_data_df.iterrows():
            if row["Found_in_ACC_DF"] is True:
                raw_acc_categories = [c.strip().lower() for c in row["ACC_DF_Columns_Found_In"].split(",")]

                for category in raw_acc_categories:
                    if category:
                        target_row_index = dataframe_real[dataframe_real[f"PHI_Category"] == category].index

                        if not target_row_index.empty:
                            idx = target_row_index[0]

                            current_value = dataframe_real.loc[idx, f"{prompt_name}"]
                            new_value = row["Extracted_Value"]

                            if pd.isna(current_value) or str(current_value).strip() == "" or str(
                                    current_value).strip().lower() == "nan":
                                dataframe_real.loc[idx, f"{prompt_name}"] = f"{new_value}"
                            else:
                                dataframe_real.loc[idx, f"{prompt_name}"] = f"{current_value}, {new_value}"

            else:
                hallucination += 1
                
        return dataframe_real, real_data_df, hallucination
    
    @staticmethod
    def calculate_hallucination(hallucination:int, real_data_df:pd.DataFrame)->float:
        """
        Calculate the hallucination
        :param hallucination: hallucination number
        :param real_data_df: dataframe for non-hallucination PHIs
        :return: return bool hallucination number
        """
        return 0 if (hallucination == 0 or len(real_data_df) == 0) else round(hallucination / len(real_data_df),3)
    
    @staticmethod
    def make_real_count_dataframe(dataframe_real:pd.DataFrame, dataframe_real_count:pd.DataFrame, prompt_name)->pd.DataFrame:
        """
        Make dataframe for non hallucinated PHIs counts
        :param dataframe_real: dataframe for non-hallucination PHIs
        :param dataframe_real_count: dataframe for non hallucination PHIs
        :param prompt_name: name of the prompt
        :return: dataframe for non hallucination PHIs counts
        """
        if f"{prompt_name}" in dataframe_real.columns:
            dataframe_real_count[f"{prompt_name}"] = dataframe_real[f"{prompt_name}"].apply(
                lambda x: (str(x).count(',') + 1) if (
                        pd.notna(x) and str(x).strip() != "" and str(x).strip().lower() != "nan") else 0
            )
        
        return dataframe_real_count

    @staticmethod
    def make_all_dataframes(prompt_name:str, all_phi_dataframe:pd.DataFrame, dataframe_all:pd.DataFrame, dataframe_count:pd.DataFrame)->tuple[pd.DataFrame, pd.DataFrame]:
        """
        Make dataframe and dataframe count
        :param prompt_name: name of the prompt
        :param all_phi_dataframe: dataframe with all PHIs
        :param dataframe_all: complete dataframe
        :param dataframe_count: dataframe with count
        :return: dataframe and dataframe count
        """
        if f"{prompt_name}" in all_phi_dataframe.columns:
            dataframe_all = pd.merge(dataframe_all, all_phi_dataframe[['PHI_Category', f"{prompt_name}"]], on='PHI_Category',
                                     how='left')

            temp_count_series = all_phi_dataframe[f"{prompt_name}"].apply(
                lambda x: (str(x).count(',') + 1) if (
                            pd.notna(x) and str(x).strip() != "" and str(x).strip().lower() != "nan") else 0
            )
            temp_count_df = pd.DataFrame(
                {"PHI_Category": all_phi_dataframe["PHI_Category"], f"{prompt_name}": temp_count_series})
            dataframe_count = pd.merge(dataframe_count, temp_count_df, on='PHI_Category', how='left')

        else:
            print(
                f"Warning: Column '{prompt_name}' not found in all_phi_dataframe for prompt. Adding empty column to dataframe_all and dataframe_count.")
            dataframe_all[f"{prompt_name}"] = pd.NA
            dataframe_count[f"{prompt_name}"] = pd.NA
        
        return dataframe_all, dataframe_count
    
    @staticmethod
    def initialize_dataframes()->tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Initiate dataframe for further work
        :return: tuple of dataframes
        """
        categories = ['mrn','ssn', 'account_number', 'health_plan_number','address', 'telephone_number','email','language',
               'marital_status', 'test_results', 'case','gender','date', 'name']

        categories_real = ["Name", "MRN", "Legal gender", "Date of Visit","Telephone", "SSN", "Doctors Name",
                      "patients_adress", "Doctors Address", "Health Plan Number", "Test Results",
                      "Billing Account Number", "RN", "Admission Type","Language spoken",
                      "Date of Discharge", "DOB", "Email", "Marital status", "Contact Person Name", "Language written"]

        expected_columns = [
            "PHI_Category",
            "Extracted_Value",
            "Found_in_ACC_DF",
            "ACC_DF_Columns_Found_In",
            "Total_Occurrences_in_ACC_DF",
            "Associated_Names",
            "Prompt"
        ]

        dataframe_all = pd.DataFrame({"PHI_Category": categories})
        dataframe_count = pd.DataFrame({"PHI_Category": categories})
        dataframe_real = pd.DataFrame({"PHI_Category": categories_real})
        dataframe_real_count = pd.DataFrame({"PHI_Category": categories_real})
        dataframe_real['PHI_Category'] = dataframe_real['PHI_Category'].str.lower()
        real_data = pd.DataFrame(columns=expected_columns)
        
        return dataframe_all, dataframe_count, dataframe_real, dataframe_real_count, real_data

    @staticmethod
    def check_dictionary(base_path:str, dataset:str, model_name:str, prompt_type:str, epoch:str, dictionary=None)->str:
        """
        Check whether the dictionary exists
        :param base_path: path to the base directory
        :param dataset: dataset name
        :param model_name: model name
        :param prompt_type: prompt type
        :param epoch: epoch
        :param dictionary: dictionary
        :return: new dictionary
        """
        if dictionary is None:
            new_directory = f"{base_path}/{dataset}/{model_name}/{epoch}/{prompt_type}"
        else:
            new_directory = f"{base_path}/{dataset}/{model_name}/{epoch}/{prompt_type}/{dictionary}"
        if not os.path.exists(new_directory):
            os.makedirs(new_directory)
        return new_directory

    @staticmethod
    def save_dataframes(data:pd.DataFrame, data_name:str, directory:str)->None:
        """
        Save dataframes
        :param data: dataframe
        :param data_name: dataset name
        :param directory: directory
        :return: None
        """
        data.to_csv(f"{directory}/{data_name}.csv", index=False)

    @staticmethod
    def make_real_dataframe_complete(dataframe_real, real_data_df, prompt_name, acc_dataframe):
        """
        Make non-hallucinated dataframe
        :param dataframe_real: initiated non-hallucinated dataframe
        :param real_data_df: dataframe with real data
        :param prompt_name: prompt name
        :param acc_dataframe: accuracy dataframe
        :return: non-hallucinated dataframe
        """
        acc_dataframe.columns = acc_dataframe.columns.str.lower()
        if f"{prompt_name}" not in dataframe_real.columns:
            dataframe_real[f"{prompt_name}"] = ""
        if 'Extracted_Value' not in dataframe_real.columns:
            dataframe_real['Extracted_Value'] = ''

        for index, row in real_data_df.iterrows():
            raw_acc_categories = [c.strip().lower() for c in row["ACC_DF_Columns_Found_In"].split(",")]

            for category in raw_acc_categories:
                if category:
                    target_row_index = dataframe_real[dataframe_real[f"PHI_Category"] == category].index

                    if not target_row_index.empty:
                        idx = target_row_index[0]

                        current_value = dataframe_real.loc[idx, f"{prompt_name}"]
                        new_value = row["Extracted_Value"]

                        if pd.isna(current_value) or str(current_value).strip() == "" or str(
                                current_value).strip().lower() == "nan":
                            dataframe_real.loc[idx, f"Extracted_Value"] = f"{new_value}"
                        else:
                            dataframe_real.loc[idx, f"Extracted_Value"] = f"{current_value}, {new_value}"

                        dataframe_real.loc[idx, f"Found_in_ACC_DF"] = row["Found_in_ACC_DF"]
                        dataframe_real.loc[idx, f"Total_Occurrences_in_ACC_DF"] = row["Total_Occurrences_in_ACC_DF"]
                        dataframe_real.loc[idx, f"Associated_Names"] = row["Associated_Names"]
                        dataframe_real.loc[idx, f"Prompt"] = row["Prompt"]

        if 'Total_Occurrences_in_ACC_DF' not in dataframe_real.columns:
            dataframe_real['Total_Occurrences_in_ACC_DF'] = 0
        if 'Associated_Names' not in dataframe_real.columns:
            dataframe_real['Associated_Names'] = ''

        new_total_occurrences = []
        new_associated_names = []

        for idx_df_real, row_df_real in dataframe_real.iterrows():
            current_counter = 0
            current_associated_name = ""

            extracted_value_from_df_real = str(row_df_real["Extracted_Value"])

            for idx_acc, acc_row in acc_dataframe.iterrows():
                Phi = row_df_real["PHI_Category"]
                if Phi not in acc_dataframe.columns:
                    print(f"Error: Column '{Phi}' not found in acc_dataframe!")
                    continue
                if extracted_value_from_df_real == str(acc_row[f"{Phi}"]):
                    current_counter += 1
                    current_associated_name += str(acc_row[f"name"]) + ", "

            new_total_occurrences.append(current_counter)
            new_associated_names.append(current_associated_name.rstrip(', '))

        dataframe_real['Total_Occurrences_in_ACC_DF'] = new_total_occurrences
        dataframe_real['Associated_Names'] = new_associated_names

        dataframe_real['Extracted_Value'] = dataframe_real['Extracted_Value'].replace('', np.nan)
        dataframe_real.dropna(subset=['Extracted_Value'], inplace=True)
        return dataframe_real
