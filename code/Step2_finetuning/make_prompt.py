#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: Step2_finetuning/make_prompt.py
Author: Svenja C. Schulze
Last Updated: 2025-09-06
Description: create prompt for MedMobile and Meerkat
"""

import re
import pandas as pd

class MakePrompt:
    def __init__(self):
        self.system_message = "The following is a multiple-choice question about medical knowledge. You are strongly required to follow the specified output format: Conclude your response with the phrase 'The answer is ([option_id]) [answer_string]'."
        self.system_message_more_entries = "The following is a question about medical knowledge.You are strongly required to follow the specified output format: Conclude your response with the phrase 'The patient has [medical_condition]."

    def create_message(self, row:pd.Series)->dict:
        """
        Create message
        :param row: single row from a dataset
        :return: dictionary message
        """
        return {
            "messages": [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": f'{row["new_question"]} Choices: {row["choices"]}'},
                {"role": "assistant", "content": f'{row["answer_idx"]}: {row["original_answer"]}'}
            ]
        }

    def create_prompt_medmobile(self, exp:str, row:pd.Series)->dict:
        """
        Create prompt for MedMobile
        Dataset1 or Dataset3: Long entries with MedQA
        Dataset 2 or Dataset 4: Short Versions
        :param exp: dataset name (exp1- exp4)
        :param row: single row of the dataset
        :return: full prompt plus the assistant message
        """
        if exp == "exp1":
             system_msg = f"<|system|>{self.system_message} <|end|> "
             user_msg = f"<|user|> {row["new_question"]} Choices: {row["choices"]} <|end|>"
             assistant_msg = f'{row["answer_idx"]}: {row["original_answer"]}'
             user_msg = re.sub(r'\n', '', user_msg)
             assistant_msg = re.sub(r'\n', '', assistant_msg)
             full_prompt = system_msg + user_msg + "<|assistant|>"
             return {
                 "text": full_prompt + assistant_msg
            }
        if exp == "exp3":
             system_msg = f"<|system|>{self.system_message} <|end|> "
             user_msg = f"<|user|> {row["new_question"]} Choices: {row["choices"]} <|end|>"
             assistant_msg = f'{row["answer_idx"]}: {row["original_answer"]}'
             user_msg = re.sub(r'\n', '', user_msg)
             assistant_msg = re.sub(r'\n', '', assistant_msg)
             full_prompt = system_msg + user_msg + "<|assistant|>"
             return {
                 "text": full_prompt + assistant_msg
            }

        if exp == "exp2":
            system_msg = f"<|system|>{self.system_message_more_entries} <|end|> "
            user_msg = f"<|user|> {row["new_question"]} What is the medical condition of the patient?<|end|>"
            user_msg = re.sub(r'\n', '', user_msg)
            assistant_msg = f'The patient has {row['disease']}'
            assistant_msg = re.sub(r'\n', '', assistant_msg)
            full_prompt = system_msg + user_msg + "<|assistant|>"
            return {
                "text": full_prompt + assistant_msg
            }
        if exp == "exp4":
            system_msg = f"<|system|>{self.system_message_more_entries} <|end|> "
            user_msg = f"<|user|> {row["new_question"]} What is the medical condition of the patient?<|end|>"
            user_msg = re.sub(r'\n', '', user_msg)
            assistant_msg = f'The patient has {row['disease']}'
            assistant_msg = re.sub(r'\n', '', assistant_msg)
            full_prompt = system_msg + user_msg + "<|assistant|>"
            return {
                "text": full_prompt + assistant_msg
            }

    def create_prompt_meerkat(self, tokenizer, exp:str, row:pd.Series)->dict:
        """
        Create prompt for MedMobile
        Dataset1 or Dataset3: Long entries with MedQA
        Dataset 2 or Dataset 4: Short Versions
        :param exp: dataset name (exp1- exp4)
        :param row: single row of the dataset
        :return: full prompt plus the assistant message
        """
        if exp == "exp1":
            messages = [
                 {"role": "system", "content": self.system_message},
                 {"role": "user", "content": f"{row['new_question']} Choices: {row['choices']}"},
                 {"role": "assistant", "content": f"{row['answer_idx']}: {row['original_answer']}"}
            ]
        if exp == "exp3":
            messages = [
                 {"role": "system", "content": self.system_message},
                 {"role": "user", "content": f"{row['new_question']} Choices: {row['choices']}"},
                 {"role": "assistant", "content": f"{row['answer_idx']}: {row['original_answer']}"}
            ]
        if exp == "exp2":
            messages = [
                {"role": "system", "content": self.system_message_more_entries},
                {"role": "user", "content": f"{row['new_question']}. What is the medical condition of the patient?"},
                {"role": "assistant", "content": f"The patient has {row['disease']}"}
            ]
        if exp == "exp4":
            messages = [
                {"role": "system", "content": self.system_message_more_entries},
                {"role": "user", "content": f"{row['new_question']}. What is the medical condition of the patient?"},
                {"role": "assistant", "content": f"The patient has {row['disease']}"}
            ]


        full_prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=False
        )

        return {"text": full_prompt}
