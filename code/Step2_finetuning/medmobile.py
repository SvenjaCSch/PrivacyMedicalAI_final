#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: Step2_finetuning/medmobile.py
Author: Svenja C. Schulze
Last Updated: 2025-09-06
Description: get model MedMobile
"""

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import torch
import os
from dotenv import load_dotenv

class Medmobile:
    def __init__(self):
        load_dotenv()
        self.model_name = "KrithikV/MedMobile"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.HF_TOKEN = os.getenv("HF_TOKEN")
        self.HF_HOME = os.getenv("HF_HOME")


    def get_model(self):
        """
        Loads and returns a fresh instance of the base model every time it's called.
        """
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            token=self.HF_TOKEN,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        num_total = sum(p.numel() for p in model.parameters())

        print(
            f"Trainable Params: {num_trainable} || Total Params: {num_total} || Trainable%: {num_trainable / num_total:.4%}")
        return model

    def get_tokenizer(self):
        """
        Loads and returns a fresh instance of the tokenizer every time it's called.
        """
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            model_max_length=512,
            token=self.HF_TOKEN,
        )
        return tokenizer