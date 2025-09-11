#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: Step2_finetuning/meerkat.py
Author: Svenja C. Schulze
Last Updated: 2025-09-06
Description: get model Meerkat
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

class Meerkat:
    def __init__(self):
        self.model_name = "dmis-lab/llama-3-meerkat-8b-v1.0"
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
            low_cpu_mem_usage=True,
        ).to("cuda")
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
            token=self.HF_TOKEN
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        return tokenizer


