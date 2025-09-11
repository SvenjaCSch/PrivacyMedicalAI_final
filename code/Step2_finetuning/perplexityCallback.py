#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: Ste2_finetuning/perplexityCallback.py
Author: Svenja C. Schulze
Last Updated: 2025-09-06
Description: calculated perplexity during epochs with training evaluate. Use of the Trainer callback. Perplexity calculated with the eval loss
"""

from transformers import TrainerCallback
import math
import os

class PerplexityCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_loss" in metrics:
            ppl = math.exp(metrics["eval_loss"])
            print(f"\nPerplexity at epoch {int(state.epoch)} only with eval loss: {ppl:.4f}")
            self.store_and_print_perplexity(state=state.epoch, ppl =ppl)

    def store_and_print_perplexity(self, state, ppl):
        file_path = f'/mnt/lustre/work/eickhoff/esx395/models/perplexity_epoch{state}.txt'
        output_directory = os.path.dirname(file_path)
        os.makedirs(output_directory, exist_ok=True)
        with open(file_path, 'a') as the_file:
            the_file.write(f"\nPerplexity at epoch {int(state)} only with eval loss: {ppl:.4f}")
            the_file.write('\n')
