#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: Ste2_finetuning/customerPerplityCallback.py
Author: Svenja C. Schulze
Last Updated: 2025-09-06
Description: makes perplexity for the original and finetuned model. In contrast to the epoch wise perplexity callback, this one needs to calculate the loss before
Therefore, a small difference between them is possible
"""

from transformers import TrainerCallback
import torch
from torch.utils.data import DataLoader
import math

class CustomPerplexityCallback(TrainerCallback):
    def __init__(self):
        pass


    def compute_perplexity_on_model(self, model, tokenizer, eval_dataset, batch_size=1):
        """
        Compute the perplexity on the model
        It uses an inner function since for encapsulation.
        Dataloader is used to simplify loading.
        Here, the inner function collate_fn is used to transform the data with the evaluation dataset.
        It returns the input_ids, attention_mask and labels.

        The labels are also the input ids. Then the models loss are calculated with the information of the collate_fn
        With the average loss, the total loss and the token numbers the perplexity is calculated and returned.

        :param model: Meerkat or MedMobile
        :param tokenizer:tokenizer for Meerkat or MedMobile
        :param eval_dataset: evaluation dataset
        :param batch_size: batch size, here only 1
        :return: Perplexity on the evaluation dataset
        """
        model.eval()

        def collate_fn(batch):
            input_ids = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(example["input_ids"]) for example in batch],
                batch_first=True,
                padding_value=tokenizer.pad_token_id
            )
            attention_mask = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(example["attention_mask"]) for example in batch],
                batch_first=True,
                padding_value=0
            )
            labels = input_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

        dataloader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=collate_fn)

        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(model.device)
                attention_mask = batch["attention_mask"].to(model.device)
                labels = batch["labels"].to(model.device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                num_tokens = (labels != -100).sum().item()

                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens

        if total_tokens == 0:
            return float("inf"), float("inf")

        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        return avg_loss, perplexity

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        """
        The epoch wise perplexity callback to caluclate the perplexity and than print it out.
        :param args: training arguments to instantiate the Trainer
        :param state: the current state of the training
        :param control: can be used to make some decisiions in the trainer
        :param model: Meerkat or MedMobile
        :param kwargs: arguments not in args, state or control
        :return: print of the epochwise perplexity
        """
        if not hasattr(self, 'tokenizer') or not hasattr(self, 'eval_dataset'):
            return
        _, perplexity = self.compute_perplexity_on_model(
            model=model,
            tokenizer=self.tokenizer,
            eval_dataset=self.eval_dataset,
            batch_size=1
        )
        print(f"[Epoch {int(state.epoch)}] Perplexity: {perplexity:.2f}")