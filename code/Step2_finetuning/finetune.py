#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: Ste2_finetuning/finetune.py
Author: Svenja C. Schulze
Last Updated: 2025-09-06
Description: finetune the MedMobile and Meerkat Models. Depending on the Experiment, the models are finetuned for another epoch.
"""

import os
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from meerkat import Meerkat
from medmobile import Medmobile
from perplexityCallback import PerplexityCallback
import evaluate
from make_prompt import MakePrompt
from get_data import GetData
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
from transformers import DataCollatorForLanguageModeling
import json
import torch
from customPerplexityCallback import CustomPerplexityCallback
from dotenv import load_dotenv

class Finetune:

    def __init__(self):
        self.id2label  = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}
        self.label2id  = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
        self.perplexity = evaluate.load('perplexity')
        self.meerkat = Meerkat()
        #self.medmobile = Medmobile()
        self.perplexity_callback = PerplexityCallback()
        self.make_prompt = MakePrompt()
        self.get_data = GetData()
        self.callback = CustomPerplexityCallback()
        load_dotenv()
        self.BASE = os.getenv("BASE_MLCLOUD")

    @staticmethod
    def get_tokenizer(base_model):
        """
        Get tokenizer
        :param base_model: base model MedMobile or Meerkat
        :return: tokenizer
        """
        return base_model.get_tokenizer()

    @staticmethod
    def get_model(base_model):
        """
        Get base model.The model is not created directly since it would create a CUDA overflow.
        :param base_model: base model MedMobile or Meerkat
        :return: base model
        """
        return base_model.get_model()

    @staticmethod
    def finetune_lora(model):
        """
        Finetune with LoRA
        Target models used: "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"
        :param model: model used MedMobile or Meerkat
        :return: finetuned model with LoRA
        """
        config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            inference_mode=False
        )
        model = get_peft_model(model, config)
        print(model.print_trainable_parameters())
        return model

    @staticmethod
    def store_dataset_as_txt(dataset:list, base:str, name:str)->None:
        """
        store the dataset as txt file. The dataset is stored both as text file and as json file.
        :param dataset: dataset list
        :param base: file base
        :param name: name
        :return: None
        """
        output_file = f"{base}_dataset_{name}.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            for row in dataset:
                text = row['text']
                f.write(text + '\n')

    @staticmethod
    def store_dataset_as_jsonl(dataset:list, base:str, name:str, dataset_number:str)->None:
        """
        store the dataset as json file. The dataset is stored both as text file and as json file.
        :param dataset: dataset list
        :param base: file base
        :param name: model name
        :param dataset_number: dataset number
        :return: None
        """
        output_path = f"{base}/models/_dataset_{name}_{dataset_number}_epoch{epoch}.jsonl"
        with open(output_path, "w", encoding="utf-8") as f:
            for row in dataset:
                json.dump(row, f)
                f.write("\n")

    def finetune(self, model, dataset, eval_dataset, tokenizer, epoch_number)->tuple:
        """
        Finetune the model into the final model
        using different epochs
        :param model: MedMobile or Meerkat
        :param dataset: dataset used for finetuning
        :param eval_dataset: corresponding eval dataset
        :param tokenizer: tokenizer MedMobile or Meerkat
        :param epoch_number: Epoch depending on the experiment
        :return: finetuned model and eval results
        """
        training_args = TrainingArguments(
              per_device_train_batch_size=1,
              gradient_accumulation_steps=1,
              evaluation_strategy="epoch",
              logging_strategy="epoch",
              eval_accumulation_steps=10,
              num_train_epochs=epoch_number,
              save_strategy="epoch",
              learning_rate=1e-4,
              save_total_limit=3,
              output_dir="experiments",
              overwrite_output_dir=True,
              optim= "adamw_torch_fused",
              lr_scheduler_type="cosine",
              warmup_ratio=0.1,
              ddp_find_unused_parameters=False
        )

        trainer = Trainer(
            model=model,
            train_dataset=dataset["train"],
            eval_dataset= eval_dataset,
            args=training_args,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
            callbacks=[self.perplexity_callback],
        )
        model.config.use_cache = False
        trainer.train()

        eval_results = trainer.evaluate()
        print(eval_results)

        return model, eval_results

    def save_finetuned_model(self, model, name:str, tokenizer, epoch_number:int, dataset:str):
        """
        Save the finetuned model
        :param model: used model
        :param name: name of the model
        :param tokenizer: tokenizer MedMobile or Meerkat
        :param epoch_number: used epoch
        :param dataset: used dataset number
        :return: model and tokenizer
        """
        model = model.merge_and_unload()
        model.generation_config.do_sample = True
        model.save_pretrained(f"{self.BASE}/models/{name}_{dataset}_epoch{epoch_number}", safe_serialization=True)
        tokenizer.save_pretrained(f"{self.BASE}/models/{name}_{dataset}_epoch{epoch_number}")
        return model, tokenizer

    def store_and_print_perplexity(self, name:str, epoch_number:int, dataset_number:str, perplexity_string:str)->None:
        """
        Store and print the perplexity
        :param name: name of the model
        :param epoch_number: epoch number
        :param dataset_number: dataset number
        :param perplexity_string: store the string with perplexity
        :return:
        """
        file_path = f'{self.BASE}/models/{name}/perplexity_epoch{epoch_number}_{dataset_number}.txt'
        output_directory = os.path.dirname(file_path)
        os.makedirs(output_directory, exist_ok=True)
        with open(file_path, 'a') as the_file:
            the_file.write(perplexity_string)
            the_file.write('\n')

    def store_epochwise_perplexity(self, name:str, epoch_number:int, dataset_number:str)->None:
        """
        store the perplexity per epoch
        :param name: name of the model
        :param epoch_number: epoch number of the finetuned model
        :param dataset_number: dataset number of the finetuned model
        :return: None
        """
        file_path = f'{self.BASE}/models/perplexity_epoch{epoch_number}.txt'
        output_directory = os.path.dirname(file_path)
        os.makedirs(output_directory, exist_ok=True)
        with open(file_path, 'a') as the_file:
            the_file.write(f"{name} {epoch_number} {dataset_number}")
            the_file.write('\n')

    @staticmethod
    def tokenize_function(examples, tokenizer):
        """
        Tokenizer function for makeing the dataset used for finetuning ad evaluation
        :param examples: dataframe
        :param tokenizer: tokenizer
        :return: get tokenized dataset
        """
        return tokenizer(examples["new_question"], truncation=True, padding="max_length", max_length=512)

    def get_tokenized_dataset(self, dataset, dataset_eval):
        """
        Tokenize the dataset
        :param dataset: training dataset
        :param dataset_eval: evaluation dataset
        :return: return the tokenized datasets
        """
        tokenized_dataset = dataset.map(self.tokenize_function, batched=True)
        tokenized_eval_dataset = dataset_eval.map(self.tokenize_function, batched=True)
        return tokenized_dataset, tokenized_eval_dataset

    def make_dataset(self, name, model_name, dataset, model=None, tokenizer=None):
        """
        Creates the datasets for the training and evaluation
        :param name: train or evaluation
        :param model_name: name of the model
        :param dataset: dataset number
        :param model: Meerkat or MedMobile
        :param tokenizer: tokenizer MedMobile or Meerkat
        :return: encoded dataset with the tokenized text field
        """
        dataset, dataset_eval = self.get_data.get_data(f"{self.BASE}/", dataset)
        if name == "train":
            dataset = self.get_data.make_data_prompt(dataset, f"{self.BASE}/", name, model_name, dataset, model, tokenizer)
        else:
            dataset = self.get_data.make_data_prompt(dataset_eval, f"{self.BASE}/", name, model_name, dataset, model, tokenizer)
        self.store_dataset_as_txt(dataset, f"{self.BASE}/", name)
        self.store_dataset_as_jsonl(dataset, f"{self.BASE}/", name, dataset)
        new_dataset = load_dataset("json", data_files={f"{self.BASE}/models/_dataset_{name}_{dataset}_epoch{epoch}.jsonl"})
        encoded_dataset = new_dataset.map(lambda row: tokenizer(row["text"], truncation=True, padding=True),batched=True)
        return encoded_dataset

    def get_perplexity(self, name, dataset_number, model, tokenizer, eval_dataset, type_model):
        """
        Another way to get perplexity
        Here, the loss needs to be calculated
        :param name: name of the model
        :param dataset_number: dataset number
        :param model: Meerkat or MedMobile
        :param tokenizer: tokenizer MedMobile or Meerkat
        :param eval_dataset: evaluation dataset
        :param type_model: Original or Finetuned model
        :return: perplexity
        """
        loss, ppl = self.callback.compute_perplexity_on_model(model, tokenizer, eval_dataset)
        print(f"{type_model} Model Eval Loss with compute_perplexity_on_model: {loss:.4f}, Perplexity: {ppl:.4f}")
        self.store_and_print_perplexity(name, epoch, dataset_number,
                                        f"{type_model} Model Eval Loss with compute_perplexity_on_model: {loss:.4f}, Perplexity: {ppl:.4f}")

    def finetuning_manager(self, name:str, base_model, dataset_number:str, epoch:int)->None:
        """
        Manages the fine-tuning of the model
        :param name: name of the model
        :param base_model: orginal model
        :param dataset_number: dataset number
        :param epoch: epoch number
        :return: None
        """
        model = self.get_model(base_model)
        tokenizer = self.get_tokenizer(base_model)
        encoded_dataset = self.make_dataset("train", name, dataset_number, model, tokenizer)
        encoded_dataset_eval = self.make_dataset("validation", name, dataset_number, model, tokenizer)
        eval_dataset = encoded_dataset_eval["train"].select(range(10))

        self.store_epochwise_perplexity(name, epoch, dataset_number)

        self.get_perplexity(name, dataset_number, model, tokenizer, eval_dataset, "Original")

        lora_model = self.finetune_lora(model)
        finetuned_model, eval_results= self.finetune(lora_model, encoded_dataset, eval_dataset, tokenizer, epoch)
        finetuned_model_saved, finetuned_tokenizer = self.save_finetuned_model(finetuned_model, name, tokenizer, epoch, dataset_number)

        self.get_perplexity(name, dataset_number, finetuned_model_saved, finetuned_tokenizer, eval_dataset, "Finetuned")

        del model
        del tokenizer
        del lora_model
        del finetuned_model
        del finetuned_model_saved
        del finetuned_tokenizer
        torch.cuda.empty_cache()



if __name__ == "__main__":
    finetune = Finetune()
    experiments = ["exp2"]
    epochs = [5, 4]
    for exp in experiments:
        for epoch in epochs:
            #print(f"Finetuning epoch {epoch}, {exp}, medmobile")
            #finetune.finetuning_manager("medmobile",  finetune.medmobile, exp, epoch)
            print(f"Finetuning epoch {epoch}, {exp}, Meerkat")
            finetune.finetuning_manager("meerkat", finetune.meerkat, exp, epoch)


