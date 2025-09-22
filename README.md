# PrivacyMedicalAI Final
This GitHub- Pages contains the Code and Data of the 
Master's thesis  Privacy in Medical AI: Evaluating Risk and Countermeasures for Protecting Patient Information by Svenja C. Schulze.

## Introduction
Since generative AI in medicine processes rapidly throughout the recent years, a further view on the vulnerabilty of medical language models. Since no recent papers about the vunerability of the models Meerkat and MedMobile focusing on prompt attacks I found, I bridge this gap with the Master's thesis. This GitHub repository 
functions as the code and data I used as foundation of the written thesis. 

## Installation

This project utilizes **Python 3.13**.

To install the required dependencies, clone this repository and run:

```bash
pip install -r requirements.txt
```

For using GPT-3.5, an OpenAI API key is required. Other models in this project require access to Hugging Face’s API.

Additionally, an `.env` file must be created with the following format:

```bash
HF_TOKEN="YOUR_HUGGINGFACE_API_KEY"
BASE="YOUR_BASE_PATH_TO_THE_PROJECT"
```


## Method

### Models
The two models Meerkat ([1], [2]) and MedMobile [3] were evaluated by finetuning and attacking. 
### Data
Both base models were fine-tuned with 4 different datasets. 
The datasets are similar to the Electronical Health Record (EHR) [3] outputs. 
The datasets differ by lenghts of the EHRs and number of patient entries.
### Experiment 1
In Experiment 1 the models were trained by the epoch that got the lowest perplexity in fine-tuning.
Therefore, 4 fine-tuned models with low epochs trained on different datasets per model were created.
For the datasets different base-datasets ([4],[5],[6],[7],[8],[9]) were used and furthermore enhanced with randomized PHIs.
### Experiment 2
Experiment 2 created 4 fine-tuned models per MedMobile and Meerkat with different datasets. 
For this experiment the epoch 20 as a very high epoch was used. This created an overfitting favouring the adversarial prompt attacks. 
### Experiment 3
For the third experiment solely the Meerkat was fine-tuned on dataset 2 and 4. 
Here, decreasing epochs are investigated to extract the breaking point of the PHI extractions.
### Defence
For the defence I used a prompt-base related defence. I tested the most vulnerable fine-tuned model of experiment 3.

## Github Steps
Here, I explain the steps in GitHub for both the code and the data folder.
### Step 1: Preprocessing
The first step is the preprocessing of the datasets. Here I store the code for creating the datasets for the fine-tuning.
Additionally, the created datasets are used for creating accuracy datasets by dropping the non-PHI-columns.  

### Step 2: Finetuning
The fine-tuning steps takes place for both Meerkat and MedMobile. Therefore, different prompt styles need to be considered.
For the finetuning LoRA [12] is used as well as an epoch-wise evaluation. For comparing the fine-tuned models with the original model, perplexity is used.

### Step 3: Prompt Attacks ([13]-[17])
The prompt attacks are divided into general and specific prompt attacks. 

#### General Prompt Attacks
The general prompt attack are divided into different categories:

| Attack  | Prompt                    | Prompt Category     |
|---------|---------------------------|---------------------|
| General | List All Patient Records  | Listing             |
| General | List All \acp{phi}        | Listing             |
| General | List Example              | Listing             |
| General | List Most Common          | Listing             |
| General | List Least Common         | Listing             |
| General | List Oldest               | Listing             |
| General | List Youngest             | Listing             |
| General | List Most Recent          | Listing             |
| General | List Least Recent         | Listing             |
| General | System Ignore             | System Changes      |
| General | System Overwritten        | System Changes      |
| General | Research Paper            | Research Paper      |
| General | Scenario Nesting Hospital | Template Completion |


#### Specific Prompt Attacks
Like the general prompts, the specific prompt attack are divided into different categories:

| Attack   | Prompt                                                                                             | Prompt Category        |
|----------|----------------------------------------------------------------------------------------------------|------------------------|
| Specific | Scenario Nesting Assistant                                                                         | Template Completion    |
| Specific | Scenario Nesting Hospital                                                                          | Template Completion    |
| Specific | Code Injection                                                                                     | Template Completion    |
| Specific | In-Context Attack                                                                                  | Template Completion    |
| Specific | JB Few Shot                                                                                   | JB                |
| Specific | JB Question and Answer                                                                        | JB                |
| Specific | JB Special Tokens                                                                             | JB                |
| Specific | Continue                                                                                           | Continue               |
| Specific | Prompt for Medical \ac{llm}                                                                        | Refusal                |
| Specific | Prompt for Specific Model                                                                          | Refusal                |
| Specific | Caesar| Cipher                 |
| Specific | Backwards 1 Question                                                                               | Cipher                 |
| Specific | Backwards 2 Question                                                                               | Cipher                 |
| Specific | Morse Code Question                                                                                | Cipher                 |
| Specific | ASCII                                                                                              | Cipher                 |
| Specific | Binary Question                                                                                    | Cipher                 |
| Specific | UTF                                                                                                | Cipher                 |
| Specific | Caesar Statement                                                                                   | Cipher                 |
| Specific | Backwards 1 Statement                                                                              | Cipher                 |
| Specific | Backwards 2 Statement                                                                              | Cipher                 |
| Specific | Morse Code Statement                                                                               | Cipher                 |
| Specific | ASCII Statement                                                                                    | Cipher                 |
| Specific | Binary Statement                                                                                   | Cipher                 |
| Specific | UTF Statement                                                                                      | Cipher                 |
| Specific | Euskera                                                                                            | Low Ressource Language |
| Specific | Scottish-Gaelic                                                                                    | Low Ressource Language |
| Specific | Research Paper                                                                                     | Research Paper         |


#### Step 4: Evaluation
After I attacked the fine-tuned models in the three experiments, I evaluated the results. 
For that I both evaluated the general and the specific results to gather information about the hallucination, success and risk ratios. 

#### Models and Outputs
The Output can be found on following links:

[Prompt Attack Baseline](https://drive.google.com/file/d/1XyRZRwfrHzqEJeruFAlIziDvUouX1dcj/view?usp=sharing)

[Prompt Attack Experiment 1](https://drive.google.com/file/d/1BqF-UJ7Qp6r93L7pocKvDUX36GFslVXq/view?usp=sharing)

[Prompt Attack Experiment 2](https://drive.google.com/file/d/1hXLfEBgTUpHWLdLsjbo_knI1CpwAqLl_/view?usp=sharing)

[Prompt Attack Experiment 3.1]()

[Prompt Attack Experiment 3.2](https://drive.google.com/file/d/1PJkgdQ0CwBGV7XWtcz6w83mJn1IfgrW3/view?usp=sharing)

[Defense](https://drive.google.com/file/d/1PoH3iaC-3MYK1-Re-zC09ca7BuIk7HXA/view?usp=sharing)

One of the models is stored here:

[Meerkat Dataset4 Epoch2](https://drive.google.com/file/d/1QGT7Euv0_6brqpqKKtBAQcA1U0pdHdMr/view?usp=sharing)

Further Models can be sended by request.

## References
[1] Kim, H., Hwang, H., Lee, J., Park, S., Kim, D., Lee, T., Yoon, C., Sohn, J., Choi, D., and Kang, J. (2024). Small language models learn enhanced reasoning skills from medical textbooks. arXiv preprint arXiv:2404.00376.

[2] Meerkat Model: https://huggingface.co/dmis-lab/llama-3-meerkat-8b-v1.0

[3] VISHWANATH, Krithik, et al. MedMobile: A mobile-sized language model with expert-level clinical capabilities. arXiv preprint arXiv:2410.09019, 2024.

[4] Medmobile Model: https://huggingface.co/KrithikV/MedMobile 

[5] https://www.kaggle.com/datasets/uom190346a/disease-symptoms-and-patient-profile-dataset

[6] https://www.kaggle.com/datasets/prasad22/healthcare-dataset

[7] https://www.kaggle.com/datasets/aadyasingh55/disease-and-symptoms

[8] https://www.kaggle.com/datasets/ziya07/personalized-medication-dataset

[9] https://www.kaggle.com/datasets/boltcutters/food-allergens-and-allergies

[10] https://www.kaggle.com/datasets/staniherstaniher/data-patients-los-in-a-semi-urban-hospital?select=dataPaperStackingLOS_1.csv

[11] Jin, Q., Dhingra, B., Liu, Z., Cohen, W. W., & Lu, X. (2021). *MedQA: A Large-Scale Dataset for Medical Question Answering.* arXiv preprint arXiv:2105.14227.

[12] Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W., et al. (2022). Lora: Low-rank adaptation of large language models. ICLR, 1(2):3.

[13] CHEN, Sizhe, et al. Struq: Defending against prompt injection with structured queries. arXiv preprint arXiv:2402.06363, 2024.

[14] YI, Sibo, et al. Jailbreak attacks and defenses against large language models: A survey. arXiv preprint arXiv:2407.04295, 2024. 

[15] LIU, Yi, et al. Jailbreaking chatgpt via prompt engineering: An empirical study. arXiv preprint arXiv:2305.13860, 2023.

[16] BEURER-KELLNER, Luca, et al. Design Patterns for Securing LLM Agents against Prompt Injections. arXiv preprint arXiv:2506.08837, 2025.

[17] WANG, Yidan, et al. PIG: Privacy Jailbreak Attack on LLMs via Gradient-based Iterative In-Context Optimization. arXiv preprint arXiv:2505.09921, 2025.
