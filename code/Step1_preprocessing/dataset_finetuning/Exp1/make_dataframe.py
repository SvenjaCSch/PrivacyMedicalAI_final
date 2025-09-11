#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: make_dataframe.py
Author: Svenja C. Schulze
Last Updated: 2025-10-03
Description: Makes the dataframe of the patient selections
"""

import os
import re
import pandas as pd
from dotenv import load_dotenv
import random
from faker import Faker
import datetime
import numpy as np

'''
Only one entry per person
Long entries
'''
class PHIDataset:
    def __init__(self):
        load_dotenv()
        self.BASE = os.getenv("BASE")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.faker = Faker('en_US')

    @staticmethod
    def get_dataset(url:str)->pd.DataFrame:
        """
        Load dataset from URL and return a random sample.
        :param url: url
        :return: pd.DataFrame
        """
        data = pd.read_csv(url)
        return data

    def random_date(self)-> datetime.date:
        """
        create random date
        :return: random date between given start and end date
        """
        start_date = self.faker.date_this_year(before_today=True)
        end_date = datetime.date.today()
        return self.faker.date_between_dates(date_start=start_date, date_end=end_date)

    @staticmethod
    def find_gender(string:str)->str:
        """
        get gender from string
        :param string: string
        :return: gender
        """
        male_persons = ["man", "male", "boy"]
        female_persons = ["woman", "female", "girl"]
        pattern = re.compile(r"\d-(month|year|day)-old\s(\w*)\s(\w*)\s(\w*)")
        substring = pattern.findall(string)
        if not substring:
            return "unknown"
        else:
            for female_gender in female_persons:
                if re.search(female_gender, str(substring[0])):
                    return "female"
            for male_gender in male_persons:
                if re.search(male_gender, str(substring[0])):
                    return "male"
            return "divers"

    def get_forename(self,string:str)-> str:
        """
        get name from gender string
        :param string: gender
        :return: name
        """
        if string == "male":
            return self.faker.first_name_male()
        if string == "female":
            return self.faker.first_name_female()
        else:
            return self.faker.first_name()

    @staticmethod
    def make_phone_number()-> str:
        """
        get phone number with different areas
        :return: telephone number
        """
        area = [212, 315, 347, 516, 607, 631, 646, 718, 845, 917, 929, 339, 351, 413, 508, 617, 774, 781, 857, 978, 215,
                267, 412, 484, 570, 610, 717, 814, 878, 201, 551, 609, 732, 848, 856, 862, 908, 203, 475, 860, 959, 217,
                309, 312, 331, 618, 630, 708, 773, 815, 872, 231, 248, 269, 313, 517, 586, 616, 734, 810, 906, 216, 234,
                330, 419, 440, 513, 567, 614, 740, 937, 219, 260, 317, 463, 574, 765, 812, 930, 262, 414, 534, 608, 715,
                920, 314, 417, 573, 636, 660, 816, 239, 305, 321, 352, 386, 407, 561, 727, 754, 786, 813, 850, 904, 941,
                954, 210, 214, 254, 281, 325, 361, 409, 430, 512, 682, 713, 726, 737, 806, 817, 830, 903, 915, 936, 940,
                956, 972, 979, 229, 404, 470, 478, 678, 706, 762, 770, 912, 252, 336, 704, 743, 828, 910, 919, 984, 276,
                434, 540, 571, 703, 757, 804, 423, 615, 629, 731, 865, 901, 931, 209, 213, 310, 323, 408, 415, 424, 442,
                530, 559, 562, 619, 626, 628, 650, 657, 661, 669, 707, 714, 747, 760, 805, 818, 831, 858, 909, 916, 925,
                949, 951, 480, 520, 602, 623, 928, 702, 725, 206, 253, 360, 425, 509, 458, 503, 541, 971, 303, 719, 720,
                970, 808, 907, 218, 320, 507, 612, 651, 763, 952, 316, 620, 785, 913, 308, 402, 531, 405, 539, 580, 918,
                319, 515, 563, 641, 712]
        return f"+1{random.choice(area)}-{random.randint(100,999)}-{random.randint(1000, 9999)}"

    @staticmethod
    def get_email(forename:str, surname:str)-> str:
        """
        get email from forename and surname
        :param forename: forename
        :param surname: surname
        :return: email
        """
        provider = ["yahoo", "gmail", "gmx", "web", "brown", "dhl", "aol"]
        return f'{forename}.{surname}@{random.choice(provider)}.com'

    @staticmethod
    def get_social_security()-> str:
        """
        get social security number
        :return: ssn
        """
        return f"{random.randint(100,999)}-{random.randint(10,99)}-{random.randint(1000,9999)}"

    @staticmethod
    def strip_last_question(questions:str)-> tuple[str, str]:
        """
        strip the last question from questions
        :param questions: question string
        :return: question without last question
        """
        extracted_final_question = ""
        hpi_without_final_question = questions.strip()
        last_q_mark_idx = questions.rfind('?')

        if last_q_mark_idx != -1:
            segment_before_q = questions[:last_q_mark_idx]
            last_period_idx_before_q = segment_before_q.rfind('.')
            last_newline_idx_before_q = segment_before_q.rfind('\n')
            if last_period_idx_before_q != -1 and last_newline_idx_before_q != -1:
                start_extract_idx = max(last_period_idx_before_q, last_newline_idx_before_q) + 1
            elif last_period_idx_before_q != -1:
                start_extract_idx = last_period_idx_before_q + 1
            elif last_newline_idx_before_q != -1:
                start_extract_idx = last_newline_idx_before_q + 1
            else:
                start_extract_idx = 0
            if start_extract_idx < 0:
                start_extract_idx = 0
            extracted_final_question = questions[start_extract_idx: last_q_mark_idx + 1].strip()

            hpi_without_final_question = questions.replace(extracted_final_question, "", 1).strip()

        else:
            print("Warning: No question mark found in the 'original_hpi_and_question' string.")
        return extracted_final_question, hpi_without_final_question

    def change_question(self, question:str, new_sentence:str, last_sentence:str)->str:
        """
        change question into the new sentence
        :param question: question string
        :param new_sentence: new sentence
        :param last_sentence: last sentence
        """
        extracted_final_question, hpi_without_final_question = self.strip_last_question(question)
        new_question = f"{new_sentence}\n{hpi_without_final_question}\n{last_sentence}\n{extracted_final_question}"
        return new_question

    @staticmethod
    def get_allergies()->str:
        """
        get allergies
        allergies by:
        https://www.kaggle.com/datasets/ziya07/personalized-medication-dataset
        https://www.kaggle.com/datasets/boltcutters/food-allergens-and-allergies

        :return: allergies
        """
        number_of_allergies = random.randint(0,10)
        allergies = ""
        for i in range(number_of_allergies):
            allergies += f'{random.choice(["Penicillin", "Sulfa", 'Allium', 'Alpha-gal Syndrome','Aquagenic Urticaria','Banana','Beer','Broccoli','Citrus','Corn','Cruciferous','Fish','Gluten','Histamine','Honey','Hypersensitivity','Insulin',
    'Lactose Intolerance','Legume','LTP','Milk / Lactose intolerance','Mint','Mushroom','Nightshade','Nut','Ochratoxin','Oral Syndrome','Peanut',
    'Pepper','Potato','Poultry','Ragweed','Rice','Salicylate','Seed','Shellfish','Soy','Stone Fruit','Sugar / Intolerance','Tannin','Thyroid'] )}\n'
        return allergies

    @staticmethod
    def get_birthyear(question:str, admission:datetime.date)-> int | str:
        """
        get birthyear
        :param question: question string
        :param admission: admission date
        :return: birthyear
        """
        pattern = re.compile(r'(\d+)[-\s]?year')
        match = pattern.search(question)

        if not match:
            return "unknown"

        age = int(match.group(1))
        admission_year = int(admission.strftime('%Y'))

        birth_year = admission_year - age
        return birth_year

    @staticmethod
    def get_discharge_date(admission:datetime.date)->str:
        """
        get discharge date
        :param admission: admission date
        :return: discharge date
        """
        admission_year = int(admission.strftime('%Y'))
        admission_month = int(admission.strftime('%m'))
        admission_day = int(admission.strftime('%d'))
        new_day = admission_day + random.randint(1,10)
        if new_day <= 28:
            discharge_date = f"{admission_year}-{admission_month}-{admission_day}"
        else:
            if admission_month + 1 <= 12:
                new_day = new_day - 28
                discharge_date = f"{admission_year}-{admission_month + 1}-{new_day}"
            else:
                new_day = new_day - 28
                discharge_date = f"{admission_year}-{admission_month - 11}-{new_day}"
                
        return discharge_date

    def get_random_birthday(self, question:str, admission:datetime.date)->str:
        """
        get random birthday
        :param question: question string
        :param admission: admission date
        :return: random birthday
        """
        birthyear = self.get_birthyear(question, admission)
        if birthyear == "unknown":
            return "unknown"

        admission_month = int(admission.strftime('%m'))
        month = random.randint(1, admission_month)

        if month == 2:
            day = random.randint(1, 29 if birthyear % 4 == 0 else 28)
        else:
            day = random.randint(1, 30)

        birthday = f"{birthyear}-{month:02d}-{day:02d}"
        return birthday

    def get_address(self)->str:
        """
        get address
        :return: address
        """
        address = self.faker.address()
        adress_new = re.sub("\n", " ", address)
        return adress_new

    def get_contact(self,question:str ,surname:str, status:str)->str:
        """
        get contact person
        :param question: question string
        :param surname: surname string
        :param status: status string
        :return: contact person
        """
        pattern = re.compile(r'(\d+)[-\s]?year')
        match = pattern.search(question)

        if not match:
            return "unknown"

        age = int(match.group(1))
        contact_male_name = self.get_forename("male")
        contact_female_name = self.get_forename("female")
        if age <18:
            return f"{contact_male_name} {surname} (father), {contact_female_name} {surname} (mother)"
        else:
            gender = random.choice(["male, female"])
            if gender == "male":
                if status == "Married":
                    return f"{contact_male_name} {np.random.choice([{surname}, {self.faker.last_name()}], p=(0.9, 0.1))} (husband)"
                else:
                    return f"{contact_male_name} {self.faker.last_name()} ({random.choice(["boyfriend", "friend", "fiancee"])})"
            else:
                if status == "Married":
                    return f"{contact_male_name} {np.random.choice([{surname}, {self.faker.last_name()}], p=(0.9, 0.1))} (wife)"
                else:
                    return f"{contact_female_name} {self.faker.last_name()} ({random.choice(["girlfriend", "friend", "fiancee"])})"

    @staticmethod
    def get_marital_status(question:str)->str:
        """
        get marital status based of the question string
        :param question: question string
        """
        pattern = re.compile(r'(\d+)[-\s]?year')
        match = pattern.search(question)

        if not match:
            return "unknown"

        age = int(match.group(1))

        if age <18:
            return np.random.choice(["Married", "Single", "Divorced", "Widowed"], p=[0.3, 0.5, 0.15, 0.05])
        else:
            return "Single"

    @staticmethod
    def left_align(text_block:str)->str:
        """
        get the left aligned text
        :param text_block: text block
        :return: left aligned text
        """
        left_aligned_lines = [line.lstrip() for line in text_block.splitlines()]
        left_aligned_text = "\n".join(left_aligned_lines)
        return left_aligned_text

if __name__ == "__main__":
    base1 = "/mnt/lustre/work/eickhoff/esx395/code/phi_data/"
    base2 = r"C:\Users\svcsc\Desktop\Studium\Masterarbeit\PrivacyAI\PrivacyMedicalAI\dataset\synthetic_dataset\medqa"
    base3 = r"C:\Users\svcsc\Desktop\Studium\Masterarbeit\PrivacyAI\PrivacyMedicalAI\data\Step1_preprocessing\dataset_finetuning\Exp1\dataset_single_entries_per_person"
    phi = PHIDataset()
    traintestval = ["train", "test", "validation"]
    for type_data in traintestval:
        dataset_url = f"{base2}_{type_data}_final_selection.csv"
        medqa = phi.get_dataset(dataset_url)
        final_data = []
        for _, row in medqa.iterrows():
                generated_data = {
                    "original_question": row["question"],
                    "original_answer": row["answer"],
                    "choices": row["options"],
                    "answer_idx": row["answer_idx"],
                    "medical_record_number": random.randint(1000000000, 9999999999),
                    "gender": phi.find_gender(row["question"]),
                    "surname": phi.faker.last_name(),
                    "date_of_admission": phi.random_date(),
                    "phonenumber": phi.make_phone_number(),
                    "social_secruity_number": phi.get_social_security(),
                    "doctors_name" : phi.faker.name(),
                    "patients_adress": phi.get_address(),
                    "doctors_adress": phi.get_address(),
                    "health_plan_number": f"HPB-{random.randint(10000,99999)}",
                    "test_results": np.random.choice(["Normal", "Abnormal", "Inconclusive"], p=[0.6, 0.3, 0.1]),
                    "billing_account_number": f"#{random.randint(100000,999999)}",
                    "nurse_name": phi.faker.name(),
                    "admission_type": random.choice(["Urgent", "Emergency", "Elective"]),
                    "language_spoken": np.random.choice(["English", "Spanish", "German", "Italian", "Russian", "Arabic", "Chinese"], p=[0.9, 0.03, 0.02, 0.01, 0.01, 0.02, 0.01]),
                    "language_written": np.random.choice(["English", "Spanish", "German", "Italian", "Russian", "Arabic", "Chinese"], p=[0.9, 0.03, 0.02, 0.01, 0.01, 0.02, 0.01]),
                    "allergies": phi.get_allergies(),
                }
                final_data.append(generated_data)

        final_df = pd.DataFrame(final_data)
        final_df["forename"] = final_df.apply(lambda final_row: phi.get_forename(final_row["gender"]), axis=1)
        final_df["discharge_date"] = final_df.apply(lambda final_row: phi.get_discharge_date(final_row["date_of_admission"]), axis=1)
        final_df["date_of_birth"] = final_df.apply(lambda final_row: phi.get_random_birthday(final_row["original_question"],final_row["date_of_admission"]), axis=1)
        final_df["email"] = final_df.apply(lambda final_row: phi.get_email(final_row["forename"], final_row["surname"]), axis=1)
        final_df["marital_status"] = final_df.apply(lambda final_row: phi.get_marital_status(final_row["original_question"]), axis=1)
        final_df["contact_person"] = final_df.apply(lambda final_row: phi.get_contact(final_row["original_question"], final_row["surname"], final_row["marital_status"]), axis=1)

        final_df["first_sentence"] = final_df.apply(lambda final_row:
f"""———————————————————————————————————————————————
Patient Name:       {final_row["forename"]} {final_row["surname"]}\t\tMRN:\t\t\t\t{final_row["medical_record_number"]} 
DOB:                {final_row["date_of_birth"]}\t\t\tLegal sex:\t\t\t{final_row["gender"]} 
Date of Visit:      {final_row["date_of_admission"]}\t\t\tContact Person:\t\t{final_row["contact_person"]}
Date of Discharge:  {final_row["discharge_date"]}\t\t\tSocial security:\t{final_row["social_secruity_number"]}
Health Plan Number: {final_row["health_plan_number"]}\t\t\tAccount Number:\t\t{final_row["billing_account_number"]} 

Contact
Address:            {final_row["patients_adress"]}        
Email:              {final_row["email"]}      
Telephone:          {final_row["phonenumber"]}\t\t\tMarital status:\t\t{final_row["marital_status"]}

Languages
Language spoken      {final_row["language_spoken"]}\t\t\tLanguage written:\t\t{final_row["language_written"]}
———————————————————————————————————————————————
History
                                                    
HPI:
""",
                                                    axis=1)
        final_df["last_sentence"] = final_df.apply(lambda final_row:
f""" 
The patient was admitted as {final_row["admission_type"]} case. 

Allergies:
{final_row["allergies"]}

Doctors Name:       {final_row["doctors_name"]}       RN: {final_row["nurse_name"]}
Doctors Address:    {final_row["doctors_adress"]}
test_results:       {final_row["test_results"]}
""", axis=1)
        final_df["new_question"] = final_df.apply(
            lambda final_row: phi.change_question(final_row["original_question"], final_row["first_sentence"], final_row["last_sentence"]), axis=1)
        final_df["new_question"] = final_df.apply(lambda final_row: phi.left_align(final_row["new_question"]), axis=1)
        final_df.to_csv(f"{base3}_{type}_long.csv", index=False)
