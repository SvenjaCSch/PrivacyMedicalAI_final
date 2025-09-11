#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: get_symptoms.py
Author: Svenja C. Schulze
Last Updated: 2025-09-06
Description: get_symptoms defined by diseases
"""

import random

import pandas as pd
from faker import Faker
import datetime
import re
import numpy as np

class PersonalData:
    def __init__(self):
        self.faker = Faker('en_US')

    def random_date(self,start_date:datetime.date = None, end_date:datetime.date = None)->datetime.date:
        """
        Get random date between start_date and end_date
        :param start_date: start date
        :param end_date: end date
        :return: random date between start_date and end_date
        """
        if start_date is None:
            start_date = self.faker.date_this_year(before_today=True)
        if end_date is None:
            end_date = datetime.date.today()
        return self.faker.date_between_dates(date_start=start_date, date_end=end_date)

    def get_forename(self,string)->str:
        """
        Get forename
        :param string: string of gender
        :return: forename
        """
        if string == "Female":
            return self.faker.first_name_female()
        elif string == "Male":
            return self.faker.first_name_male()
        else:
            return self.faker.first_name()

    @staticmethod
    def make_phone_number()->str:
        """
        Make phone number
        :return: phone number
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
    def get_social_security()->str:
        """
        Get social security number
        :return: ssn
        """
        return f"{random.randint(100,999)}-{random.randint(10,99)}-{random.randint(1000,9999)}"

    @staticmethod
    def get_birthyear(age: str, admission:datetime.date)->int:
        """
        Get birthyear
        :param age: age
        :param admission: admission date
        :return: birthyear
        """
        age = int(age)
        admission_year = int(str(admission).split('-')[0])
        birth_year = admission_year - age
        return birth_year

    @staticmethod
    def get_marital_status()->list:
        """
        Get marital status based on probabilities
        :return: marital status
        """
        return np.random.choice(["Married", "Single", "Divorced", "Widowed"], p=[0.3, 0.5, 0.15, 0.05])

    def get_random_birthday(self, age:str, admission:datetime.date)->int|str:
        birthyear = self.get_birthyear(age, admission)
        if birthyear == "unknown":
            return "unknown"

        admission_month = int(str(admission).split('-')[1])
        month = random.randint(1, admission_month)

        if month == 2:
            day = random.randint(1, 29 if birthyear % 4 == 0 else 28)
        else:
            day = random.randint(1, 30)

        birthday = f"{birthyear}-{month:02d}-{day:02d}"
        return birthday

    @staticmethod
    def more_exposure_per_person(data:pd.DataFrame, start:int, person_id:int, end:int):
        """
        Extract a specific amount of rows for a certain person with fitting gender and age
        :param data: data
        :param start: start date
        :param person_id: person id
        :param end: end date
        :return: person rows for one person
        """
        assignments = {}
        if end > len(data):
            raise ValueError(f"Not enough rows for {person_id}")
        assignments[f"Person {person_id}"] = data.iloc[start:end]
        return assignments

    def get_address(self)->str:
        """
        get address
        :return: address
        """
        address = self.faker.address()
        adress_new = re.sub("\n", " ", address)
        return adress_new

    def get_contact(self,age:str ,surname:str, status:str = None)->str:
        """
        get contact person depending on the age group, the surname and the marital status
        :param age: age group
        :param surname: surname
        :param status: martial status
        :return: contact person
        """
        contact_male_name = self.get_forename("male")
        contact_female_name = self.get_forename("female")
        if age != ">18":
            return f"{contact_male_name} {surname} (father), {contact_female_name} {surname} (mother)"
        else:
            gender = random.choice(["male, female"])
            if gender == "male":
                if status == "Married":
                    return f"{contact_male_name} {surname} (husband)"
                else:
                    return f"{contact_male_name} {self.faker.last_name()} ({random.choice(["boyfriend", "friend", "fiancee"])})"
            else:
                if status == "Married":
                    return f"{contact_female_name} {surname} (wife)"
                else:
                    return f"{contact_female_name} {self.faker.last_name()} ({random.choice(["girlfriend", "friend", "fiancee"])})"

    @staticmethod
    def change_question(new_sentence:str, last_sentence:str)->str:
        """
        embedding the question
        :param new_sentence: new sentence
        :param last_sentence: last sentence
        :return: new sentence
        """
        new_question = f"{new_sentence}\n{last_sentence}"
        print("################################################")
        print(new_question)
        return new_question




