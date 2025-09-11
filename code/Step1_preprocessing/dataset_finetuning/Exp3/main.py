import os
import re
import pandas as pd
from dotenv import load_dotenv
from personal_data_adults import Personal_adults
from personal_data_children import PersonalChildren
from combine_datasets import Combine
from disease_symptom_data import Disease
from get_symptoms import Symptoms
from grouping import Grouping
from personal_data import PersonalData
import numpy as np
import random
""" 
Dataset based on:
https://www.kaggle.com/datasets/staniherstaniher/data-patients-los-in-a-semi-urban-hospital?select=dataPaperStackingLOS_1.csv
https://www.kaggle.com/datasets/uom190346a/disease-symptoms-and-patient-profile-dataset
https://www.kaggle.com/datasets/prasad22/healthcare-dataset
https://www.kaggle.com/datasets/aadyasingh55/disease-and-symptoms

"""

class Main:
    def __init__(self):
        load_dotenv()
        self.BASE = os.getenv("BASE")
        self.adult = Personal_adults()
        self.child = PersonalChildren()
        self.combine = Combine()
        self.disease = Disease()
        self.symptoms = Symptoms()
        self.grouping = Grouping()
        self.personal = PersonalData()

    @staticmethod
    def get_dataset(url: str) -> pd.DataFrame:
        """
        Load dataset from URL and return a random sample.
        :param url: url of dataset
        :return: dataset as pandas dataframe
        """
        data = pd.read_csv(url)
        return data


    def strip_last_question(self, questions:str)->tuple[str,str]:
        """
        Extract the last question to receive the final question and a statement without the final question.
        Also excludes the question mark in the end
        :param questions: question
        :return: last question and part without last question
        """
        extracted_final_question = ""
        hpi_without_final_question = questions.strip()
        last_question_mark = questions.rfind('?')

        if last_question_mark != -1:
            text_before_question_mark = questions[:last_question_mark]
            last_period_before_question_mark = text_before_question_mark.rfind('.')
            last_newline_before_question_mark = text_before_question_mark.rfind('\n')
            if last_period_before_question_mark != -1 and last_newline_before_question_mark != -1:
                start_extract = max(last_period_before_question_mark, last_newline_before_question_mark) + 1
            elif last_period_before_question_mark != -1:
                start_extract = last_period_before_question_mark + 1
            elif last_newline_before_question_mark != -1:
                start_extract = last_newline_before_question_mark + 1
            else:
                start_extract = 0
            if start_extract < 0:
                start_extract = 0
            extracted_final_question = questions[start_extract: last_question_mark + 1].strip()

            hpi_without_final_question = questions.replace(extracted_final_question, "", 1).strip()

        else:
            print("Warning: No question mark found in the 'original_hpi_and_question' string.")
        return extracted_final_question, hpi_without_final_question

    def get_hpi_age_grouping(self, hpi:pd.Dataframe)->str:
        """
        Group the HPIs in age groups
        :param hpi: HPI data by the MedQA questions
        :return: age groups
        """
        pattern = re.compile(r'(\d+)[-\s]?year')
        match = pattern.search(hpi["question"])

        if not match:
            return "unknown"

        age = int(match.group(1))
        if age <= 2:
            return "0-2"
        elif 3 <= age <= 12:
            return "3-12"
        elif 13 <= age <= 18:
            return "13-18"
        else:
            return ">18"

    def get_hpi(self, data:pd.DataFrame)->str:
        """
        Creating the HPI part for a longer EHR than for datasets 2 and 4
        :param data: dataframe
        :return: prompt
        """
        if data["question"] == "NaN":
            prompt = \
                f"""
                {data["first_sentence"]}
                Chief Complaint 
                Patient presents with {data["symptoms"]}. The patient was admitted as {data["admission_type"]} case. 
                {data["last_sentence"]}
                """

        else:
            extracted_final_question, hpi_without_final_question = self.strip_last_question(data["question"])
            prompt = \
                f"""
                {data["first_sentence"]}
                HPI:
                {hpi_without_final_question}
                {data["last_sentence"]}
                {extracted_final_question}
                """
        return prompt

    def get_gender(self, random_question_row:pd.Series)->str:
        """
        Extract the gender from the question
        :param random_question_row: question randomly extracted for the HPI merge to test the gender
        :return: gender of the patient in the question
        """
        females = ["woman", "female", "girl", "lady"]
        question_text = random_question_row['question']
        first_sentence = question_text.split(".")[0]
        for female in females:
            if female in first_sentence:
                return "Female"
        return "Male"

    def merge_data(self, hpi:str, combined_phi:pd.DataFrame)->pd.DataFrame:
        """
        Merge the data with the combined dataset and the HPI-part.
        First, the hpis got grouped by age_group
        Every time the age groups of hpi and combined_phi are aligned, a randowm row is sampled.
        The gender is extracted and analyzed. If same gender than get new information merged otherwise not
        :param hpi: hpi part of the dataset
        :param combined_phi: combined dataset
        :return: final dataset combined with all datasets
        """
        print(f"Original Length of hpi: {len(hpi)}")
        print(f"Original Length of combined_phi: {len(combined_phi)}")

        merged_rows = []
        hpi_grouped = hpi.groupby('age_group')

        for index, phi_row in combined_phi.iterrows():
            age_group = phi_row['age_group']

            if age_group in hpi_grouped.groups:
                available_questions = hpi_grouped.get_group(age_group)
                while True:
                    random_question_row = available_questions.sample(n=1, random_state=random.randint(0, 1000000)).iloc[
                        0]
                    gender = self.get_gender(random_question_row)
                    if gender == phi_row['gender']:
                        new_row = phi_row.copy()

                        new_row['question'] = random_question_row['question']
                        new_row['original_answer'] = random_question_row['answer']
                        new_row['choices'] = random_question_row['options']
                        new_row['answer_idx'] = random_question_row['answer_idx']

                        merged_rows.append(new_row)
                        break

            else:
                print(f"Warning: No HPI questions found for age_group: {age_group}")
                new_row = phi_row.copy()
                new_row['question'] = np.nan
                new_row['answer'] = np.nan
                new_row['choices'] = np.nan
                new_row['answer_idx'] = np.nan
                merged_rows.append(new_row)

        final_combined_phi = pd.DataFrame(merged_rows)

        hpi_summary = hpi.groupby('age_group')['question'].nunique().reset_index()
        hpi_summary.rename(columns={'question': 'unique_questions_count'}, inplace=True)
        print("HPI Summary (unique questions):")
        print(hpi_summary)
        print(f"Length of aggregated hpi: {len(hpi_summary)}")
        print(f"Does aggregated hpi have duplicate age_groups? {hpi_summary['age_group'].duplicated().any()}")

        print(f"\nFinal Length of combined_phi after merge: {len(final_combined_phi)}")
        print("\nFinal Merged DataFrame (partial view):")
        print(final_combined_phi.head())

        if len(final_combined_phi) == len(combined_phi):
            print("\nMerge successful! Length of combined_phi maintained.")
        else:
            print("\nWarning: Length mismatch. Something might still be off.")

        return final_combined_phi

    def change_age(self, data:pd.DataFrame)->pd.DataFrame:
        """
        Change the age: Since I have age-groups and not the exact age,
        I need to change the age of the question to align with the rest of the patient data.
        The real age is the Dat of admission - the DOB
        :param data:
        :return:
        """
        pattern = re.compile(r'(\d+)[-\s]?year')
        match = pattern.search(data["question"])

        if pd.isna(data["date_of_birth"]) or pd.isna(data["admission_date"]):
            print(f"Warning: Missing or invalid date for row: {data.name}")
            return "unknown_date_error"

        if not match:
            return "unknown"

        birth_year = data["date_of_birth"].year
        admission_year = data["admission_date"].year
        new_age = admission_year - birth_year

        new_question = re.sub(r'(\d+)([-\s]?year)', fr'{new_age}\2', data["question"], 1)
        print(new_question)

        return new_question

    def left_align(self, textblock:str)->str:
        """
        left alignment of textblock
        used for the "new_question"
        :param textblock: whole textblock
        :return: left aligned textblock
        """
        left_aligned_lines = [line.lstrip() for line in textblock.splitlines()]
        left_aligned_text = "\n".join(left_aligned_lines)
        return left_aligned_text

    def manage_data(self, base: str, adults_url:str, child_plus_symptoms_url:str, adult_symptoms_url:str, disease_url:str, hpi:pd.Dataframe)-> None:
        """
        Manages the different datasets by loading and preprocess them
        Children only under 18
        Grouping the datasets to receive data by age_group and gender for receiving more potential entries for one patient
        Afterwards receiving symptoms and diseases based on age and gender
        receive contact person
        Make EHR entries out of the data
        :param base: url for storing the received datasets
        :param adults_url: url for the adults dataset
        :param child_plus_symptoms_url:  url for the child plus symptoms dataset
        :param adult_symptoms_url: url for the adults symptoms dataset
        :param disease_url: url for the disease dataset
        :return: None
        """
        hpi = hpi.sample(frac=1).reset_index(drop=True)
        hpi["age_group"] = hpi.apply(lambda row: self.get_hpi_age_grouping(row), axis=1)

        disease = self.get_dataset(disease_url)
        adults = self.get_dataset(adults_url)

        child_plus_symptoms = self.get_dataset(child_plus_symptoms_url)
        child_plus_symptoms = child_plus_symptoms[child_plus_symptoms["Age"]<=18]

        child_plus_symptoms['Age_years_int'] = child_plus_symptoms['Age'].astype(int)
        adult_symptoms = self.get_dataset(adult_symptoms_url)

        adult_symp = self.disease.manage_dataset(adult_symptoms)
        adult_symp['Age_years_int'] = adult_symp['Age'].astype(int)

        adult_phi = self.grouping.manage_dataset(adults, "adults")
        children_phi = self.grouping.manage_dataset(child_plus_symptoms, "children")

        combined_phi =self.combine.combine(adult_phi, children_phi)
        combined_phi = self.combine.shuffle_grouped(combined_phi)

        combined_phi[["disease", "symptoms"]] = combined_phi.apply(
            lambda row: pd.Series(self.symptoms.get_symptoms(row, child_plus_symptoms, "children")),
            axis=1
        )
        combined_phi[["disease", "symptoms"]] = combined_phi.apply(
            lambda row: pd.Series(self.symptoms.get_symptoms(row, adult_symp, "adults")),
            axis=1
        )
        combined_phi["symptoms"] = combined_phi.apply(
            lambda row: self.symptoms.get_symptoms_by_disease(row["disease"], row["symptoms"], disease, row['gender']), axis=1)

        combined_phi["contact_person"] = combined_phi.apply(
            lambda row: self.personal.get_contact(row["age_group"], row["surname"]), axis=1)

        combined_phi["first_sentence"] = combined_phi.apply(lambda row:
        f"""———————————————————————————————————————————————
        Patient Name:       {row["forename"]} {row["surname"]}\t\tMRN:\t\t\t\t{row["medical_record_number"]} 
        DOB:                {row["date_of_birth"]}\t\t\tLegal sex:\t\t\t{row["gender"]} 
        Date of Visit:      {row["admission_date"]}\t\t\tContact Person:\t\t{row["contact_person"]}
        Date of Discharge:  {row["discharge_date"]}\t\t\tSocial security:\t{row["social_secruity_number"]}
        Health Plan Number: {row["health_plan_number"]}\t\t\tAccount Number:\t\t{row["billing_account_number"]} 
        
        Contact
        Address:            {row["patients_adress"]}        
        Email:              {row["email"]}      
        Telephone:          {row["phonenumber"]}\t\t\tMarital status:\t\t{row["marital_status"]}
        
        Languages
        Language spoken      {row["language_spoken"]}\t\t\tLanguage written:\t\t{row["language_written"]}
        ———————————————————————————————————————————————
        History
    
        """,
                                                    axis=1)
        combined_phi["last_sentence"] = combined_phi.apply(lambda row:
        f"""The patient was admitted as {row["admission_type"]} case.  
        
        Allergies:
        {row["allergies"]}
        
        Doctors Name:       {row["doctors_name"]}       RN: {row["nurse_name"]}
        Doctors Address:    {row["doctors_adress"]}
        test_results:       {row["test_results"]}
        """, axis=1)

        combined_phi = self.merge_data(hpi, combined_phi)
        combined_phi['date_of_birth'] = pd.to_datetime(combined_phi['date_of_birth'], errors='coerce')
        combined_phi['admission_date'] = pd.to_datetime(combined_phi['admission_date'], errors='coerce')
        combined_phi["question"] = combined_phi.apply(lambda row: self.change_age(row), axis=1)
        combined_phi["new_question"] = combined_phi.apply(lambda row: self.get_hpi(row), axis=1)
        combined_phi["new_question"] = combined_phi.apply(lambda row: self.left_align(row["new_question"]), axis=1)


        train, test = self.combine.make_train_test_split(combined_phi)
        print(f"train: \n{train.head()}")
        self.combine.save_train_test(train, test)
        train_shuffled, test_shuffled = self.combine.shuffle_data(train, test)
        combined_phi.to_csv(f"{base}\dataset_multiple_entries_per_person_long.csv")
        train_shuffled.to_csv(f"{base}\dataset_multiple_entries_per_person_train_shuffled_long.csv")
        test_shuffled.to_csv(f"{base}\dataset_multiple_entries_per_person_test_shuffled_long.csv")


if __name__ == "__main__":
    main = Main()
    base = rf"{main.BASE}data\Step1_preprocessing\dataset_finetuning\Exp3"
    adult = rf"{main.BASE}PrivacyMedicalAI\dataset\kaggle\healthcare_dataset.csv"
    adult_symptons = rf"{main.BASE}dataset\kaggle\Disease_symptom_and_patient_profile_dataset.csv"
    children_symptoms = rf"{main.BASE}dataset\kaggle\dataPaperStackingLOS_1.csv"
    diseases = rf"{main.BASE}dataset\kaggle\Diseases_Symptoms.csv"


    traintestval = ["train", "test", "validation"]
    medqa = pd.DataFrame()
    for type in traintestval:
        dataset_url = f"{base}_{type}_final_selection.csv"
        dataset = main.get_dataset(dataset_url)
        medqa = main.combine.combine(medqa, dataset)

    main.manage_data(base, adult, children_symptoms, adult_symptons, diseases, medqa)




