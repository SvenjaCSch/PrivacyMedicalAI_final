import os
import pandas as pd
from dotenv import load_dotenv
from personal_data_adults import Personal_adults
from personal_data_children import Personal_children
from combine_datasets import Combine
from disease_symptom_data import Disease
from get_symptoms import Symptoms
from grouping import Grouping
from personal_data import PersonalData
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
        self.child = Personal_children()
        self.combine = Combine()
        self.disease = Disease()
        self.symptoms = Symptoms()
        self.grouping = Grouping()
        self.personal = PersonalData()

    @staticmethod
    def get_dataset(url:str)->pd.DataFrame:
        """
        Load dataset from URL and return a random sample.
        :param url: url of dataset
        :return: dataset as pandas dataframe
        """
        data = pd.read_csv(url)
        return data

    def left_align(self, textblock):
        """
        left alignment of textblock
        used for the "new_question"
        :param textblock: whole textblock
        :return: left aligned textblock
        """
        left_aligned_lines = [line.lstrip() for line in textblock.splitlines()]
        left_aligned_text = "\n".join(left_aligned_lines)
        return left_aligned_text

    def manage_data(self, base: str, adults_url:str, child_plus_symptoms_url:str, adult_symptoms_url:str, disease_url:str)-> None:
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
        disease = self.get_dataset(disease_url)
        print(f"disease: \n{disease.head()}")
        
        adults = self.get_dataset(adults_url)
        print(f"adults: \n{adults.head()}")

        child_plus_symptoms = self.get_dataset(child_plus_symptoms_url)
        child_plus_symptoms = child_plus_symptoms[child_plus_symptoms["Age"]<=18]
        print(f"child_plus_symptoms: \n{child_plus_symptoms.head()}")

        child_plus_symptoms['Age_years_int'] = child_plus_symptoms['Age'].astype(int)
        adult_symptoms = self.get_dataset(adult_symptoms_url)
        print(f"adult_symptoms: \n{adult_symptoms.head()}")


        adult_symp = self.disease.manage_dataset(adult_symptoms)
        print(f"adult_symp: \n{adult_symp.head()}")
        adult_symp['Age_years_int'] = adult_symp['Age'].astype(int)

        adult_phi = self.grouping.manage_dataset(adults, "adults")
        children_phi = self.grouping.manage_dataset(child_plus_symptoms, "children")
        print(f"children_phi: \n{children_phi.head()}")
        print(f"adult_phi: \n{adult_phi.head()}")

        combined_phi =self.combine.combine(adult_phi, children_phi)
        combined_phi = self.combine.shuffle_grouped(combined_phi)
        print(f"combined_phi unique gender: \n{combined_phi["gender"].unique()}")

        print(f"combined_phi: \n{combined_phi.head()}")
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
        
        Chief Complaint 
        Patient presents with {row["symptoms"]}. The patient was admitted as {row["admission_type"]} case. 
        
        """,
                                                    axis=1)
        combined_phi["last_sentence"] = combined_phi.apply(lambda row:
                                                   f""" 
        Doctors Name:       {row["doctors_name"]}       RN: {row["nurse_name"]}
        Doctors Address:    {row["doctors_adress"]}
        test_results:       {row["test_results"]}
        """, axis=1)
        combined_phi["new_question"] = combined_phi.apply(
            lambda row: self.personal.change_question(row["first_sentence"], row["last_sentence"]),
            axis=1)
        combined_phi["new_question"] = combined_phi.apply(lambda row: self.left_align(row["new_question"]), axis=1)


        train, test = self.combine.make_train_test_split(combined_phi)
        print(f"train: \n{train.head()}")
        self.combine.save_train_test(train, test)
        train_shuffled, test_shuffled = self.combine.shuffle_data(train, test)
        combined_phi.to_csv(f"{base}\dataset_single_entries_per_person.csv")
        train_shuffled.to_csv(f"{base}\dataset_single_entries_per_person_train_shuffled.csv")
        test_shuffled.to_csv(f"{base}\dataset_single_entries_per_person_test_shuffled.csv")

if __name__ == "__main__":
    main = Main()
    base = rf"{main.BASE}data\Step1_preprocessing\dataset_finetuning\Exp4"
    adult = rf"{main.BASE}PrivacyMedicalAI\dataset\kaggle\healthcare_dataset.csv"
    adult_symptons = rf"{main.BASE}dataset\kaggle\Disease_symptom_and_patient_profile_dataset.csv"
    children_symptoms = rf"{main.BASE}dataset\kaggle\dataPaperStackingLOS_1.csv"
    diseases = rf"{main.BASE}dataset\kaggle\Diseases_Symptoms.csv"
    main.manage_data(base, adult, children_symptoms, adult_symptons, diseases)



