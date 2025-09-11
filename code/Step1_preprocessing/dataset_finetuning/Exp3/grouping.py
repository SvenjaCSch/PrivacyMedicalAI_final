from personal_data_adults import Personal_adults
from personal_data_children import PersonalChildren
import pandas as pd
import random

class Grouping:
    def __init__(self):
        self.adults = Personal_adults()
        self.children = PersonalChildren()

    @staticmethod
    def get_age_group(age):
        if age <= 2:
            return "0-2"
        elif 3 <= age <= 12:
            return "3-12"
        elif 13 <= age <= 18:
            return "13-18"
        else:
            return ">18"

    @staticmethod
    def create_combined_groups(data: pd.DataFrame)->dict[tuple, pd.DataFrame]:
        """
        Create combined groups according the person types. Person type can be either adult or children.
        The returned data is divided by age group and gender of the patient.
        :param data: dataset
        :param person_type: adults or children
        :return:  combined groups with gender and age groups
        """

        return {
            (gender, age_group): group_df.sample(frac=1, random_state=42).reset_index(drop=True)
            for (gender, age_group), group_df in data.groupby(["Gender", "AgeGroup"])
        }

    @staticmethod
    def get_gender_weights(person_type:str)->dict:
        """
        Return gender weights (can be extended for other groups).
        :param person_type: adults or children
        :return:  gender weights
        """
        if person_type == "adults":
            return {
                "Female": 0.51,
                "Male": 0.49,
            }
        else:
            return {
                0: 0.49,
                1: 0.51,
            }

    def generate_weighted_random_dataset(self, combined_groups:dict, combined_indices:dict, gender_weights:dict, person_type:str)->pd.DataFrame:
        """
        Generate the final dataset using weighted sampling by gender.
        :param combined_groups: combined groups
        :param combined_indices: combined indices
        :param gender_weights: gender weights
        :param person_type: adults or children
        :return: dataset with weighted samples
        """
        final_df = pd.DataFrame()
        person_id = 1

        for _ in range(50):
            for group_size in range(1, 15):
                eligible_keys = self.get_eligible_keys(combined_groups, combined_indices, group_size)

                if not eligible_keys:
                    continue

                selected_key = self.select_group_key(eligible_keys, gender_weights)

                if not selected_key:
                    print(f"No valid key could be selected for {person_type}. Skipping.")
                    continue

                group_df = combined_groups[selected_key]
                start_idx = combined_indices[selected_key]
                end_idx = start_idx + group_size

                person_rows = group_df.iloc[start_idx:end_idx].reset_index(drop=True)
                combined_indices[selected_key] = end_idx

                group = self.build_group_entry(person_rows, selected_key, person_type)

                final_df = pd.concat([final_df, group], ignore_index=True)
                person_id += 1

        return final_df

    @staticmethod
    def get_eligible_keys(combined_groups:dict, combined_indices:dict, group_size:int)->list:
        """
        Return keys that have enough remaining samples for a group of given size.
        :param combined_groups: combined groups
        :param combined_indices: combined indices
        :param group_size: group size
        """
        return [
            key for key, df in combined_groups.items()
            if combined_indices[key] + group_size <= len(df)
        ]

    @staticmethod
    def select_group_key(eligible_keys:list, gender_weights:dict)->str|None:
        """
        Select a (gender, age_group) key based on gender weights.
        :param eligible_keys: eligible keys
        :param gender_weights: gender weights
        :return: selected key
        """
        eligible_genders = list({key[0] for key in eligible_keys})
        gender_probs = [gender_weights.get(g, 0) for g in eligible_genders]
        total_weight = sum(gender_probs)

        if total_weight == 0:
            return None

        normalized_probs = [w / total_weight for w in gender_probs]
        selected_gender = random.choices(eligible_genders, weights=normalized_probs, k=1)[0]
        gender_filtered_keys = [key for key in eligible_keys if key[0] == selected_gender]

        if not gender_filtered_keys:
            return None

        return random.choice(gender_filtered_keys)

    def build_group_entry(self, person_rows: pd.Series, selected_key: int,person_type:str)->pd.DataFrame:
        """
        Construct the full group (first + rest) DataFrame.
        :param person_rows: person rows
        :param selected_key: selected key
        :param person_type: adults or children
        :return: full group
        """
        selected_gender, selected_age_group = selected_key

        if person_type == "adults":
            first_row = self.adults.make_dataframe_first(person_rows.iloc[0], selected_age_group)
        else:
            first_row = self.children.make_dataframe_first(person_rows.iloc[0], selected_age_group)
        group = first_row

        for j in range(1, len(person_rows)):
            if person_type == "adults":
                rest_row = self.adults.make_dataframe_rest(person_rows.iloc[j], first_row.iloc[0])
            else:
                rest_row = self.children.make_dataframe_rest(first_row.iloc[0])
            group = pd.concat([group, rest_row], ignore_index=True)

        return group

    def manage_dataset(self, med_data:pd.DataFrame, person_type:str)->pd.DataFrame:
        """
        Manage datasets iwith grouping. The idea is to get enough data for each age and gender. Gender is according the gender population in the USA.
        First: Define Agegroup according to the different ages of the patients in the dataset.
        Second: Get combined groups
        Gender weights is an extra PHIs, that was implemented but not further used. For future work it might still be relevant
        :param med_data:
        :param person_type:
        :return:
        """
        med_data["AgeGroup"] = med_data["Age"].apply(self.get_age_group)

        combined_groups = self.create_combined_groups(med_data, person_type)
        for key, df in combined_groups.items():
            print(f"Group {key} has {len(df)} samples for {person_type}.")
        combined_indices = {key: 0 for key in combined_groups}

        gender_weights = self.get_gender_weights(person_type)

        final_df = self.generate_weighted_random_dataset(combined_groups, combined_indices, gender_weights,
                                                                  person_type)
        return final_df