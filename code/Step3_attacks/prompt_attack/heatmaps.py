#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: Step3_attacks/heatmaps.py
Author: Svenja C. Schulze
Last Updated: 2025-09-06
Description: heatmaps creation with three different normalizations
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

class BigHeatmap:

    def __init__(self):
        pass

    @staticmethod
    def global_min_max_scaling(df_heatmap: pd.DataFrame)->pd.DataFrame:
        """
        Normalized Heatmap scaling: Global Min-Max Scaling
        :param df_heatmap: heatmap dataframe
        :return: normalized heatmap dataframe
        """
        df_numeric = df_heatmap.copy()
        for col in df_numeric.columns:
            df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')

        min_val_global = df_numeric.min().min()
        max_val_global = df_numeric.max().max()

        if pd.isna(min_val_global) or pd.isna(max_val_global) or (max_val_global == min_val_global):
            df_normalized_global = df_numeric * 0.0
        else:
            df_normalized_global = (df_numeric - min_val_global) / (max_val_global - min_val_global)
        return df_normalized_global.astype(float)

    @staticmethod
    def normalization_column_sum(df_heatmap: pd.DataFrame)->pd.DataFrame:
        """
        Normalized Heatmap scaling: normalized by column sum
        :param df_heatmap: heatmap dataframe
        :return: normalized heatmap dataframe
        """
        df_numeric = df_heatmap.copy()
        for col in df_numeric.columns:
            df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')

        column_sums = df_numeric.sum(axis=0)
        epsilon = np.finfo(float).eps
        column_sums_safe = column_sums.mask((column_sums == 0) | pd.isna(column_sums), epsilon)

        df_normalized_column_sum = df_numeric.div(column_sums_safe, axis=1)
        return df_normalized_column_sum.astype(float)

    @staticmethod
    def normalization_by_row_sum(df_heatmap):
        """
        Normalized Heatmap scaling: normalized by row sum
        :param df_heatmap: heatmap dataframe
        :return: normalized heatmap dataframe
        """
        df_numeric = df_heatmap.copy()
        for col in df_numeric.columns:
            df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')

        row_sums = df_numeric.sum(axis=1)
        epsilon = np.finfo(float).eps
        row_sums_safe = row_sums.mask((row_sums == 0) | pd.isna(row_sums), epsilon)

        df_normalized_row_sum = df_numeric.div(row_sums_safe, axis=0)
        return df_normalized_row_sum.astype(float)

    @staticmethod
    def plotting_heatmaps(df_heatmap:pd.DataFrame, df_normalized:pd.DataFrame, base:str, title:str, ending:str)->None:
        """
        Plotting the heatmaps according to the given normalization
        :param df_heatmap: dataframe with heatmaps
        :param df_normalized: dataframe with normalized heatmaps
        :param base: base path
        :param title: title of the plot
        :param ending: ending of the path
        :return: None
        """
        common_heatmap_kwargs = {
            "cmap": "YlOrRd",
            "annot": True,
            "cbar": False,
            "fmt": ".2f",
            "annot_kws": {"fontsize": 8, "color": "black", "va": "center", "ha": "center"}
        }
        fig_height = 0.6 * len(df_heatmap)
        plt.figure(figsize=(12, fig_height))
        sns.heatmap(df_normalized, **common_heatmap_kwargs)
        plt.title(f"{title}")
        plt.xlabel("Prompt Columns")
        plt.ylabel("PHI Category")
        plt.tight_layout()
        plt.savefig(f"{base}/{ending}.png")
        plt.close()

    def make_heatmap(self, base: str, counter_dataframe:pd.DataFrame)->None:
        """
        Create heatmaps with three different normalization methods
        :param base: base path
        :param counter_dataframe: counter dataframe
        :return: None
        """
        df_heatmap = counter_dataframe.copy()
        if 'PHI_Category' in df_heatmap.columns:
            df_heatmap = df_heatmap.set_index('PHI_Category')
        else:
            print("Warning: 'PHI_Category' column not found to set as index.")
        for col in df_heatmap.columns:
            df_heatmap[col] = pd.to_numeric(df_heatmap[col], errors='coerce')

        df_heatmap = df_heatmap.fillna(0)
        df_normalized_global = self.global_min_max_scaling(df_heatmap)
        df_normalized_column_sum = self.normalization_column_sum(df_heatmap)
        df_normalized_row_sum = self.normalization_by_row_sum(df_heatmap)

        self.plotting_heatmaps(df_heatmap, df_normalized_global, base, "Heatmap (Global Min-Max Normalized Counts)", "global_min_max_heatmap")
        self.plotting_heatmaps(df_heatmap, df_normalized_column_sum, base, "Heatmap (Normalized by Column Sum - Proportions)", "df_normalized_column_sum")
        self.plotting_heatmaps(df_heatmap, df_normalized_row_sum, base, "Heatmap (Normalized by Row Sum - Proportions)", "df_normalized_row_sum")


