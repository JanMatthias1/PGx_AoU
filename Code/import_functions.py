__author__ = "Jan Matthias"
__date__ = "15 Mar 2025"

"""
import_functions.py: functions needed for data wrangling and statistical analysis 
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import rankdata
from scipy.stats import spearmanr
from scipy.stats import chi2_contingency
from statsmodels.stats.proportion import proportions_ztest
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import re
import random

def clean_pgx_data(file_path):
    """
    Cleans a PGx dataset by applying standard preprocessing steps.

    Parameters:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Load dataset
    df = pd.read_csv(file_path, low_memory=False)

    # Remove rows where 'phenotype' column contains 'Likely ' (case-insensitive)
    df = df.applymap(lambda x: np.nan if isinstance(x, str) and x.lower().startswith("likely") else x)

    # Define metabolizer pattern and extract values
    metabolizer_pattern = r'(?i)(Normal|Intermediate|Poor|Rapid|Ultrarapid)'
    columns_to_clean = ["CYP2D6", "CYP2C19", "CYP2C9", "CYP2B6", "CYP3A5", "CYP4F2"]

    for col in columns_to_clean:
        df[col] = df[col].str.extract(metabolizer_pattern, expand=False).str.capitalize()

    # Remove rows where 'dose_per_day' is infinite or zero
    df = df[~df['dose_per_day'].isin([float('inf'), 0])]

    # Remove rows where 'standard_concept_name_drug' contains '/'
    df = df[~df['standard_concept_name_drug'].str.contains(r'\s/\s', na=False)]

    # Map 'sex_at_birth' values to numeric
    df['sex_at_birth'] = df['sex_at_birth'].replace({'Female': 0.0, 'Male': 1.0})

    # Remove unwanted values from 'sex_at_birth'
    unwanted_values = ['PMI: Skip', 'No matching concept', 'I prefer not to answer', 'Intersex', np.nan]
    df = df[~df['sex_at_birth'].isin(unwanted_values)] 

    # Convert 'sex_at_birth' to numeric
    df['sex_at_birth'] = pd.to_numeric(df['sex_at_birth'], errors='coerce')

    return df

def perform_kruskal_test(grouped_data, expected_groups):
    
    groups = [group["Adjusted_Dosage"].values for _, group in grouped_data]
    
    if len(groups) == expected_groups and all(len(g) >= 10 for g in groups):
        statistic, p_value = stats.kruskal(*groups)
        return round(p_value, 5), round(statistic, 5),  "3_KW"
    return "Skipped", "Skipped", "3_KW"

def perform_wilcoxon_test(grouped_data, expected_groups):
    """
    Perform a Wilcoxon rank-sum test if there are exactly two groups with sufficient sample size.
    """
    groups = [group["Adjusted_Dosage"].values for _, group in grouped_data]

    if len(groups) == expected_groups and all(len(g) >= 10 for g in groups):  # Ensure exactly two groups with enough data
        statistic, p_value = stats.ranksums(*groups)
        return round(p_value, 5), round(statistic, 5), "2_Wilcoxon"
    
    return "Skipped", "Skipped", "2_Wilcoxon"

def find_metabolizing_enzyme(drug_name, CYP_gene, CYP_drug, drug_to_enzymes):
    """ Determine the metabolizing enzyme(s) for a given drug. """
    metabolizing_enzymes = drug_to_enzymes[drug_name]
    
    if CYP_gene in metabolizing_enzymes:
        return CYP_gene
    elif len(metabolizing_enzymes) > 1 and CYP_gene not in metabolizing_enzymes:
        return "/".join(sorted(metabolizing_enzymes))
    else:
        return CYP_drug

def remove_outliers(df, column="dose_per_day", lower_quantile=0.25, upper_quantile=0.75, iqr_multiplier=1.5):
    """remove outlier in drug dosage data. """
    Q1 = df[column].quantile(lower_quantile)
    Q3 = df[column].quantile(upper_quantile)
    IQR = Q3 - Q1

    df_filtered = df[
        (df[column] >= Q1 - iqr_multiplier * IQR) &
        (df[column] <= Q3 + iqr_multiplier * IQR)
    ].drop_duplicates(subset='person_id', keep='last').copy()

    return df_filtered

def calculate_adjusted_dosage(mean_dose_per_day, pca_columns):
    """ Perform covariate-adjusted regression analysis and return adjusted dosage values. """
    X = mean_dose_per_day[["sex_at_birth", "BMI", "age_calculated_years"] + pca_columns]
    X = sm.add_constant(X)

    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[pca_columns + ["BMI", "age_calculated_years"]] = scaler.fit_transform(X_scaled[pca_columns + ["BMI", "age_calculated_years"]])

    y = mean_dose_per_day['dose_per_day']
    model = sm.OLS(y, X_scaled).fit()

    mean_dose_per_day['Adjusted_Dosage'] = model.resid 
    return mean_dose_per_day, model

def update_summary_statistics(mean_dose_per_day, CYP_drug, CYP_gene, drug_name, summary_stats):
    """ Compute and store summary statistics for a given drug and enzyme pair. """
    median_dose = mean_dose_per_day["dose_per_day"].median()
    sd_dose = mean_dose_per_day["dose_per_day"].std()
    min_dose = mean_dose_per_day["dose_per_day"].min()
    max_dose = mean_dose_per_day["dose_per_day"].max()
    
    summary_stats.loc[len(summary_stats)] = [CYP_drug, CYP_gene, drug_name, median_dose, sd_dose, min_dose, max_dose]
    
    return summary_stats

def store_feature_importance(model, CYP_drug, CYP_gene, drug_name, pca_columns, feature_importance):
    """ Extracts p-values and beta coefficients from a regression model and appends them to feature_importance. """
    new_row = {
        "CYP_drug": CYP_drug, 
        "CYP_gene": CYP_gene, 
        "drug_name": drug_name,
        "sex_at_birth_p": model.pvalues.get("sex_at_birth", None),
        "sex_at_birth_beta": model.params.get("sex_at_birth", None),
        "BMI_p": model.pvalues.get("BMI", None),
        "BMI_beta": model.params.get("BMI", None),
        "age_calculated_years_p": model.pvalues.get("age_calculated_years", None),
        "age_calculated_years_beta": model.params.get("age_calculated_years", None)
    }

    # Add PCA feature p-values and beta coefficients dynamically
    for pca_col in pca_columns:
        new_row[f"{pca_col}_p"] = model.pvalues.get(pca_col, None)
        new_row[f"{pca_col}_beta"] = model.params.get(pca_col, None)

    # Append new row to feature_importance DataFrame
    return pd.concat([feature_importance, pd.DataFrame([new_row])], ignore_index=True)

def store_statistical_results(mean_dose_per_day, CYP_drug_modified, CYP_gene, drug_name, data_stat):
    """ Perform Kruskal-Wallis test and store statistical results if sufficient data is available. """
    participants = len(mean_dose_per_day)
    metabolizer_groups_str = ", ".join(map(str, mean_dose_per_day["metabolizer_group"].unique()))
    
    if participants >= 50:
        p_value, statistic = perform_kruskal_test(mean_dose_per_day.groupby("metabolizer_group"), 3)
        data_stat.loc[len(data_stat)] = [CYP_drug_modified, CYP_gene, drug_name, p_value, statistic,
                                         len(mean_dose_per_day["metabolizer_group"].unique()), 
                                         participants, "3_KW",
                                         metabolizer_groups_str]
    else:
        data_stat.loc[len(data_stat), :] = [CYP_drug_modified, CYP_gene, drug_name, "Skipped", "Skipped", 
                                            "Skipped", participants, "Skipped", metabolizer_groups_str]
        
