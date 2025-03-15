__author__ = "Jan Matthias"
__date__ = "15 Mar 2025"

"""
daily_dosage_calculations.py: dose_per_day calculation from the All of Us dataset. 
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

def drug_dosage_processing(df_new):
    """"
    Calculates the dose_per_day for each prescription

    Parameters:
        df_new (pd.DataFrame): The DataFrame containing prescription data. It must include the following columns:
                            - standard_concept_name: Name of the prescribed drug.
                            - quantity: Total quantity of the drug dispensed.
                            - days_supply: Number of days the prescription is intended to last.
    Returns:
        df_new (pd.DataFrame): The input DataFrame with an additional column, dose_per_day, representing the calculated daily dosage for each prescription.
    """
    # DATA PREPROCESSING
    df_new["standard_concept_name_drug"] = df_new["standard_concept_name"]
    df_new = df_new.drop(columns=["standard_concept_name"])
    df_new = df_new.dropna(subset=["quantity", "days_supply"])

    # Convert 'quantity' and 'days_supply' data to numeric format
    df_new.loc[:, 'quantity'] = pd.to_numeric(df_new['quantity'], errors='coerce')
    df_new.loc[:, 'days_supply'] = pd.to_numeric(df_new['days_supply'], errors='coerce')

    # Remove rows where 'days_supply' is 1 
    df_new = df_new[df_new['days_supply'] != 1]

    # Pattern 1: ML + MG/ML cases (e.g., "10 ML fosphenytoin sodium 75 MG/ML Injection")
    pattern_ml_mgml = r"(?P<volume>\d+)\s+ML\s+(?P<drug_name>.+?)\s+(?P<concentration>\d*\.?\d+)\s+MG/ML\s+(?P<drug_type>.+)"

    # Pattern 2: Standard MG cases (e.g., "aspirin 500 MG Tablet")
    pattern_mg = r"(?P<drug_name>.+?)\s+(?P<drug_dose>\d*\.?\d+)\s+MG\s+(?P<drug_type>.+)"

    # Create empty columns
    df_new["drug_name"], df_new["drug_dose"], df_new["drug_type"] = None, None, None

    # Identify rows that match the ML + MG/ML pattern
    mask_ml_mgml = df_new["standard_concept_name_drug"].str.contains(r"\d+\s+ML.*\d+\s+MG/ML", regex=True, na=False)

    # Extract ML + MG/ML data
    df_ml_mgml = df_new.loc[mask_ml_mgml, "standard_concept_name_drug"].str.extract(pattern_ml_mgml)

    # Convert numeric fields
    df_ml_mgml["volume"] = pd.to_numeric(df_ml_mgml["volume"], errors='coerce')
    df_ml_mgml["concentration"] = pd.to_numeric(df_ml_mgml["concentration"], errors='coerce')

    # Compute total MG where applicable and store in drug_dose
    df_ml_mgml["drug_dose"] = df_ml_mgml["volume"] * df_ml_mgml["concentration"]

    # Assign extracted values for ML + MG/ML rows
    df_new.loc[mask_ml_mgml, ["drug_name", "drug_dose", "drug_type"]] = df_ml_mgml[["drug_name", "drug_dose", "drug_type"]]

    # Apply standard MG pattern for remaining rows
    mask_mg = ~mask_ml_mgml  # Select rows that do NOT match ML+MG/ML pattern
    df_mg = df_new.loc[mask_mg, "standard_concept_name_drug"].str.extract(pattern_mg)

    # Assign extracted values for MG-only rows
    df_new.loc[mask_mg, ["drug_name", "drug_dose", "drug_type"]] = df_mg

    # Convert drug_dose to numeric for consistency
    df_new["drug_dose"] = pd.to_numeric(df_new["drug_dose"], errors='coerce')

    # Create 'quantity_per_day' by dividing 'quantity' by 'days_supply', leaving NaN for missing or zero values
    df_new.loc[:, 'quantity_per_day'] = np.where(
        df_new[['quantity', 'days_supply']].notna().all(axis=1),
        df_new['quantity'] / df_new['days_supply'],
        np.nan
    )

    # Convert 'drug_dose' to numeric, retaining NaN values
    df_new.loc[:, 'drug_dose'] = pd.to_numeric(df_new['drug_dose'], errors='coerce')

    # Calculate 'dose_per_day' as 'drug_dose' * 'quantity_per_day', leaving NaN if any component is missing
    df_new.loc[:, 'dose_per_day'] = np.where(
        df_new[['drug_dose', 'quantity_per_day']].notna().all(axis=1),
        df_new['drug_dose'] * df_new['quantity_per_day'],
        np.nan
    )

    # Drop rows where 'drug_name' is NaN (previously handled incorrectly)
    df_new = df_new.dropna(subset=['drug_name'])

    # Remove infinite values resulting from division errors
    df_new.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df_new
