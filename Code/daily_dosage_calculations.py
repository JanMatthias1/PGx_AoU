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
    """
    Calculates the dose_per_day for each prescription.

    Parameters:
        df_new (pd.DataFrame): DataFrame with prescription data including:
            - standard_concept_name
            - quantity
            - days_supply

    Returns:
        pd.DataFrame: Modified DataFrame with:
            - drug_name
            - drug_dose
            - drug_type
            - quantity_per_day
            - dose_per_day
    """
    # Copy and cleanup
    df = df_new.copy()
    df.rename(columns={"standard_concept_name": "standard_concept_name_drug"}, inplace=True)
    df.dropna(subset=["quantity", "days_supply"], inplace=True)

    # Convert columns to numeric
    df["quantity"] = pd.to_numeric(df["quantity"], errors='coerce')
    df["days_supply"] = pd.to_numeric(df["days_supply"], errors='coerce')

    # Remove rows with 1-day supply
    df = df[df["days_supply"] != 1]

    # Init columns
    df[["drug_name", "drug_dose", "drug_type"]] = None

    # Define patterns
    patterns = {
    "ml_mgml": r"(?P<volume>\d*\.?\d+)\s+ML\s+(?P<drug_name>.+?)\s+(?P<concentration>\d*\.?\d+)\s+MG/ML\s+(?P<drug_type>.+)",
    "mg_actuat": r"(?P<drug_name>.+?)\s+(?P<drug_dose>\d*\.?\d+)\s+MG/ACTUAT\s+(?P<drug_type>.+)",
    "mg": r"(?P<drug_name>.+?)\s+(?P<drug_dose>\d*\.?\d+)\s+MG\s+(?P<drug_type>.+)"
    }

    # Apply ML MG/ML pattern
    mask_ml = df["standard_concept_name_drug"].str.contains(r"\d+\s+ML.*\d+\s+MG/ML", regex=True, na=False)
    df_ml = df.loc[mask_ml, "standard_concept_name_drug"].str.extract(patterns["ml_mgml"])
    df_ml["volume"] = pd.to_numeric(df_ml["volume"], errors='coerce')
    df_ml["concentration"] = pd.to_numeric(df_ml["concentration"], errors='coerce')
    df.loc[mask_ml, "drug_dose"] = df_ml["volume"] * df_ml["concentration"]
    df.loc[mask_ml, ["drug_name", "drug_type"]] = df_ml[["drug_name", "drug_type"]]

    # Apply MG/ACTUAT pattern
    mask_actuat = df["standard_concept_name_drug"].str.contains(r"\d*\.?\d+\s+MG/ACTUAT", regex=True, na=False)
    df_actuat = df.loc[mask_actuat, "standard_concept_name_drug"].str.extract(patterns["mg_actuat"])
    df.loc[mask_actuat, ["drug_name", "drug_dose", "drug_type"]] = df_actuat

    # Extract ACTUAT multiplier if present
    df["actuat_multiplier"] = pd.to_numeric(
        df.loc[mask_actuat, "standard_concept_name_drug"].str.extract(r"^(\d+)\s+ACTUAT")[0],
        errors='coerce'
    )
    # Drop invalid ACTUAT rows
    df = df[~(mask_actuat & df["actuat_multiplier"].isna())]
    mask_actuat = df["standard_concept_name_drug"].str.contains(r"\d*\.?\d+\s+MG/ACTUAT", regex=True, na=False)  # recompute

    # Adjust quantity
    df["adjusted_quantity"] = np.where(
        mask_actuat,
        df["quantity"] * df["actuat_multiplier"],
        df["quantity"]
    )

    # Apply MG pattern to remaining rows
    remaining_mask = ~(mask_ml | mask_actuat)
    df_remaining = df.loc[remaining_mask, "standard_concept_name_drug"].str.extract(patterns["mg"])
    df.loc[remaining_mask, ["drug_name", "drug_dose", "drug_type"]] = df_remaining

    # Convert drug_dose to numeric
    df["drug_dose"] = pd.to_numeric(df["drug_dose"], errors='coerce')
    
    # Create 'quantity_per_day' by dividing 'quantity' by 'days_supply', leaving NaN for missing or zero values
    df.loc[:, 'quantity_per_day'] = np.where(
        df[['adjusted_quantity', 'days_supply']].notna().all(axis=1),
        df['adjusted_quantity'] / df['days_supply'],
        np.nan
    )

    # Calculate 'dose_per_day' as 'drug_dose' * 'quantity_per_day', leaving NaN if any component is missing
    df.loc[:, 'dose_per_day'] = np.where(
        df[['drug_dose', 'quantity_per_day']].notna().all(axis=1),
        df['drug_dose'] * df['quantity_per_day'],
        np.nan
    )

    # Drop rows without parsed drug name
    df.dropna(subset=["drug_name"], inplace=True)

    # Replace inf with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Remove rows with any negative values in relevant numeric columns
    numeric_cols = ["quantity", "days_supply", "drug_dose", "quantity_per_day", "dose_per_day", "adjusted_quantity", "actuat_multiplier"]
    df = df[~(df[numeric_cols] < 0).any(axis=1)]

    return df
