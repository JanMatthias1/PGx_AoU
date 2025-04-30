
__author__ = "Jan Matthias"
__date__ = "15 Mar 2025"

"""
statistical_analysis_kKW: Kruskal-Wallis statistical testing 
"""

from import_functions import clean_pgx_data
from import_functions import perform_kruskal_test
from import_functions import find_metabolizing_enzyme
from import_functions import remove_outliers
from import_functions import calculate_adjusted_dosage
from import_functions import update_summary_statistics
from import_functions import store_feature_importance
from import_functions import store_statistical_results
from import_functions import process_person

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
random.seed(42)

seed = 42
np.random.seed(seed)


# ---------- IMPORTING DRUG DOSAGE DATA -----------
df = clean_pgx_data('')


# ---------- DROPPING NAN IN FEATURES THAT NEED TO BE PRESENT ------------
number_pca = 16
pca_columns = [f"pca_feature_{i}" for i in range(1, number_pca + 1)]

df["drug_exposure_end_datetime"] = pd.to_datetime(df["drug_exposure_end_datetime"], errors='coerce')

# these features are needed for all CYPs
df = df.dropna(subset=["person_id", "sex_at_birth", "BMI", "age_calculated_years", 
                       "dose_per_day", "drug_exposure_end_datetime"] + pca_columns)

#---------------- DATA FRAMES CREATION ------------

data_stat = pd.DataFrame(columns=["CYP_drug", "CYP_gene", "drug_name", "p_value", "stat_test", 
                                  "number_of_metabolizer", "number_people", 
                                  "statistical_test", "metabolizer_groups"])

summary_stats = pd.DataFrame(columns=["CYP_drug", "CYP_gene", "drug_name", "median_dose", "SD_dose", 
                                      "min_dose", "max_dose"])

data_to_export = pd.DataFrame()

# Define feature_importance DataFrame with both p-values and beta coefficients
feature_importance_columns = [
    "CYP_drug", "CYP_gene", "drug_name", "sex_at_birth_p", "sex_at_birth_beta", "BMI_p", "BMI_beta",
    "age_calculated_years_p", "age_calculated_years_beta"
]

for pca_col in pca_columns:
    feature_importance_columns.extend([f"{pca_col}_p", f"{pca_col}_beta"])

feature_importance = pd.DataFrame(columns=feature_importance_columns)


# ---------- DRUGS WITH THEIR RESPECTIVE CYP METABOLIZER -----------
drug_dictionary= {
    "CYP1A2": ["rucaparib", "olanzapine"],
    "CYP27A1": ["cholic acid"],
    "CYP2A6": ["letrozole"],
    "CYP2B6": ["prasugrel", "efavirenz", "tenofovir disoproxil", "ospemifene", "emtricitabine"],
    "CYP2C19": ["flibanserin", "nelfinavir", "dexlansoprazole", "lansoprazole", "brivaracetam",
                "axitinib", "clopidogrel", "rabeprazole", "amitriptyline", "lacosamide",
                "pantoprazole", "prasugrel", "carisoprodol", "mavacamten", "drospirenone",
                "esomeprazole", "voriconazole", "omeprazole", "diazepam", "citalopram",
                "formoterol", "clobazam", "phenytoin", "doxepin", "ticagrelor", "moclobemide",
                "belzutifan", "ethinyl estradiol", "abrocitinib", "atazanavir", "escitalopram"],
    "CYP2C9": ["dronabinol", "losartan", "acenocoumarol", "flibanserin", "meloxicam", "brivaracetam",
                "ospemifene", "rimegepant", "prasugrel", "etrasimod", "flurbiprofen", "glimepiride",
                "warfarin", "avatrombopag", "fosphenytoin", "celecoxib", "glyburide", "phenytoin",
                "nateglinide", "siponimod", "piroxicam", "lesinurad", "abrocitinib", "erdafitinib"],
    "CYP2D6": ["fesoterodine", "eletriptan", "amoxapine", "sertindole", "paliperidone", "modafinil",
                "fluvoxamine", "cariprazine", "dihydrocodeine", "donepezil", "dextromethorphan",
                "gefitinib", "protriptyline", "valbenazine", "eliglustat", "paroxetine", "fluoxetine",
                "palonosetron", "caffeine", "tiotropium", "brexpiprazole", "oliceridine",
                "propranolol", "darifenacin", "vortioxetine", "viloxazine", "nortriptyline",
                "trimipramine", "amphetamine", "iloperidone", "nebivolol", "upadacitinib",
                "arformoterol", "pimozide", "imipramine", "formoterol", "aripiprazole lauroxil",
                "pitolisant", "quinidine", "dronedarone", "opium",
                "desvenlafaxine", "mirabegron", "zuclopenthixol", "thioridazine", "escitalopram",
                "propafenone", "flibanserin", "primaquine", "duloxetine", "atomoxetine", "cevimeline",
                "tolterodine", "timolol", "tamoxifen", "ondansetron", "amitriptyline", "galantamine",
                "ritonavir", "tramadol", "metoclopramide", "carvedilol", "tamsulosin", "clozapine",
                "perphenazine", "codeine", "terbinafine", "citalopram", "paracetamol", "desipramine",
                "quinine", "aripiprazole", "tolperisone", "ibrutinib", "oxycodone", "vernakalant",
                "rucaparib", "metoprolol", "nefazodone", "deutetrabenazine", "ranolazine",
                "bupropion", "lofexidine", "trospium", "dapoxetine", "umeclidinium", "venlafaxine",
                "olanzapine", "doxepin", "clomipramine", "risperidone", "tetrabenazine",
                "xanomeline", "acetaminophen", "meclizine", "vilanterol"],
    "CYP3A4": ["zonisamide", "losartan", "cabazitaxel", "nelfinavir", "tolterodine",
                "posaconazole", "ritonavir", "sunitinib", "dolutegravir", "tipranavir",
                "sirolimus", "tamsulosin", "diazepam", "ivabradine", "lonafarnib",
                "dronedarone", "darunavir", "tacrolimus", "fosamprenavir", "telithromycin",
                "indinavir", "ruxolitinib"],
    "CYP3A5": ["prasugrel", "nelfinavir", "dolutegravir", "maraviroc", "tacrolimus"],
    "CYP7A1": ["cholic acid"]}

# Create a defaultdict where each drug will map to a set of enzymes
drug_to_enzymes = defaultdict(set)

# Populate the mapping
for enzyme, drugs in drug_dictionary.items():
    for drug in drugs:
        drug_to_enzymes[drug].add(enzyme)


# ----------------- STATISTICAL TESTING ----------------
for subset in ["all", "White", "Black or African American","Hispanic or Latino" ]:
    
    if subset == "Hispanic or Latino":
        df_race = df.dropna(subset=["ethnicity"]).copy()
        df_race = df_race[df_race["ethnicity"] == subset].copy()

    elif subset in ["White", "Black or African American"]:
        df_race = df.dropna(subset=["self_reported_category"]).copy()
        df_race = df_race[df_race["self_reported_category"] == subset].copy()

    elif subset == "all":
        df_race = df.copy()
      
    for j, CYP_gene in enumerate(["CYP2D6", "CYP2C19", "CYP2C9", "CYP2B6", "CYP3A5"]):
    
        print(CYP_gene, "CYP____________")
        
        # --- Filtering for CYP and dropping null values
        df_copy=df.copy()
        df_copy["metabolizer_type"] = df_copy[CYP_gene]
        df_copy = df_copy.dropna(subset=["metabolizer_type"])
        
        # Track tested drugs to avoid duplicate tests
        tested_drugs = set()
    
        for CYP_drug, drug_list in drug_dictionary.items():
            
            for drug_name in drug_list:
                
                print(f"Before {CYP_gene} with {drug_name} (CYP_drug: {CYP_drug})")
                
                if drug_name in tested_drugs:
                    print(f" ---------- Skipping {drug_name}, {CYP_drug} already tested -------------")
                    continue
                    
                # if the drug_name has already been tested, no need to duplicate
                tested_drugs.add(drug_name)
                
                CYP_drug_modified = find_metabolizing_enzyme(drug_name, CYP_gene, CYP_drug, drug_to_enzymes)
    
                print(f"Testing {CYP_gene} with {drug_name} (CYP_drug: {CYP_drug_modified})")
    
                # ---------------- DRUG extraction from data set -------------
                pattern = fr'\b{drug_name}\b'
    
                # Create a dataframe of patients on the drug
                df_first_drug = df_copy[df_copy['drug_name'].str.contains(pattern, case=False, na=False)].copy()
            
                # ------------- filter for at least 5 observations
                # Optimize filtering using .value_counts()
                valid_persons = df_first_drug['person_id'].value_counts()
                valid_persons = valid_persons[valid_persons >= 5].index
                df_filtered = df_first_drug[df_first_drug['person_id'].isin(valid_persons)]
    
                # Apply per person
                final_rows = df_filtered.groupby("person_id").apply(process_person).reset_index(drop=True)
    
                # Now compute mean dose per person from these cleaned 5 observations
                mean_dose_per_day_five = final_rows.groupby("person_id")["dose_per_day"].mean().round(10).reset_index()
    
                # Columns to merge
                merge_cols = ["person_id", "metabolizer_type", "sex_at_birth", "BMI", "age_calculated_years"] + pca_columns
        
                # Merge additional information     
                mean_dose_per_day = mean_dose_per_day_five.merge(df_first_drug[merge_cols].drop_duplicates(subset="person_id"),
                                                                 on="person_id", how="left")
    
                # 2) ---------- REMOVING OUTLIERS FROM THE OBSERVATIONS 
                mean_dose_per_day = mean_dose_per_day[mean_dose_per_day["dose_per_day"] != 0]
                # Drop duplicates again (ensure only last observation per person is kept)
                
                mean_dose_per_day= remove_outliers(mean_dose_per_day, column="dose_per_day", lower_quantile=0.25, 
                                                   upper_quantile=0.75, iqr_multiplier=1.5)
                
                mean_dose_per_day = mean_dose_per_day.sort_values("person_id").reset_index(drop=True)
                
                mean_dose_per_day[["drug_name", "gene_name", "drug_CYP"]] = drug_name, CYP_gene, CYP_drug_modified
                
                if CYP_gene == "CYP3A5" or CYP_gene == "CYP2C9":
                    mean_dose_per_day['metabolizer_group'] = mean_dose_per_day['metabolizer_type']
                
                else: 
                    # Group metabolizer types
                    mean_dose_per_day['metabolizer_group'] = mean_dose_per_day['metabolizer_type'].replace({
                        "Poor": "Poor/Intermediate",
                        "Intermediate": "Poor/Intermediate",
                        "Rapid": "Rapid/Ultrarapid",
                        "Ultrarapid": "Rapid/Ultrarapid"
                    })
                
                participants = len(mean_dose_per_day)
                
                # Convert the unique metabolizer groups list to a string before storing
                metabolizer_groups_str = ", ".join(map(str, mean_dose_per_day["metabolizer_group"].unique()))
                
                check_grouped_data = mean_dose_per_day.groupby("metabolizer_group")["dose_per_day"]
                has_variation = all(group.std() > 0 for _, group in check_grouped_data)
                
                if participants >= 50 and has_variation:
                    
                    # Compute summary statistics **after filtering**
                    if not mean_dose_per_day.empty:
                        summary_stats= update_summary_statistics(mean_dose_per_day, CYP_drug_modified, CYP_gene, drug_name, summary_stats)
            
                    # ------- Covariate adjusted Drug Dosage 
                    mean_dose_per_day, model = calculate_adjusted_dosage(mean_dose_per_day, pca_columns) 
    
                    # Store p-values and beta coefficients
                    feature_importance = store_feature_importance(model, CYP_drug_modified, CYP_gene, drug_name, pca_columns, 
                                                                  feature_importance)
                
                    data_to_export = pd.concat([data_to_export, mean_dose_per_day], ignore_index=True)
                    
                    # Count unique metabolizer groups
                    unique_groups = mean_dose_per_day["metabolizer_group"].nunique()
                    
                    if unique_groups == 3:
                        # Perform Kruskal-Wallis test for three groups
                        p_value, statistic, test_type = perform_kruskal_test(mean_dose_per_day.groupby("metabolizer_group"), 3)
    
                    else:
                        # If there are not exactly 2 or 3 groups, skip the test
                        p_value = "Skipped"
                        statistic = "Skipped"
                        test_type = "Skipped"
                    
    
                    # Append to data frame
                    data_stat.loc[len(data_stat)] = [CYP_drug_modified, CYP_gene, drug_name, p_value, statistic,
                                                     len(mean_dose_per_day["metabolizer_group"].unique()), 
                                                     participants, test_type,
                                                     metabolizer_groups_str]
                
                else:
                    p_value = "Skipped"  # Not enough data for the test
                    statistic = "Skipped"
                    
                    number_of_metabolizer= "Skipped"
                    data_stat.loc[len(data_stat)] = [CYP_drug_modified, CYP_gene, drug_name, p_value, statistic, 
                                                        number_of_metabolizer, participants, "Skipped",
                                                        metabolizer_groups_str]
                

 #----------------- SIGNIFICANT P-VALUES ----------------
data_stat["p_value"] = pd.to_numeric(data_stat["p_value"], errors="coerce")
# Add 'significant' column, leaving it blank when p_value >= 0.05
data_stat["significant"] = np.where(
    (data_stat["CYP_drug"] == data_stat["CYP_gene"]) & (data_stat["p_value"] < 0.05),
    "yes",  "")
