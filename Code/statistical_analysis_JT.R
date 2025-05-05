__author__ = "Jan Matthias"
__date__ = "30 April 2025"

"""
statistical_analysis_JT: Jonckheere-Terpstra statistical testing 
"""
library(readr)
library(dplyr)
library(ggplot2)
library(clinfun)

set.seed(42)  # Set seed for reproducibility

# ---------------- DRUG DICTIONARY ------------
drug_dictionary <- list(
  CYP2B6 = c("sertraline", "efavirenz", "methadone", "bupropion", "nevirapine"),
  CYP2C19 = c(
    "rabeprazole", "trimipramine", "clobazam", "clomipramine", "esomeprazole",
    "lansoprazole", "sertraline", "venlafaxine", "duloxetine", "fluvoxamine",
    "escitalopram", "brivaracetam", "amitriptyline", "imipramine", "omeprazole",
    "dexlansoprazole", "carisoprodol", "citalopram", "doxepin", "flibanserin",
    "fluoxetine", "paroxetine", "clopidogrel", "diazepam", "vortioxetine",
    "pantoprazole", "voriconazole"
  ),
  CYP2C8 = c("rosiglitazone", "ibuprofen", "diclofenac"),
  CYP2C9 = c(
    "fosphenytoin", "lornoxicam", "ibuprofen", "aspirin", "celecoxib", "phenytoin",
    "flurbiprofen", "warfarin", "meloxicam", "erdafitinib", "piroxicam",
    "fluvastatin", "lesinurad", "naproxen", "aceclofenac", "siponimod",
    "nabumetone", "dronabinol", "tenoxicam", "lumiracoxib", "flibanserin",
    "diclofenac", "avatrombopag", "acenocoumarol", "indomethacin"
  ),
  CYP2D6 = c(
    "lofexidine", "palonosetron", "atomoxetine", "ondansetron", "oliceridine",
    "quinidine", "flecainide", "dolasetron", "fluvoxamine", "aripiprazole",
    "escitalopram", "nebivolol", "galantamine", "amitriptyline", "darifenacin",
    "tolterodine", "nortriptyline", "eliglustat", "iloperidone", "zuclopenthixol",
    "propranolol", "fesoterodine", "paroxetine", "flibanserin", "pimozide",
    "tamsulosin", "amphetamine", "oxycodone", "codeine", "metoprolol",
    "propafenone", "citalopram", "mirabegron", "timolol", "dextromethorphan",
    "hydrocodone", "terbinafine", "clomipramine", "valbenazine", "methadone",
    "mirtazapine", "methylphenidate", "carvedilol", "desipramine", "tropisetron",
    "amoxapine", "aripiprazole lauroxil", "metoclopramide", "thioridazine",
    "bisoprolol", "donepezil", "imipramine", "tramadol", "protriptyline",
    "doxepin", "betaxolol", "pitolisant", "acebutolol", "vortioxetine",
    "perphenazine", "trimipramine", "tamoxifen", "modafinil", "sertraline",
    "venlafaxine", "haloperidol", "gefitinib", "duloxetine", "clozapine",
    "quinine", "tetrabenazine", "risperidone", "meclizine", "deutetrabenazine",
    "fluoxetine", "brexpiprazole", "labetalol", "cevimeline"
  ),
  CYP3A4 = c(
    "simvastatin", "tacrolimus", "atorvastatin", "fluvastatin",
    "lovastatin", "rosuvastatin", "pravastatin", "pitavastatin"
  ),
  CYP3A5 = c(
    "simvastatin", "tacrolimus", "sirolimus", "atorvastatin",
    "fluvastatin", "lovastatin", "rosuvastatin", "midazolam",
    "cyclosporine", "pravastatin", "pitavastatin"
  ),
  CYP4F2 = c("warfarin", "acenocoumarol", "phenprocoumon")
)

# Populate the mapping
drug_to_enzymes <- list()
for (enzyme in names(drug_dictionary)) {
  for (drug in drug_dictionary[[enzyme]]) {
    drug_to_enzymes[[drug]] <- c(drug_to_enzymes[[drug]], enzyme)
  }}


# Initialize a results data frame
results <- data.frame(
  CYP_drug = character(),  # Enzyme that metabolizes the drug 
  CYP_gene = character(),  # Gene of the phenotype
  drug_name = character(),
  metabolizer_grouping = character(),  # "3-group" or "5-group"
  p_value = numeric(),
  stat_test = numeric(),
  number_of_metabolizer = integer(),
  participants = integer(),
  subset=character(),
  stringsAsFactors = FALSE)


# --------------- JT STATISTICAL TESTING ------------------
for (subset_loop in c("all","White", "Black or African American", "Hispanic or Latino")){
  
  df_race <- df %>% filter(subset == subset_loop)

  # Loop over the CYP genes and drugs
  for (CYP_gene in c("CYP2D6","CYP2C19", "CYP2C9", "CYP2B6", "CYP3A5")) {
    
    # Filter for the current CYP gene
    df_filtered <- df_race %>% filter(gene_name == CYP_gene)
    
    tested_drugs <- c() 
    
    for (CYP_drug in names(drug_dictionary)) {
      for (drug in drug_dictionary[[CYP_drug]]) {
        
        #name change to CYP_Drug
        if (drug %in% tested_drugs) {
          cat(sprintf(" ---------- Skipping %s, %s already tested -------------\n", drug, CYP_drug))
          next
        }
        
        # If the drug has already been tested, no need to duplicate
        tested_drugs <- unique(c(tested_drugs, drug))
        
        metabolizing_enzymes <- drug_to_enzymes[[drug]]
        
        if (CYP_gene %in% metabolizing_enzymes) {
          CYP_drug_modified <-  CYP_gene
        } else if (length(metabolizing_enzymes) > 1 && !(CYP_gene %in% metabolizing_enzymes)) {
          CYP_drug_modified <- paste(sort(unique(metabolizing_enzymes)), collapse = "/")
        } else {
          CYP_drug_modified <- CYP_drug
        }
        
        cat(sprintf("Testing %s with %s (CYP_drug: %s)\n", CYP_gene, drug, CYP_drug_modified))
        
        # Filter the drug 
        df_drug <- df_filtered %>% filter(drug_name == drug)

        # Check if the drug is missing from the dataset
        if (nrow(df_drug) == 0) {
          cat("Skipping drug:", drug, "- not found in dataset\n")
          
          results <- rbind(
            results,
            data.frame(
              CYP_drug = CYP_drug_modified,
              CYP_gene = CYP_gene,  
              drug_name = drug,
              metabolizer_grouping = "Skipped",
              p_value = "Skipped",
              stat_test = "Skipped",
              number_of_metabolizer = "Skipped",
              participants = "Skipped",
              subset= subset_loop,
              stringsAsFactors = FALSE
            )
          )
          
          next  # Skip to the next drug in the loop
        }
        ### ----------- 3-GROUP TEST (Poor/Intermediate, Normal, Rapid/Ultrarapid) ----------- ###
        
        if (CYP_gene %in% c("CYP2C9", "CYP3A5")) {
          metabolizer_order_3 <- c("Poor", "Intermediate", "Normal")
        } else {
          metabolizer_order_3 <- c("Poor/Intermediate", "Normal", "Rapid/Ultrarapid")
        }
        
        df_drug$metabolizer_group <- factor(df_drug$metabolizer_group, levels = metabolizer_order_3, ordered = TRUE)
        
        grouped_3 <- df_drug %>% 
          group_by(metabolizer_group) %>% 
          summarise(count = n(), dose_values = list(Adjusted_Dosage), .groups = "drop")
        
        # Get distinct count of metabolizers
        number_of_metabolizer_3 <- n_distinct(df_drug$metabolizer_group)
        
        groups_3 <- grouped_3$dose_values
        
        if (length(groups_3) == 3 && all(grouped_3$count >= 10)) {
          test_result_3 <- tryCatch(
            {
              set.seed(123)  
              jonckheere.test(df_drug$Adjusted_Dosage, df_drug$metabolizer_group, nperm = 1000)
            }, 
            error = function(e) {
              cat("Error in Jonckheere test for drug:", drug, "- skipping\n")
              return(NULL)
            }
          )
          
          p_value_3 <- ifelse(is.null(test_result_3), NA, round(test_result_3$p.value, 5))
          statistic_3 <- ifelse(is.null(test_result_3), NA, round(test_result_3$statistic, 5))
          
        } else {
          cat("Skipping drug:", drug, "- does not meet the 3-group count requirements\n")
          p_value_3 <- "Skipped"
          statistic_3 <- "Skipped"
        }
        
        results <- rbind(
          results,
          data.frame(
            CYP_drug = CYP_drug_modified,
            CYP_gene = CYP_gene,  
            drug_name = drug,
            metabolizer_grouping = "3-group",
            p_value = p_value_3,
            stat_test = statistic_3,
            number_of_metabolizer = number_of_metabolizer_3,
            participants = nrow(df_drug),
            subset= subset_loop,
            stringsAsFactors = FALSE
          )
        )
      }
    }
  }}

