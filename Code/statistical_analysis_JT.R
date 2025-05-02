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
  CYP1A2 = c("olanzapine", "rucaparib"),
  CYP27A1 = c("cholic acid"),
  CYP2A6 = c("letrozole"),
  CYP2B6 = c("prasugrel", "ospemifene", "efavirenz"),
  CYP2C19 = c(
    "citalopram", "belzutifan", "pantoprazole", "voriconazole", "diazepam",
    "axitinib", "amitriptyline", "ticagrelor", "lacosamide", "nelfinavir",
    "prasugrel", "phenytoin", "atazanavir", "clobazam", "doxepin",
    "esomeprazole", "brivaracetam", "flibanserin", "moclobemide", "abrocitinib",
    "dexlansoprazole", "lansoprazole", "mavacamten", "rabeprazole",
    "carisoprodol", "omeprazole", "formoterol", "escitalopram", "clopidogrel"
  ),
  CYP2C9 = c(
    "fosphenytoin", "siponimod", "dronabinol", "losartan", "meloxicam",
    "avatrombopag", "prasugrel", "phenytoin", "brivaracetam", "flurbiprofen",
    "lesinurad", "erdafitinib", "ospemifene", "nateglinide", "etrasimod",
    "flibanserin", "celecoxib", "piroxicam", "glimepiride", "glyburide",
    "abrocitinib", "rimegepant", "acenocoumarol", "warfarin"
  ),
  CYP2D6 = c(
    "ondansetron", "dextromethorphan", "quinine",
    "eliglustat", "nefazodone", "gefitinib", "terbinafine", "rucaparib",
    "oxycodone", "metoclopramide", "vernakalant", "desvenlafaxine", "fluoxetine",
    "tolperisone", "flibanserin", "amphetamine", "thioridazine", "paliperidone",
    "oliceridine", "cevimeline", "deutetrabenazine", "formoterol", "escitalopram",
    "mirabegron", "tamoxifen", "duloxetine", "risperidone", "modafinil",
    "donepezil", "aripiprazole lauroxil", "clomipramine", "umeclidinium",
    "pitolisant", "dapoxetine", "ritonavir", "atomoxetine", "sertindole",
    "timolol", "codeine", "upadacitinib", "vortioxetine", "ibrutinib",
    "desipramine", "tiotropium", "eletriptan", "olanzapine", "carvedilol",
    "protriptyline", "cariprazine", "citalopram", "meclizine", "nebivolol",
    "clozapine", "valbenazine", "propafenone", "amoxapine", "primaquine",
    "fesoterodine", "paroxetine", "iloperidone", "brexpiprazole", "palonosetron",
    "metoprolol", "tamsulosin", "fluvoxamine", "propranolol", "aripiprazole",
    "arformoterol", "tolterodine", "dronedarone", "nortriptyline", "galantamine",
    "tetrabenazine", "zuclopenthixol", "doxepin", "tramadol", "bupropion",
    "venlafaxine", "trimipramine", "viloxazine", "perphenazine", "lofexidine",
    "darifenacin", "imipramine", "pimozide", "ranolazine", "quinidine",
    "amitriptyline"
  ),
  CYP3A4 = c(
    "dronedarone", "darunavir", "fosamprenavir", "diazepam", "losartan",
    "zonisamide", "sunitinib", "nelfinavir", "ruxolitinib", "dolutegravir",
    "cabazitaxel", "lonafarnib", "ritonavir", "sirolimus", "indinavir",
    "posaconazole", "tamsulosin", "tacrolimus", "tipranavir", "tolterodine",
    "telithromycin", "ivabradine"
  ),
  CYP3A5 = c("nelfinavir", "prasugrel", "maraviroc", "dolutegravir", "tacrolimus"),
  CYP7A1 = c("cholic acid")
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


