rm(list = ls())
library(readr)
library(plyr)
library(readxl)
library(stringr)
library(magrittr)
setwd('/Users/amandabuch/Documents/clusterCCA/revision1/clusterCCA/data/ScienceDirect_files_15Jan2022_22-44-41.490/CVDSBA-master')
source("datamining_library_ge20200306.R")



## Clinical and blood count information
subject_shared_info = read_xlsx("data/1-s2.0-S0092867420306279-mmc1.xlsx",col_names = T, sheet='Clinical_information', col_types='text') # Contains patient IDS for metabolomcis and proteomoics datasets
clinical_cols = c("Patient ID a", "MS ID b", "Metabolomics ID e","Sex g","Age (year)", "BMI h", "Test date...18", "WBC count, ×109/L", "Lymphocyte count, ×109/L", "Monocyte count, ×109/L",
"Platelet count, ×109/L" , "CRP i, mg/L", "ALT j,  U/L", "AST k, U/L","GGT l, U/L", "TBIL m, μmol/L",
"DBIL n, μmol/L", "Creatinine, μmol/L", "Glucose, mmol/L")
clinical = subject_shared_info[,clinical_cols]

subject_shared_info = subject_shared_info[,names(subject_shared_info[c(1,3,8)])]
names(subject_shared_info)=c('Patient_ID','Proteomics_ID','Metabolomics_ID')
subject_shared_info_overlap = subject_shared_info[subject_shared_info$Proteomics_ID!='/' & subject_shared_info$Metabolomics_ID!='/',]

## Metabolomics
df <- read_xls("data/Taizhou20200328.xls",col_names = T) # Contains metabolomics data


# Extract subject information
df1 <- df[-c(1:2),-1] %>% as.data.frame()
row.names(df1) <- df$...1[-c(1:2)] # Names of molecules
names(df1) <- df[1,-1] # Contains labels for severe COVID and nonsevere COVID
label <- gsub("\\d+","",names(df1)) # removes subject ID
# Separate controls from patients
zx_type <- which(label=="ZX") # COVID severe
pt_type <- which(label=="PT") # COVID not severe
jbdz_type <- which(label=="jbdz") # Noncovid
jkbz_type <- which(label=="jkdz") # Healthy

# Convert to log2
df2 <- apply(df1,1, as.numeric) %>% t() %>% as.data.frame() %>% log2() #
names(df2) <- names(df1)

# Remove 36 drugs detected
detect <- read_xlsx("data/Detected Drugs information from Calibra.xlsx",col_names = F)
df2 <- df2[-which(row.names(df2) %in% detect$...1),]

# remove NA
not_na = data.frame(apply(df2, 1, function(x) (sum(is.na(x)) == 0  )  ))[,1]
subjects_keep_not_na = which(label=="ZX" | label=="PT" | label=="jkdz" | label=="jbdz")
metabolomics = t(df2[not_na,subjects_keep_not_na])
label = label[subjects_keep_not_na]

subjects_shared = rownames(metabolomics) %in% subject_shared_info_overlap$Metabolomics_ID
metabolomics = metabolomics[subjects_shared,]
label = label[subjects_shared]

subjects_shared = clinical$`Metabolomics ID e` %in% rownames(metabolomics)
clinical = clinical[subjects_shared,]

label = data.frame(label)

write.csv(metabolomics,file = "metabolomics_data_preprocessed.csv", quote = TRUE, row.names = TRUE, col.names = TRUE)
write.csv(clinical,file = "clinical_preprocessed.csv", quote = TRUE, row.names = TRUE, col.names = TRUE)
write.csv(label[,1],file = "metabolomics_labels.csv", quote = FALSE, row.names = FALSE, col.names = FALSE)


## Proteomics
rm(list = ls())
library(readr)
library(plyr)
library(readxl)
library(stringr)
library(magrittr)
setwd('/Users/amandabuch/Documents/clusterCCA/revision1/clusterCCA/data/ScienceDirect_files_15Jan2022_22-44-41.490/CVDSBA-master')
source("datamining_library_ge20200306.R")
df <- read_xlsx("data/COVID19_123456.xlsx",sheet = 2)
df1 <- df[,grepl("Ratio:",names(df))] %>% as.data.frame()
row.names(df1) <- df$Accession
nm1 <- ge.split(names(df1),"\\)",1)
nm2 <- ge.split(nm1,"\\(",2)
names(df1) <- gsub(", ","_",nm2)

info <- read_xlsx("data/sampleinfo2.xlsx")
info$TMT <- gsub("^b","F",info$TMT)

## Clinical and blood count information
clinical = read_csv("clinical_preprocessed.csv", col_names = TRUE, col_types='c') # Contains patient IDS for metabolomcis and proteomoics datasets
clinical$type = info$Type[info$TMT %in% clinical$`MS ID b`] # Contains patient IDS for metabolomcis and proteomoics datasets

# # Convert to log2
# df2 <- apply(df1,1, as.numeric) %>% t() %>% as.data.frame() %>% log2() #
# names(df2) <- names(df1)

# Remove NA
df2 <- df1[apply(df1,1, function(x){sum(!is.na(x))})>0,]

# Get batch and subject information
batch = data.frame(ge.split(names(df2),"_",1))
label = data.frame(info$Type[match(names(df2),info$TMT)])

# Separate controls from patients
zx_type <- which(label=="ZX")
pt_type <- which(label=="PT")
jkdz_type <- which(label=="jkdz")
jbdz_type <- which(label=="jbdz")

not_na = data.frame(apply(df2, 1, function(x) (sum(is.na(x)) == 0  )  ))[,1]
subjects_keep = which(label=="ZX" | label=="PT" | label=="jkdz" | label=="jbdz")
proteomics = t(df2[not_na,subjects_keep])
label = label[subjects_keep,]
batch = batch[subjects_keep,]

subjects_shared = rownames(proteomics) %in% clinical$`MS ID b`
proteomics = proteomics[subjects_shared,]
label = data.frame(label[subjects_shared])
batch = data.frame(batch[subjects_shared])

subjects_shared = clinical$`MS ID b` %in% rownames(proteomics)
clinical_shared = clinical[subjects_shared,]

## Subset metabolomics
metabolomics = read_csv("metabolomics_data_preprocessed.csv", col_names = TRUE, col_types='c') # Contains patient IDS for metabolomcis and proteomoics datasets
metabolomics_labels = read_csv("metabolomics_labels.csv", col_names = TRUE, col_types='c') # Contains patient IDS for metabolomcis and proteomoics datasets

subjects_shared = data.frame(metabolomics[,1])[,1] %in% clinical_shared$`Metabolomics ID e`
metabolomics_shared = metabolomics[subjects_shared,]
metabolomics_labels_shared = data.frame(metabolomics_labels[subjects_shared,])


metabolomics_shared_sort = merge(data.frame(clinical_shared)[,3:4],data.frame(metabolomics_shared), by.x=2,  by.y=1, sort=FALSE)
names(metabolomics_shared_sort) = names(metabolomics)
metabolomics_labels_shared_sort = data.frame(gsub("\\d+","",metabolomics_shared_sort$Metabolomics.ID.e) )

proteomics_shared_sort = merge(data.frame(clinical_shared)[,3:4],data.frame(proteomics), by.x=1,  by.y=0, sort=FALSE)
proteomics_batches_sort = data.frame(ge.split(proteomics_shared_sort$MS.ID.b,"_",1))
proteomics_labels_shared_sort = data.frame(clinical_shared$type)


# write.csv(clinical_shared,file = "clinical_shared.csv", quote = TRUE, row.names = TRUE, col.names = TRUE)
write.csv(metabolomics_shared_sort,file = "metabolomics_data_preprocessed_shared.csv", quote = TRUE, row.names = TRUE, col.names = TRUE)
data.table:::fwrite(metabolomics_shared_sort,file = "metabolomics_data_preprocessed_shared.csv", quote = TRUE, row.names = TRUE, col.names = TRUE)
# write.csv(metabolomics_labels_shared_sort,file = "metabolomics_labels_shared.csv", quote = FALSE, row.names = FALSE, col.names = FALSE)
# write.csv(proteomics_shared_sort,file = "proteomics_data_preprocessed_shared.csv", quote = TRUE, row.names = TRUE, col.names = TRUE)
# write.csv(proteomics_labels_shared_sort,file = "proteomics_labels_shared.csv", quote = FALSE, row.names = FALSE, col.names = FALSE)
# write.csv(proteomics_batches_sort,file = "proteomics_batch_shared.csv", quote = FALSE, row.names = FALSE, col.names = FALSE)






str_replace(str_replace(names(metabolomics),'  ','__'),':','0')