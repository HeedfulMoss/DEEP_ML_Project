import pandas as pd

# Load the dataset
df = pd.read_csv(r"C:\Users\Alex\Documents\GitHub\DEEP_ML_Project\data\preprocessed\summary_results.csv")

# Select only the ICD code columns (excluding metadata)
icd_columns = df.columns.difference(['SUBJECT_ID', 'HADM_ID', 'summary_snippet'])

# Find rows where all ICD codes are 0
no_icd_rows = df[icd_columns].sum(axis=1) == 0

# Display those rows (if any)
rows_without_icd = df[no_icd_rows]
print(f"Number of rows without any ICD codes: {len(rows_without_icd)}")
print(rows_without_icd)