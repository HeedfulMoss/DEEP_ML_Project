import csv

input_csv = "S:\\Downloads\\Backup_DEEP_ML_Project_stuff\\pre-processing_files_1\\NOTEEVENTS.csv"     # Replace with your actual path
output_csv = "S:\\Downloads\\Backup_DEEP_ML_Project_stuff\\pre-processing_files_1\\NOTEEVENTS_trimmed.csv"
max_rows = 500

with open(input_csv, mode='r', newline='', encoding='utf-8') as infile, \
     open(output_csv, mode='w', newline='', encoding='utf-8') as outfile:
    
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    for i, row in enumerate(reader):
        writer.writerow(row)
        if i == max_rows:
            break

print(f"✅ Saved first {max_rows} rows to '{output_csv}'")
