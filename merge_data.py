import pandas as pd
import os
data_folder="raw_data"
print()
all_files=os.listdir(data_folder)
for file in all_files:
    print(file)
excel_files=[]
for file_name in all_files:
    if file_name.endswith(".xlsx") or file_name.endswith(".xls") :
        excel_files.append(file_name)
print(excel_files)
all_data=[]
for excel_file in excel_files:
    file_path=os.path.join(data_folder,excel_file)
    print(excel_file)
    df = pd.read_excel(file_path)
    all_data.append(df)
combined_data=pd.concat(all_data)
print('merged successfully')
print(len(combined_data))
print(len(combined_data.columns))
output_file="merged.csv"
combined_data.to_csv(output_file,index=False,encoding='utf-8')
print(output_file)