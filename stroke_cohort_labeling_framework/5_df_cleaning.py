""" Step 5: (Final step) This file is used for multiple purposes:
1. Step1 + Step 2: Cleaning the merged scans/msx/icd data and deduplicating any unnecessary rows
    a. If a record or a visit have flowsheet value rows and empty flowsheet value rows. 
And if empty flowsheet value rows don’t add any information we should remove them
    b. Deduplicate the OP (ED) and IP - merging rows: for one patient we have multiple 
rows because some patients come to the hospital through the ED then get admitted.
2. 
"""
import pandas as pd


df = pd.read_excel('output_data/mergedICDScanCaboodle-02012024_modified.xlsx', sheet_name='mergedICDScanCaboodle-02012024')

# Step 0: Keep only stroke CTs if the stroke date and scan date are the same
df = df[(df['ExamCode'] != 'CTNHEADAS0') | (
    (df['ExamCode'] == 'CTNHEADAS0') & (df['FormattedStrokeCodeDate'] == pd.to_datetime(df['ScanDate']).dt.date))]

# Step 1: deduplicate Flowsheets vs scans.
df.sort_values(by=['MRN', 'ACC', 'FinalDate', 'UniqueActivatedStrokeCode'], na_position='last', inplace=True)
columns_to_consider = df.columns.difference([
    'FinalDate',
    'StrokeCodeDateTime', 
    'StrokeCodeTime', 
    'StrokeCodeDate', 
    'FormattedStrokeCodeDate', 
    'UniqueActivatedStrokeCode',
    'TimeOfDayKey',
    'FlowsheetRowKey',	
    'FlowsheetTemplateName',
    'StrokeCode_Provider',
])

df['KeepRow'] = ~df.duplicated(subset=columns_to_consider, keep='first')
# print(df[['UniqueActivatedStrokeCode','ACC','StrokeCode_Provider', 'ImagingKey', 'KeepRow']])
df_filtered = df[df['KeepRow']]
# print(df_filtered)

# Step 2: merge IP and OP (ED) rows together.
ip_op_columns_to_consider = ['unique id']

def first_non_null(series):
    return series.dropna().iloc[0] if not series.dropna().empty else None


def concatenate_non_null(series):
    return ', '.join(series.dropna().astype(str).unique())

columns_to_concatenate = ['DIAG_TYPE', 'SEC_DIAG_CD', 'pat_type', 'ENCOUNTER_NO']
aggregation_functions = {col: first_non_null for col in df.columns}
for col in columns_to_concatenate:
    aggregation_functions[col] = concatenate_non_null

df_filtered = df_filtered.groupby(list(ip_op_columns_to_consider)).agg(aggregation_functions)
df_filtered.reset_index(drop=True, inplace=True)

df_filtered.to_csv('output_data/DfIpOpGrouped.csv')


def is_ich_code(sec_diag_cd):
    split_ich_icd = sec_diag_cd.split(',')
    return any(icd.startswith('I61') or icd.startswith(' I61') or icd.startswith('431') or icd.startswith(' 431') for icd in split_ich_icd)


# Step 3:identify unique activated stroke codes
def assign_activated_stroke_code_num(group):
    group = group.sort_values('CompletedDTTM').reset_index(drop=True)
    stroke_code_num = 1
    last_non_null_flowsheet_value = None
    first_appearance_dict = {}

    for index, row in group.iterrows():
        current_code = row['UniqueActivatedStrokeCode']
        exam_code = row['ExamCode']
        sec_diag_cd = row['SEC_DIAG_CD']

        if index == 0:
            # For the first row in each group, assign stroke_code_num = 1
            # Update the dictionary if current_code is not null
            if pd.notna(current_code):
                first_appearance_dict[current_code] = stroke_code_num
        else:
            # Apply incrementing rules for subsequent rows
            if pd.isna(current_code) and exam_code == 'CTNHEADAS0':
                stroke_code_num += 1

            elif pd.isna(last_non_null_flowsheet_value) and pd.notna(current_code):
                if exam_code.startswith('CT') and not is_ich_code(sec_diag_cd):
                    if current_code in first_appearance_dict:
                        stroke_code_num = first_appearance_dict[current_code]
                    else:
                        # If not, increment stroke_code_num and add it to the dictionary
                        stroke_code_num += 1
                        first_appearance_dict[current_code] = stroke_code_num
            
            elif pd.notna(current_code):
                if current_code != last_non_null_flowsheet_value:
                    # Check if current_code exists in first_appearance_dict
                    if current_code in first_appearance_dict:
                        stroke_code_num = first_appearance_dict[current_code]
                    else:
                        # If not, increment stroke_code_num and add it to the dictionary
                        stroke_code_num += 1
                        first_appearance_dict[current_code] = stroke_code_num
                else:
                    stroke_code_num = first_appearance_dict.get(current_code, stroke_code_num)

        # Update last non-null flowsheet value if current code is not null
        # if pd.notna(current_code):
        last_non_null_flowsheet_value = current_code

        # Assign the stroke code number to the current row
        group.at[index, 'ActivatedStrokeCodeNum'] = stroke_code_num

    return group


df_filtered = df_filtered.groupby('EncounterKey').apply(assign_activated_stroke_code_num)
df_filtered.reset_index(drop=True, inplace=True)

stroke_code_date = df_filtered.groupby(['EncounterKey', 'ActivatedStrokeCodeNum'])['FinalDate'].min().reset_index()
stroke_code_date.rename(columns={'FinalDate': 'StrokeCodeDate'}, inplace=True)

# For each encounter key and activated stroke code, if any of the labels were True then it is a true stroke, otherwise false.
true_stroke_code = df_filtered.groupby(['EncounterKey', 'ActivatedStrokeCodeNum'])['FinalLabel'].any().reset_index()
true_stroke_code.rename(columns={'FinalLabel': 'TrueStrokeCodeActivated'}, inplace=True)

# print(stroke_code_date)
# For each encounter key and activated stroke code, stroke date and time is the minimum of the Final Date.
df_final = pd.merge(df_filtered, stroke_code_date, on=['EncounterKey', 'ActivatedStrokeCodeNum'], how='left')
df_final = pd.merge(df_final, true_stroke_code, on=['EncounterKey', 'ActivatedStrokeCodeNum'], how='left')

# df_final.to_csv('df_final.csv')
# print(df_final)
df_final.to_csv('output_data/CleanedMergedIcdScanCaboodle02012024.csv')
