"""Step 4: Use this to merge ICD codes (MSX) data with imaging data (RIS) then merge on Caboodle Date.
1. Merge ICD on Scans: Using completed date and visit date for OP (ED) and completed date and admit/discharge for IP.
2. Determine what is a visit using the 48hr rule. We use this to determine which ICD codes to keep and which 
to discard. This can be changed to reflect the actual hospitalization which can be determined based on the 
Admit date and Discharge date. But if we change this we need to be careful with the OP (ED) visits
3. Keeping and Discarding ICD codes are done based on specific rules (in process_visit function).
4. We join caboodle data on the merged and filtered MSX/RIS data using MRN and Scan Accession number.
"""
import pandas as pd


def process_visit(group):    
    positive_scans = group[group['NLPw48hrs']]
    negative_scans = group[~group['NLPw48hrs']]
    positive_icd = group['ICDStrokeLabel'].any()
    negative_icd = not group['ICDStrokeLabel'].all()

    group['keep'] = False

    # Scenario 1: Mix of Positive and Negative Scans, Mix of ICD codes
    if not positive_scans.empty and not negative_scans.empty:
        if positive_icd and negative_icd:
            group.loc[group['NLPw48hrs'], 'keep'] = group['ICDStrokeLabel']
            group.loc[~group['NLPw48hrs'], 'keep'] = ~group['ICDStrokeLabel']
        else:
            group['keep'] = True
    # Scenario 2: Only Positive Scans
    elif not positive_scans.empty:
        if positive_icd and negative_icd:
            # Keep only Positive ICD codes
            group['keep'] = group['ICDStrokeLabel']
        else:
            # If only Negative ICD codes, keep all
            group['keep'] = True

    # Scenario 3: Only Negative Scans
    elif not negative_scans.empty:
        if positive_icd and negative_icd:
            # When scan is negative but mixed ICD codes, keep only Positive ICD codes
            group['keep'] = group['ICDStrokeLabel']
        elif negative_icd:
            # Keep all if only Negative ICD codes
            group['keep'] = True
        else:
            # If only Positive ICD codes, keep them
            group['keep'] = True

    return group


def min_date(row):
    if pd.isna(row['OrderedDTTM']):
        return row['CompletedDTTM']
    if pd.isna(row['CompletedDTTM']):
        return row['OrderedDTTM']
    return min(row['OrderedDTTM'], row['CompletedDTTM'])


if __name__ == '__main__':    
    scans = pd.read_csv("output_data/AllNeuroAccessionsLabeledV22W48hrs.csv",
                        dtype={'MRN': str})
    scans = scans.dropna(subset=['ACC'])
    scans['ACC'] = scans['ACC'].astype(int).astype(str)
    # print(scans['ACC'].dtype)
    icd_codes = pd.read_csv("output_data/aggregated_MSX_ICD_data_12212023.csv", 
                            dtype={'MSMRN': str, 'MRN_MSX': str, 'ICD_VERSION':str})
    
    merged_data = pd.merge(scans, icd_codes, left_on='MRN', right_on='MSMRN', how='left')
    
    merged_data['COMPLETED_DATE'] = pd.to_datetime(merged_data['CompletedDTTM'], format='%Y-%m-%d %H:%M:%S').dt.date
    merged_data['DSCH_DT_SRC'] = pd.to_datetime(merged_data['DSCH_DT_SRC'], format='%m/%d/%y').dt.date
    merged_data['ADMIT_DT_SRC'] = pd.to_datetime(merged_data['ADMIT_DT_SRC'], format='%m/%d/%y').dt.date
    merged_data['VISIT_DT_SRC'] = pd.to_datetime(merged_data['VISIT_DT_SRC'], format='%m/%d/%y').dt.date
     
    # Filtering conditions
    ip_condition = (
        (merged_data['pat_type'] == 'IP') &
        (
            ((merged_data['ADMIT_DT_SRC'] <= merged_data['COMPLETED_DATE']) &
             (merged_data['COMPLETED_DATE'] <= merged_data['DSCH_DT_SRC'])) |
            ((merged_data['ADMIT_DT_SRC'] <= merged_data['COMPLETED_DATE'] - pd.Timedelta(days=1)) &
             (merged_data['COMPLETED_DATE'] - pd.Timedelta(days=1) <= merged_data['DSCH_DT_SRC'])) |
            ((merged_data['ADMIT_DT_SRC'] <= merged_data['COMPLETED_DATE'] - pd.Timedelta(days=2)) &
             (merged_data['COMPLETED_DATE'] - pd.Timedelta(days=2) <= merged_data['DSCH_DT_SRC'])) |
            ((merged_data['ADMIT_DT_SRC'] <= merged_data['COMPLETED_DATE'] + pd.Timedelta(days=1)) &
             (merged_data['COMPLETED_DATE'] + pd.Timedelta(days=1) <= merged_data['DSCH_DT_SRC'])) |
            ((merged_data['ADMIT_DT_SRC'] <= merged_data['COMPLETED_DATE'] + pd.Timedelta(days=2)) &
             (merged_data['COMPLETED_DATE'] + pd.Timedelta(days=2) <= merged_data['DSCH_DT_SRC']))
        )
    )
    op_condition = (
        (merged_data['pat_type'] == 'OP') &
        (
            (merged_data['COMPLETED_DATE'] == merged_data['VISIT_DT_SRC']) |
            (merged_data['COMPLETED_DATE'] - pd.Timedelta(days=1) == merged_data['VISIT_DT_SRC']) |
            (merged_data['COMPLETED_DATE'] - pd.Timedelta(days=2) == merged_data['VISIT_DT_SRC']) |
            (merged_data['COMPLETED_DATE'] + pd.Timedelta(days=1) == merged_data['VISIT_DT_SRC']) |
            (merged_data['COMPLETED_DATE'] + pd.Timedelta(days=2) == merged_data['VISIT_DT_SRC'])

        )
    )

    filtered_df = merged_data[ip_condition | op_condition]
    df = filtered_df.drop_duplicates()
    
    df.sort_values(by=['MRN', 'CompletedDTTM'], inplace=True)
    df.rename(columns={'stroke': 'ICDStrokeLabel', 'updated_stroke_label': 'NLPw48hrs'}, inplace=True)
    df['ICDStrokeLabel'] = df['ICDStrokeLabel'].astype(bool)
    df['CompletedDTTM'] = pd.to_datetime(merged_data['CompletedDTTM'], format='%Y-%m-%d %H:%M:%S')
    
    # Determine what is considered a visit
    last_datetimes = {} 
    global_visit_number = 0 
    visit_numbers = []
    
    for index, row in df.iterrows():
        mrn = row['MRN']
        scan_datetime = row['CompletedDTTM']
        if mrn not in last_datetimes:
            last_datetimes[mrn] = None
        if last_datetimes[mrn] and abs((scan_datetime - last_datetimes[mrn]).total_seconds()) <= 48 * 60 * 60:
            pass
        else:
            global_visit_number += 1

        last_datetimes[mrn] = scan_datetime
        visit_numbers.append(global_visit_number)

    df['VisitNumber'] = visit_numbers
    df.sort_values(by=['MRN', 'CompletedDTTM', 'VisitNumber'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Determine what ICD codes to keep and what to discard
    df = df.groupby('VisitNumber').apply(process_visit)
    df = df[df['keep']]
    
    # Join caboodle data
    caboodle = pd.read_csv("output_data/caboodle_fs_imaging_02012024.csv", dtype={'PrimaryMrn': str, 'AccessionNumber': str})
    df_with_caboodle = pd.merge(df, caboodle, left_on=['MRN', 'ACC'], right_on=['PrimaryMrn', 'AccessionNumber'], how='left')
    df_with_caboodle.to_csv("output_data/mergedICDScanCaboodle-01312024.csv", sep=',', encoding='utf-8')
    