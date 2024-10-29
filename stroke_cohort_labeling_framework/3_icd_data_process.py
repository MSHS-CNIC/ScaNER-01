""" Step3: This is used only if I have more ICD data to process.
It is the file used for aggregating the IP and OP ICD codes raw data.
it is a pre-processing step for ICD data.
"""

import pandas as pd
import glob


def stroke_icd_codes(df):
    condition1 = (df['ICD_VERSION'] == 0) & (
    df['SEC_DIAG_CD'].str.contains('^I61.*') |
    df['SEC_DIAG_CD'].str.contains('^I63.*') |
    df['SEC_DIAG_CD'].str.contains('^I64.*') |
    df['SEC_DIAG_CD'].str.contains('^H341.*') |
    df['SEC_DIAG_CD'].str.contains('^I60.*') |
    df['SEC_DIAG_CD'].str.contains('^G45(?!4)')
)

    condition2 = (df['ICD_VERSION'] == 9) & (
        df['SEC_DIAG_CD'].str.contains('^431.*') |
        df['SEC_DIAG_CD'].str.contains('^433.*1$') |
        df['SEC_DIAG_CD'].str.contains('^434.*[^0]$') |
        df['SEC_DIAG_CD'].str.contains('^436.*') |
        df['SEC_DIAG_CD'].str.contains('^435.*') |
        df['SEC_DIAG_CD'].str.contains('^430.*')
    )

    df['stroke'] = condition1 | condition2
    return df


def load_accession_data(data_path):
    dfs = []
    for file in glob.glob(data_path):
        df = pd.read_csv(file)
        
        df = df[(df['DIAG_TYPE'] != '0') & (df['DIAG_TYPE'] != 'A')]
        
        icd_9_condition = (df['ICD_VERSION'] == 9)
        icd_10_condition = (
            (df['ICD_VERSION'] == 0) & 
            (
                ((df['SEC_DIAG_NUM'] == 1) & (df['FACILITY_MSX'] != 'MSH')) |
                ((df['SEC_DIAG_NUM'].isin([1, 2])) & (df['FACILITY_MSX'] == 'MSH'))
            )
        )
        df = df[icd_9_condition | icd_10_condition]
        df = stroke_icd_codes(df)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=False)


def unique_concat(series):
    # Convert all items to strings before joining
    return ', '.join(map(str, series.unique()))


ip_data_path = "data/ip_msx_other_icd_*.csv"
ip_icd_codes = load_accession_data(ip_data_path)
ip_icd_codes['DSCH_DT_SRC'] = pd.to_datetime(ip_icd_codes['DSCH_DT_SRC'], format='%d-%b-%y')
ip_icd_codes['ADMIT_DT_SRC'] = pd.to_datetime(ip_icd_codes['ADMIT_DT_SRC'], format='%d-%b-%y')
ip_icd_codes.to_csv("output_data/IPICD8k.csv", index=False)


op_data_path = "data/op_msx_other_icd_*.csv"
op_icd_codes = load_accession_data(op_data_path)
op_icd_codes['VISIT_DT_SRC'] = pd.to_datetime(op_icd_codes['VISIT_DT_SRC'], format='%d-%b-%y')
op_icd_codes.to_csv("output_data/OPICD8k.csv", index=False)

# ip_icd_codes = pd.read_csv('output_data/IPICD8k.csv')
# op_icd_codes = pd.read_csv('output_data/OPICD8k.csv')
ip_icd_codes['pat_type'] = 'IP'
op_icd_codes['pat_type'] = 'OP'
op_icd_codes.rename(columns={'ENCOUNTER_NO_SRC': 'ENCOUNTER_NO'}, inplace=True)

icd_codes = pd.concat([ip_icd_codes, op_icd_codes]).reset_index(drop=True)

icd_codes['MSMRN'] = icd_codes['MSMRN'].astype('string')
icd_codes['ENCOUNTER_NO'] = icd_codes['ENCOUNTER_NO'].astype('string')

aggregated_df = icd_codes.groupby(['MSMRN', 'ENCOUNTER_NO', 'stroke']).agg(unique_concat)
aggregated_df = aggregated_df.reset_index()
aggregated_df.to_csv('output_data/aggregated_ICD_data.csv')

