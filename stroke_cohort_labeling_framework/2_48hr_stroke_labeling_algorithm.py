"""Step2: This file is used to get the 48hrs labels after running the main algorithm.
"""
import pandas as pd
from tqdm import tqdm

labeled_data = pd.read_csv('output_data/AllNeuroAccessionsLabeledv22.csv')
# labeled_data = pd.read_excel('data/one_patient_test.xlsx')
excluded_scans = ['CANHEAD1', 'CANHEAD2', 'CANHEANEC1', 'CANNECK1', 'CANNECK2', 'MANHEAD0', 'MANHEAD1', 'MANHEAD2', 'MANNECK0', 'MANNECK1', 'MANNECK2']
labeled_data = labeled_data[~labeled_data['ExamCode'].isin(excluded_scans)]
labeled_data['CompletedDTTM'] = pd.to_datetime(labeled_data['CompletedDTTM'])

labeled_data['updated_stroke_label'] = False

# Sort data by MRN and CompletedDTTM to ensure correct chronological processing
labeled_data.sort_values(by=['MRN', 'CompletedDTTM'], inplace=True)

# Iterate over exams, identify visit times, and update labels
for mrn, group in tqdm(labeled_data.groupby('MRN'), desc="Processing MRNs"):
    positive_indices = []  # List to keep track of indices where positive labels are detected

    # Iterate over the rows in the current group
    for index, row in group.iterrows():
        # Skip if the row label is NaN
        if pd.isna(row['Label']):
            continue

        # If a positive stroke label is found, add index to positive_indices
        if 'StrokeLabel.POSITIVE' in row['Label']:
            positive_indices.append(index)

    # Iterate over the positive indices and update the labels within the 48-hour window
    for pos_index in positive_indices:
        pos_time = group.at[pos_index, 'CompletedDTTM']
        # Define the time window
        window_start = pos_time - pd.Timedelta(hours=48)
        window_end = pos_time

        # Update the records within the 48-hour window before the positive label
        window_mask = (group['CompletedDTTM'] >= window_start) & (group['CompletedDTTM'] <= window_end)
        labeled_data.loc[group[window_mask].index, 'updated_stroke_label'] = True

labeled_data.to_csv('output_data/AllNeuroAccessionsLabeledV22W48hrs.csv', index=False)
