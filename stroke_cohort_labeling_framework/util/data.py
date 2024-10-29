import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
import yaml

from util.sample_size_estimation import required_sample_size_1


def load_neuro_codes(yaml_path='data/neuro_codes.yml'):
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    return data['neuro_codes']

def load_accession_data():
    neuro_codes = load_neuro_codes()

    dfs = []
    for file in glob.glob("data/accession*.xls"):
        df = pd.read_excel(file)
        dfs.append(df[df['ExamCode'].isin(neuro_codes)])

    return pd.concat(dfs, ignore_index=False)
    

if __name__ == '__main__':
    # accesion_data = load_accession_data()
    accession_data = pd.read_csv("output_data/NeuroAccessionsLabeledv13.csv")
    # accession_data = accession_data[accession_data['Already reviewed? 1, 0'] == 0]
    print(accession_data.columns)
    print(accession_data.dtypes)

    data_size = len(accession_data)

    pd.options.mode.use_inf_as_na = True

    """
    Get number/% of empty notes prior to pre-processing
    """
    empty_notes_number = (
        accession_data['ReportText']
        .apply(lambda x: pd.isna(x) or not str(x).strip())
        .sum()
    )

    empty_notes_percent = empty_notes_number/data_size
    print("Records without a Note prior to pre-processing", empty_notes_percent)

    """
    Get number/% of empty order date/time prior to pre-processing
    """
    empty_order_date = accession_data['OrderedDTTM'].isna().sum()
    empty_order_date_percent = empty_order_date/data_size
    print("Records without an order date prior to pre-processing", empty_order_date_percent)

    """
    Pre-pocess assumptions:
    1. Imputing orderedDTTM when appropriate
    2. Cleaning notes and merge when appropriate
    3. Remove order date nans
    4. Remove when difference between Completed vs. ordered negative
    5. Check the difference between Completed vs. ordered for when ordered date exists
    """
    accession_data = accession_data.dropna(subset=['ReportText'])

    # accession_data['ordered_date'] = accession_data['OrderedDTTM'].dt.date
    accession_data['year'] = pd.to_datetime(accession_data['CompletedDTTM']).dt.year
    # accession_data['sampling_unique_id'] = accession_data['completed_date'].astype(str) + "," + accession_data['MRN']
    print(accession_data)
    # print("Minimum date", min(accesion_data['completed_date']))
    # print("Maximum date", max(accesion_data['completed_date']))

    # accesion_data['diff_ordered_completed_minute'] = (accesion_data['CompletedDTTM'] - accesion_data['OrderedDTTM'])/ np.timedelta64(1, 'h')

    # # Filter data from outliers
    # # Calculate the IQR
    # Q1 = accesion_data['diff_ordered_completed_minute'].quantile(0.25)
    # Q3 = accesion_data['diff_ordered_completed_minute'].quantile(0.75)
    # IQR = Q3 - Q1

    # # Define a threshold for outliers
    # threshold = 1.5 * IQR

    # # Filter outliers and negative difference values
    # filtered_data = accesion_data[
    #     (accesion_data['diff_ordered_completed_minute'] >= Q1 - threshold) 
    #     & (accesion_data['diff_ordered_completed_minute'] <= Q3 + threshold)
    #     & (accesion_data['diff_ordered_completed_minute'] >= 0)
    #     ]

    # plt.hist(filtered_data['diff_ordered_completed_minute'], bins=200, edgecolor='black')
    # plt.xlabel('Hours')
    # plt.ylabel('Frequency')
    # plt.title('Distribution of Ordered vs. Completed Data')
    # plt.show()

    # # Get unique values for exam code
    # unique_exam_codes = filtered_data['ExamCode'].unique()
    # print("Unique Exam Codes: ", unique_exam_codes)
    
    # # Get year only based on ordered time
    # filtered_data['year'] = filtered_data['OrderedDTTM'].dt.year
    # filtered_data = filtered_data[filtered_data['year'] >= 2010]

    # # Group by 'Year' and 'Category' and calculate the counts
    # category_counts = filtered_data.groupby(['year', 'ExamCode']).size().unstack(fill_value=0)
    # category_counts.plot(kind='bar', stacked=True)
    # plt.xlabel('Year')
    # plt.ylabel('Counts')
    # plt.title('Stacked Bar Plot of Categories by Year')
    # plt.show()

    # Calculate needed sample size
    N = 20000
    print(N)
    Z = 1.96
    p = 0.5
    E = 0.2
    print(N)
    sample_size = required_sample_size_1(N, Z, p, E)
    print(sample_size)

    # # 1. Calculate proportional counts
    # target_samples = 300
    # proportional_counts = (accession_data['year'].value_counts(normalize=True) * target_samples).round().astype(int)

    # # 2. Adjust counts
    # while proportional_counts.sum() != target_samples:
    #     if proportional_counts.sum() < target_samples:
    #         # Find the year with the highest fractional part that's not yet adjusted and increment its count
    #         fractional_parts = (accession_data['year'].value_counts(normalize=True) * target_samples) % 1
    #         year_to_adjust = fractional_parts.idxmax()
    #         proportional_counts[year_to_adjust] += 1
    #     else:
    #         # Find the year with the smallest fractional part that's not yet adjusted and decrement its count
    #         fractional_parts = (accession_data['year'].value_counts(normalize=True) * target_samples) % 1
    #         year_to_adjust = fractional_parts.idxmin()
    #         proportional_counts[year_to_adjust] -= 1

    # # 3. Stratified sampling based on adjusted counts
    # stratified_sample = pd.concat([
    #     accession_data[accession_data['year'] == year].sample(n=count)
    #     for year, count in proportional_counts.items()
    # ])
    # stratified_sample.to_csv('300StratifiedSamples.csv', index=False)
    