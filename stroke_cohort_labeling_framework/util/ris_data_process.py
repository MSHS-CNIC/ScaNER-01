import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
import math

from sample_size_estimation import required_sample_size_1

def load_data():
    os.chdir("/Users/erekaa02/neurology/data")
    all_files = glob.glob("accession*.xls")
    combined_ris_df = []

    for file in all_files:
        df = pd.read_excel(file)
        combined_ris_df.append(df)

    return pd.concat(combined_ris_df, ignore_index=True)


accesion_data = load_data()
print(accesion_data.columns)
print(accesion_data.dtypes)

data_size = len(accesion_data)

pd.options.mode.use_inf_as_na = True

"""
Get number/% of empty notes prior to pre-processing
"""
empty_notes_number = (
    accesion_data['ReportText']
    .apply(lambda x: pd.isna(x) or not str(x).strip())
    .sum()
)

empty_notes_percent = empty_notes_number/data_size
print("Records without a Note prior to pre-processing", empty_notes_percent)

"""
Get number/% of empty order date/time prior to pre-processing
"""
empty_order_date = accesion_data['OrderedDTTM'].isna().sum()
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
accesion_data = accesion_data.dropna(subset=['OrderedDTTM'])

accesion_data['ordered_date'] = accesion_data['OrderedDTTM'].dt.date
accesion_data['completed_date'] = accesion_data['CompletedDTTM'].dt.date

print("Minimum date", min(accesion_data['completed_date']))
print("Maximum date", max(accesion_data['completed_date']))

accesion_data['diff_ordered_completed_minute'] = (accesion_data['CompletedDTTM'] - accesion_data['OrderedDTTM'])/ np.timedelta64(1, 'h')

# Filter data from outliers
# Calculate the IQR
Q1 = accesion_data['diff_ordered_completed_minute'].quantile(0.25)
Q3 = accesion_data['diff_ordered_completed_minute'].quantile(0.75)
IQR = Q3 - Q1

# Define a threshold for outliers
threshold = 1.5 * IQR

# Filter outliers and negative difference values
filtered_data = accesion_data[
    (accesion_data['diff_ordered_completed_minute'] >= Q1 - threshold) 
    & (accesion_data['diff_ordered_completed_minute'] <= Q3 + threshold)
    & (accesion_data['diff_ordered_completed_minute'] >= 0)
    ]

plt.hist(filtered_data['diff_ordered_completed_minute'], bins=200, edgecolor='black')
plt.xlabel('Hours')
plt.ylabel('Frequency')
plt.title('Distribution of Ordered vs. Completed Data')
plt.show()

# Get unique values for exam code
unique_exam_codes = filtered_data['ExamCode'].unique()
print("Unique Exam Codes: ", unique_exam_codes)

# Get year only based on ordered time
filtered_data['year'] = filtered_data['OrderedDTTM'].dt.year
filtered_data = filtered_data[filtered_data['year'] >= 2010]

# Group by 'Year' and 'Category' and calculate the counts
category_counts = filtered_data.groupby(['year', 'ExamCode']).size().unstack(fill_value=0)
category_counts.plot(kind='bar', stacked=True)
plt.xlabel('Year')
plt.ylabel('Counts')
plt.title('Stacked Bar Plot of Categories by Year')
plt.show()

# Calculate needed sample size
N = len(filtered_data)
Z = 1.96
p = 0.5
E = 0.2
print(N)
sample_size = required_sample_size_1(N, Z, p, E)
print(sample_size)


# # Initialize the StratifiedShuffleSplit with the desired number of splits
# # we only need one split
# stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=sample_size, random_state=42)

# # Create the train and test indices using stratified sampling based on 'ExamCode' and 'year'
# for train_index, test_index in stratified_split.split(filtered_data, filtered_data['year']):
#     stratified_sample = filtered_data.iloc[test_index]

# print(stratified_sample)
