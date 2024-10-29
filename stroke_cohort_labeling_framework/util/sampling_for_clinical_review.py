import pandas as pd


def stratified_sample(df, strata, n_samples):
    # Get unique combinations of the strata
    unique_strata = df[strata].drop_duplicates()

    # Calculate the number of unique combinations
    n_strata = len(unique_strata)

    # Initialize an empty DataFrame for the stratified sample
    stratified_sample = pd.DataFrame()

    # Sampling from each stratum
    for _, stratum_values in unique_strata.iterrows():
        # Filter the dataframe for the stratum
        stratum_df = df[(df[strata[0]] == stratum_values[strata[0]]) & (df[strata[1]] == stratum_values[strata[1]])]
        
        # Calculate the sample size for the stratum
        stratum_sample_size = min(len(stratum_df), n_samples // n_strata)

        # Sample from the stratum and add to the stratified sample
        stratified_sample = pd.concat([stratified_sample, stratum_df.sample(stratum_sample_size)])

    return stratified_sample

if __name__ == '__main__':
    df = pd.read_csv('/Users/erekaa02/neurology/output_data/UniqueStrokeCodeEncountersForSampling.csv')
    sample = stratified_sample(df, ['Year', 'TrueStrokeCodeActivated'], 600)

    encounter_keys = set(sample['EncounterKey'])
    for key in encounter_keys:
        associated_rows = df[df['EncounterKey'] == key]
        sample = pd.concat([sample, associated_rows]).drop_duplicates()

    sample.to_csv('StratifiedUniqueStrokeCodeEncounters.csv')