import pandas as pd


imaging_w_fs = pd.read_csv("/Users/erekaa02/neurology/data/stroke_fs_imaging_w_pat_status_01302024.csv")
imaging_wo_fs = pd.read_csv("/Users/erekaa02/neurology/data/stroke_imaging_w_pat_status_01302024.csv")

df_concatenated = pd.concat([imaging_w_fs, imaging_wo_fs])
df_concatenated.to_csv("output_data/caboodle_fs_imaging.csv")
