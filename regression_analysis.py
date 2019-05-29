import os
import pandas as pd


y = os.getcwd()
# read csv into DF
raw_data_df = pd.read_csv(y + '/raw_data_01.csv')

raw_data_df.bid = round(raw_data_df.bid)


print('x')