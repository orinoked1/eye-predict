
import torch
import pandas as pd
import os

use_cuda = torch.cuda.is_available()

print(use_cuda)

x = pd.read_csv('eye_tracking_data_parser/bdm_bmm_short_data_df.csv')

#df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}, index=['a', 'b', 'c'])
#df.to_hdf('data.h5', key='df', mode='w')

os.remove('./*h5')

#original_df = pd.DataFrame({"foo": range(5), "bar": range(5, 10)})
#original_df.to_pickle("./dummy.pkl")


#Read pikle into DF
file_name = 'bdm_bmm_short_data_df'
df = pd.read_pickle('./eye_tracking_data_parser/' + file_name)
df