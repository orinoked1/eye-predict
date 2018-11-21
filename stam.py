
import torch
import pandas as pd

use_cuda = torch.cuda.is_available()

print(use_cuda)

#Read pikle into DF
file_name = 'bdm_bmm_short_data_df'
df = pd.read_pickle('./eye_tracking_data_parser/' + file_name)
df