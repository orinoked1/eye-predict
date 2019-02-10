import visualize_data_functions as vis
import pickle
import pandas as pd



with open('fixation_dataset_v1.pkl', 'rb') as f:
    fixation_dataset = pickle.load(f)



x = vis.map(fixation_dataset[2], fixation_dataset[3], 'bdm_bmm_short_data/stim/', fixation_dataset[0],  showSalMap=False)

print('x')
