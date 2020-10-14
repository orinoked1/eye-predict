
from modules.data.datasets import DatasetBuilder
import numpy as np

stimType = 'Snack'

datasetbuilder = DatasetBuilder()
fixation_event_data, saccad_event_data = datasetbuilder.load_data_for_fix_sacc_statistics(stimType)

snack_df = datasetbuilder.get_fixations_scanpath_df(fixation_event_data, stimType)

stimType = 'Face'

datasetbuilder = DatasetBuilder()
fixation_event_data, saccad_event_data = datasetbuilder.load_data_for_fix_sacc_statistics(stimType)

face_df = datasetbuilder.get_fixations_scanpath_df(fixation_event_data, stimType)

face_stim_std = face_df.groupby(['stimName'])['bid'].std().reset_index()
snack_stim_std = snack_df.groupby(['stimName'])['bid'].std().reset_index()

mean_face_std = np.mean(face_stim_std)
mean_snack_std = np.mean(snack_stim_std)


print('v')