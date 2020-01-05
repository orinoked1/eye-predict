import sys
sys.path.append('../')
import pandas as pd
from modules.data.visualization import DataVis
from modules.data.stim import Stim
import os
import yaml


path = os.getcwd()
expconfig = "/modules/config/experimentconfig.yaml"
with open(path + expconfig, 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

stimSnack = Stim(cfg['exp']['etp']['stimSnack']['name'], cfg['exp']['etp']['stimSnack']['id'], cfg['exp']['etp']['stimSnack']['size'])
stimFace = Stim(cfg['exp']['etp']['stimFace']['name'], cfg['exp']['etp']['stimFace']['id'], cfg['exp']['etp']['stimFace']['size'])

print("Log.....Reading fixation data")
fixation_df = pd.read_pickle(path + "/etp_data/processed/fixation_df__40_subjects.pkl")
print("Log.....Reading Scanpath data")
scanpath_df = pd.read_pickle(path + "/etp_data/processed/scanpath_df__40_subjects.pkl")

datavis = DataVis(cfg['exp']['etp']['stim_path'], cfg['exp']['etp']['visualization_path'], [stimSnack, stimFace], 1)

datavis.visualize(fixation_df, scanpath_df)