import nilearn
from nilearn import plotting, masking
from nilearn import datasets, image
from nilearn.maskers import NiftiLabelsMasker
import seaborn as sns
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import zipfile, os, multiprocessing, itertools
import scipy.stats as stat
from sklearn.metrics import DistanceMetric

stat_event_map = {
"pe1":	"wait",
"pe2":	"go_45_prep",
"pe3":	"go_45_prep 1st deriv",
"pe4":	"go_99_prep",
"pe5":	"go_99_prep 1st deriv",
"pe6":	"go_153_prep",
"pe7":	"go_153_prep 1st deriv",
"pe8":	"go_207_prep",
"pe9":	"go_207_prep 1st deriv",
"pe10":	"nogo_45_prep",
"pe11":	"nogo_45_prep 1st deriv",
"pe12":	"nogo_99_prep",
"pe13":	"nogo_99_prep 1st deriv",
"pe14":	"nogo_153_prep",
"pe15":	"nogo_153_prep 1st deriv",
"pe16":	"nogo_207_prep",
"pe17":	"nogo_207_prep 1st deriv",
"pe18":	"movement_45",
"pe19":	"movement_45 1st deriv",
"pe20":	"movement_99",
"pe21":	"movement_99 1st deriv",
"pe22":	"movement_153",
"pe23":	"movement_153 1st deriv",
"pe24":	"movement_207",
"pe25":	"movement_207 1st deriv",
"pe26":	"non_movement_45",
"pe27":	"non_movement_45 1st deriv",
"pe28":	"non_movement_99",
"pe29":	"non_movement_99 1st deriv",
"pe30":	"non_movement_153",
"pe31":	"non_movement_153 1st deriv",
"pe32":	"non_movement_207",
"pe33":	"non_movement_207 1st deriv"
}

event_stat_map = dict([(value, key) for key, value in stat_event_map.items()])
dataset_juelich = datasets.fetch_atlas_juelich("maxprob-thr25-2mm")
masks = {"premotor":"GM Premotor cortex BA6", 
         "motor": "GM Primary motor cortex BA4a",
         "v1": "GM Visual cortex V1 BA17",
         "sup_parietal": "GM Superior parietal lobule 7P",
         "inf_parietal": "GM Inferior parietal lobule PF",
         "somatosensory": "GM Primary somatosensory cortex BA1"
        }

dataset_msdl = datasets.fetch_atlas_msdl()
masks = {"motor": "Motor", 
         "parietal": "L Par",
         "vis": "Vis",
         "cerebellum": "Cereb"       
        }
# atlas_filename = dataset_ho['maps']

def extract_roi_signals_juelich(path, conditions, region, masks=masks, dataset_juelich=dataset_juelich):
    #determine mask image from juelich atlas
    mask_image = image.new_img_like(dataset_juelich.maps, image.get_data(dataset_juelich.maps) == np.where(np.array(dataset_juelich.labels) == masks[region])[0][0])
    print(path)
    #Apply the mask to data for all (33) conditions where each condition corresponds to stat map (output of fsl feat)
    fmri_masked = [
        masking.apply_mask(image.load_img(path+f'{event_stat_map[condition]}.nii.gz'), 
        mask_img = image.resample_to_img(mask_image, image.load_img(path+f'{event_stat_map[condition]}.nii.gz'), 
        interpolation='nearest')
        )
        for condition in conditions  
    ]
    return fmri_masked