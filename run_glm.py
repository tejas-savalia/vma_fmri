from nilearn.image import load_img, mean_img
from nilearn.plotting import plot_stat_map, view_img, plot_design_matrix
from nilearn.glm.first_level import make_first_level_design_matrix, FirstLevelModel
import os
import numpy as np
import pandas as pd

# sub_list = np.array([str(i).zfill(2) for i in np.delete(np.arange(1, 26), [9, 20])])
# PPt 25 doesn't have confounds.tsv for run 6 onwards...
sub_list = np.array(['26'])

runs = np.array([str(i).zfill(2) for i in np.arange(1, 11)])
tasks = np.tile(['straight', 'rotate'], 5)

    
for sub in sub_list:
    for i, run in enumerate(runs):

        task = tasks[i]
        
        output_dir = f'glm_outputs/basic_contrasts/sub-{sub}/run-{run}_{task}'

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        #Handle confounds            
        confounds = pd.read_csv(f'fmriprep_derivatives/sub-{sub}/ses-S1/func/sub-{sub}_ses-S1_task-{task}_run-{run}_desc-confounds_timeseries.tsv', delimiter='\t')
        #Chosen based on fMRIprep usage page. 
        chosen_confounds = confounds[[i for i in list(confounds.columns) if i.startswith('trans')] + [i for i in list(confounds.columns) if i.startswith('rot')] + [i for i in list(confounds.columns) if i.startswith('csf')][:-1] + [i for i in list(confounds.columns) if i.startswith('white_matter')] + [i for i in list(confounds.columns) if i.startswith('global_signal')] + [i for i in list(confounds.columns) if i.startswith('motion_outlier')] + [i for i in list(confounds.columns) if i.startswith('non_steady_state_outlier')]+ ['framewise_displacement', 'rmsd', 'dvars', 'std_dvars'] + [i for i in list(confounds.columns) if i.startswith('cosine')] + [i for i in list(confounds.columns) if i.startswith('t_comp')] + [i for i in list(confounds.columns) if i.startswith('a_comp_cor')][:5]]
        chosen_confounds.fillna(0, inplace = True)

        #Handle Event Files
        events = pd.read_csv(f'event_files/sub-{sub}/run-{run}/events.csv').drop('Unnamed: 0', axis = 1)

        #Load Image files
        func_img_path = f"fmriprep_derivatives/sub-{sub}/ses-S1/func/sub-{sub}_ses-S1_task-{task}_run-{run}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
        mask_img_path = f"fmriprep_derivatives/sub-{sub}/ses-S1/func/sub-{sub}_ses-S1_task-{task}_run-{run}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz"
        func_img = load_img(func_img_path)
        mask_img = load_img(mask_img_path)


        #Setting up GLMs
        first_level_model = FirstLevelModel(t_r = 1.25, mask_img=mask_img, smoothing_fwhm=5, standardize=True, n_jobs = -1)
        fmri_glm = first_level_model.fit(run_imgs=func_img, events=events, confounds=chosen_confounds)

        for contrast in events.trial_type.unique():
            zmap = fmri_glm.compute_contrast(f"{contrast}")
            zmap.to_filename(f"{output_dir}/{contrast}.nii.gz")
            
            # if contrast != "wait":
            #     zmap = fmri_glm.compute_contrast(f"{contrast} - wait")
            #     zmap.to_filename(f"{output_dir}/{contrast}.nii.gz")
    print("Sub done: ", sub)