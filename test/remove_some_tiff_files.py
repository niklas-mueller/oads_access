import os
import shutil
import tqdm

eeg_stimulus_dir = '/mnt/z/Projects/2023_Scholte_FMG1441/Stimuli/reduced'

eeg_stimulus_filenames = [x for x in os.listdir(eeg_stimulus_dir) if x.endswith('.tiff')]

tiff_dir = '/home/niklas/projects/data/oads/oads_arw/tiff'

# print(os.listdir(tiff_dir)[0], eeg_stimulus_filenames[0])
for x in tqdm.tqdm(os.listdir(tiff_dir)):
    if x in eeg_stimulus_filenames:
        continue
    else:
        if os.path.isdir(os.path.join(tiff_dir, x)):
            continue
        os.unlink(os.path.join(tiff_dir, x))