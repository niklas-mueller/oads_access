import tqdm
import numpy as np
import os
import numpy as np
from PIL import Image
from pytorch_utils.pytorch_utils import ToOpponentChannel, ToRGBEdges
from oads_access.oads_access import OADS_Access
from oads_access.utils import loadmat
from pytorch_utils.pytorch_utils import ToRGBEdges

if __name__ == "__main__":
    oads = OADS_Access(basedir='/home/nmuller/projects/data/oads/', n_processes=8)
    
    crop_dir = '/home/nmuller/projects/data/oads/oads_arw/crops/ML_edges'


    threshold_lgn_path = f'{os.path.expanduser("~")}/projects/lgnpy/ThresholdLGN.mat'
    default_config_path = f'{os.path.expanduser("~")}/projects/lgnpy/lgnpy/CEandSC/default_config.yml'
    threshold_lgn = loadmat(threshold_lgn_path)['ThresholdLGN']


    image_ids = []
    for image_name, info in oads.image_names.items():
        # if len(dataset_names) > 0 and info['dataset_name'] not in dataset_names:
        #     continue
        for index, _ in oads.image_names[image_name]['object_labels'].items():
                image_ids.append((image_name, index))

    image_ids = list(set(image_ids))


    ToEdges = ToRGBEdges(threshold_lgn=threshold_lgn, default_config_path=default_config_path)


    for image_id in tqdm.tqdm(image_ids):
        image_name, index = image_id
        (img, obj) = oads.load_crop_from_image(image_name=image_name, index=index)
        class_name = obj['classTitle']
        edges = ToEdges(img)

        os.makedirs(os.path.join(crop_dir, class_name), exist_ok=True)

        np.save(os.path.join(crop_dir, class_name, f'{image_name}_{index}.npy'), arr=edges, allow_pickle=True)