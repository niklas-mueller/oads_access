import tqdm
import numpy as np
import os
import numpy as np
from PIL import Image
from pytorch_utils.pytorch_utils import ToOpponentChannel, ToRGBEdges
from oads_access.oads_access import OADS_Access, get_annotation_dimensions
from oads_access.utils import loadmat
from pytorch_utils.pytorch_utils import ToRGBEdges

def load_edge_map_crops_for_image(image_name, ToEdges):
    tup = oads.load_image(image_name=image_name)

    if tup is None:
        print(f'tup is none: {image_name}')
        return None
    
    img, label = tup

    img = ToEdges(img)

    is_raw = label['is_raw']
    # if self.use_jpeg:
    #     img = ToJpeg(resize=False, p=self.jpeg_p, quality=self.jpeg_quality)(img)
    #     # is_raw = False

    # This index needs to be normalized/adjust somehow
    crops = {}
    for index in range(len(label['objects'])):
        obj = label['objects'][index]

        ((left, top), (right, bottom)) = get_annotation_dimensions(obj, is_raw=is_raw)
        crop = []
        for _x in img.transpose((2,0,1)):
            crop.append(np.array(Image.fromarray(_x).crop(((left, top, right, bottom)))))

        parvo = np.dstack((crop[0], crop[1], crop[2]))
        magno = np.dstack((crop[3], crop[4], crop[5]))

        edges = np.dstack((crop[0], crop[1], crop[2], crop[3], crop[4], crop[5]))

        crops[index] = {'edges': edges, 'parvo': parvo, 'magno': magno, 'image_name': image_name, 'class': obj['classTitle']}

    return crops

def make_and_safe_crops(args):
    image_id, ToEdges, crop_dir, save_in_one_file = args

    crops = load_edge_map_crops_for_image(image_name=image_id, ToEdges=ToEdges)

    for index, crop in crops.items():
        class_name = crop['class']

        os.makedirs(os.path.join(crop_dir, class_name), exist_ok=True)

        if save_in_one_file:
            edges = crop['edges']
            np.save(os.path.join(crop_dir, class_name, f'{image_id}_{index}.npy'), arr=edges, allow_pickle=True)
        else:
            parvo = crop['parvo']
            magno = crop['magno']
            np.save(os.path.join(crop_dir, class_name, f'{image_id}_{index}_parvo.npy'), arr=parvo, allow_pickle=True)
            np.save(os.path.join(crop_dir, class_name, f'{image_id}_{index}_magno.npy'), arr=magno, allow_pickle=True)

if __name__ == "__main__":
    oads = OADS_Access(basedir='/home/nmuller/projects/data/oads/', n_processes=8)
    
    crop_dir = '/home/nmuller/projects/data/oads/oads_arw/crops/ML_edges'

    save_in_one_file = True


    threshold_lgn_path = f'{os.path.expanduser("~")}/projects/lgnpy/ThresholdLGN.mat'
    default_config_path = f'{os.path.expanduser("~")}/projects/lgnpy/lgnpy/CEandSC/default_config.yml'
    threshold_lgn = loadmat(threshold_lgn_path)['ThresholdLGN']


    image_ids = []
    for image_name, info in oads.image_names.items():
        image_ids.append(image_name)

    image_ids = list(set(image_ids))


    ToEdges = ToRGBEdges(threshold_lgn=threshold_lgn, default_config_path=default_config_path)


    for image_id in tqdm.tqdm(image_ids):
        make_and_safe_crops((image_id, ToEdges, crop_dir, save_in_one_file))
        break