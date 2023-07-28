import os
nproc = 6

os.environ["OMP_NUM_THREADS"] = str(nproc)
os.environ["OPENBLAS_NUM_THREADS"] = str(nproc)
os.environ["MKL_NUM_THREADS"] = str(nproc)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(nproc)
os.environ["NUMEXPR_NUM_THREADS"] = str(nproc)

import tqdm
import numpy as np
import os
import numpy as np
from PIL import Image
from pytorch_utils.pytorch_utils import ToOpponentChannel, ToRGBEdges
from oads_access.oads_access import OADS_Access, get_annotation_dimensions
from oads_access.utils import loadmat
from pytorch_utils.pytorch_utils import ToRGBEdges
import multiprocessing

def load_edge_map_crops_for_image(image_name, ToEdges):
    tup = oads.load_image(image_name=image_name)

    if tup is None:
        print(f'tup is none: {image_name}')
        return None
    
    img, label = tup

    img = ToEdges(img)

    return {'edges': img, 'image_name': image_name}

    # is_raw = label['is_raw']
    # # if self.use_jpeg:
    # #     img = ToJpeg(resize=False, p=self.jpeg_p, quality=self.jpeg_quality)(img)
    # #     # is_raw = False

    # # This index needs to be normalized/adjust somehow
    # crops = {}
    # for index in range(len(label['objects'])):
    #     obj = label['objects'][index]

    #     ((left, top), (right, bottom)) = get_annotation_dimensions(obj, is_raw=is_raw)
    #     crop = []
    #     for _x in img.transpose((2,0,1)):
    #         crop.append(np.array(Image.fromarray(_x).crop(((left, top, right, bottom)))))

    #     parvo = np.dstack((crop[0], crop[1], crop[2]))
    #     magno = np.dstack((crop[3], crop[4], crop[5]))

    #     edges = np.dstack((crop[0], crop[1], crop[2], crop[3], crop[4], crop[5]))

    #     crops[index] = {'edges': edges, 'parvo': parvo, 'magno': magno, 'image_name': image_name, 'class': obj['classTitle']}

    # return crops

def make_and_safe_edges(args):
    image_id, ToEdges, edge_dir, save_in_one_file = args

    if os.path.exists(os.path.join(edge_dir, f'{image_id}.npy')):
        return

    edges = load_edge_map_crops_for_image(image_name=image_id, ToEdges=ToEdges)

    if save_in_one_file:
        edges = edges['edges']
        np.save(os.path.join(edge_dir, f'{image_id}.npy'), arr=edges, allow_pickle=True)
    else:
        parvo = edges['parvo']
        magno = edges['magno']
        np.save(os.path.join(edge_dir,  f'{image_id}_parvo.npy'), arr=parvo, allow_pickle=True)
        np.save(os.path.join(edge_dir,  f'{image_id}_magno.npy'), arr=magno, allow_pickle=True)

if __name__ == "__main__":
    oads = OADS_Access(basedir='/home/nmuller/projects/data/oads/', n_processes=8)
    
    edge_dir = '/home/nmuller/projects/data/oads/oads_arw/edges'

    save_in_one_file = True


    threshold_lgn_path = f'{os.path.expanduser("~")}/projects/lgnpy/ThresholdLGN.mat'
    default_config_path = f'{os.path.expanduser("~")}/projects/lgnpy/lgnpy/CEandSC/default_config.yml'
    threshold_lgn = loadmat(threshold_lgn_path)['ThresholdLGN']


    image_ids = []
    for image_name, info in oads.image_names.items():
        image_ids.append(image_name)

    image_ids = list(set(image_ids))


    ToEdges = ToRGBEdges(threshold_lgn=threshold_lgn, default_config_path=default_config_path)

    with multiprocessing.Pool(nproc) as pool:
        results = pool.map(make_and_safe_edges, [(image_id, ToEdges, edge_dir, save_in_one_file) for image_id in image_ids])
    
    # for image_id in tqdm.tqdm(image_ids):
    #     make_and_safe_edges((image_id, ToEdges, edge_dir, save_in_one_file))
    #     # break