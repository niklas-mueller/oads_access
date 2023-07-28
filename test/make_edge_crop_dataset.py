import multiprocessing
import tqdm
import numpy as np
import os
import numpy as np
from PIL import Image
from pytorch_utils.pytorch_utils import ToOpponentChannel, ToRGBEdges
from oads_access.oads_access import OADS_Access, get_annotation_dimensions
from oads_access.utils import loadmat
from pytorch_utils.pytorch_utils import ToRGBEdges

def iterate(args):
    label, index, img, image_name, is_raw, save_in_one_file, crop_dir = args
    
    
    obj = label['objects'][index]

    # if os.path.exists(os.path.join(crop_dir, obj['classTitle'], f'{image_name}_{index}.npy')):
    #     return
    
    ((left, top), (right, bottom)) = get_annotation_dimensions(obj, is_raw=is_raw)
    crop = []
    for _x in img.transpose((2,0,1)):
        crop.append(np.array(Image.fromarray(_x).crop(((left, top, right, bottom)))))

    # parvo = np.dstack((crop[0], crop[1], crop[2]))
    # magno = np.dstack((crop[3], crop[4], crop[5]))

    edges = np.dstack((crop[0], crop[1], crop[2], crop[3], crop[4], crop[5]))

    # crops[index] = {'edges': edges, 'parvo': parvo, 'magno': magno, 'image_name': image_name, 'class': obj['classTitle']}

    if save_in_one_file:
        os.makedirs(os.path.join(crop_dir, obj['classTitle']), exist_ok=True)

        if save_in_one_file:
            # edges = crop['edges']
            np.save(os.path.join(crop_dir, obj['classTitle'], f'{image_name}_{index}.npy'), arr=edges, allow_pickle=True)

def load_edge_map_crops_for_image(tup, image_name, ToEdges, save_in_one_file:bool=False, crop_dir:str='', multiprocess:bool=True):
    # tup = oads.load_image(image_name=image_name)

    if tup is None:
        print(f'tup is none: {image_name}')
        return None
    
    img, label = tup
    # all_done = True
    # for index in range(len(label['objects'])):
    #     if not os.path.exists(os.path.join(crop_dir, label['objects'][index]['classTitle'], f'{image_name}_{index}.npy')):
    #         all_done = False

    # if all_done:
    #     # print(f'Skipping {image_name}')
    #     return

    img = ToEdges(img)

    is_raw = label['is_raw']
    # if self.use_jpeg:
    #     img = ToJpeg(resize=False, p=self.jpeg_p, quality=self.jpeg_quality)(img)
    #     # is_raw = False

    if multiprocess:
        with multiprocessing.Pool(12) as pool:
            pool.map(iterate, [(label, index, img, image_name, is_raw, save_in_one_file, crop_dir) for index in range(len(label['objects']))])
    
    else:

        for index in range(len(label['objects'])):
            # obj = label['objects'][index]

            # ((left, top), (right, bottom)) = get_annotation_dimensions(obj, is_raw=is_raw)
            # crop = []
            # for _x in img.transpose((2,0,1)):
            #     crop.append(np.array(Image.fromarray(_x).crop(((left, top, right, bottom)))))

            # edges = np.dstack((crop[0], crop[1], crop[2], crop[3], crop[4], crop[5]))

            # if save_in_one_file:
            #     os.makedirs(os.path.join(crop_dir, obj['classTitle']), exist_ok=True)

            #     if save_in_one_file:
            #         edges = crop['edges']
            #         np.save(os.path.join(crop_dir, obj['classTitle'], f'{image_name}_{index}.npy'), arr=edges, allow_pickle=True)
            iterate((label, index, img, image_name, is_raw, save_in_one_file, crop_dir))

    # This index needs to be normalized/adjust somehow
    # crops = {}
    # for index in range(len(label['objects'])):
    #     obj = label['objects'][index]

    #     ((left, top), (right, bottom)) = get_annotation_dimensions(obj, is_raw=is_raw)
    #     crop = []
    #     for _x in img.transpose((2,0,1)):
    #         crop.append(np.array(Image.fromarray(_x).crop(((left, top, right, bottom)))))

    #     # parvo = np.dstack((crop[0], crop[1], crop[2]))
    #     # magno = np.dstack((crop[3], crop[4], crop[5]))

    #     edges = np.dstack((crop[0], crop[1], crop[2], crop[3], crop[4], crop[5]))

    #     # crops[index] = {'edges': edges, 'parvo': parvo, 'magno': magno, 'image_name': image_name, 'class': obj['classTitle']}
    #     crops[index] = {'edges': edges, 'image_name': image_name, 'class': obj['classTitle']}

    # return crops

def make_and_safe_crops(args):
    image_id, oads, ToEdges, crop_dir, save_in_one_file, multiprocess = args
    tup = oads.load_image(image_name=image_id)
    # print('load')
    load_edge_map_crops_for_image(tup=tup, image_name=image_id, ToEdges=ToEdges, save_in_one_file=save_in_one_file, crop_dir=crop_dir, multiprocess=multiprocess)
    # print(f'Done {image_id}')
    # image_id, ToEdges, crop_dir, save_in_one_file = args

    # crops = load_edge_map_crops_for_image(image_name=image_id, ToEdges=ToEdges)

    # for index, crop in crops.items():
    #     class_name = crop['class']

    #     os.makedirs(os.path.join(crop_dir, class_name), exist_ok=True)

    #     if save_in_one_file:
    #         edges = crop['edges']
    #         np.save(os.path.join(crop_dir, class_name, f'{image_id}_{index}.npy'), arr=edges, allow_pickle=True)
    #     else:
    #         parvo = crop['parvo']
    #         magno = crop['magno']
    #         np.save(os.path.join(crop_dir, class_name, f'{image_id}_{index}_parvo.npy'), arr=parvo, allow_pickle=True)
    #         np.save(os.path.join(crop_dir, class_name, f'{image_id}_{index}_magno.npy'), arr=magno, allow_pickle=True)

if __name__ == "__main__":
    home_path = os.path.expanduser('~')
    oads_dir = f'{home_path}/projects/data/oads/'
    # oads_dir = f'/mnt/c/Users/nikla/OneDrive\ -\  UvA/OADS_Backup/ARW/'
    oads = OADS_Access(basedir=oads_dir, n_processes=12)
    
    # crop_dir = f'{home_path}/projects/data/oads/oads_arw/crops/ML_edges'
    crop_dir = f'/mnt/z/Projects/2023_Scholte_FMG1441/oads_edge_crops'

    save_in_one_file = True


    threshold_lgn_path = f'{home_path}/projects/lgnpy/ThresholdLGN.mat'
    default_config_path = f'{home_path}/projects/lgnpy/lgnpy/CEandSC/default_config.yml'
    threshold_lgn = loadmat(threshold_lgn_path)['ThresholdLGN']


    # image_ids = oads.image_names.keys()
    # image_ids = list(set(image_ids))
    with open('/mnt/c/Users/nikla/OneDrive/PhD/Projects/oads_edgemaps_correpted_files.txt', 'r') as f:
        filenames = f.readlines()
    image_ids = [x.split(',')[-1].split('_')[0] for x in filenames]

    ToEdges = ToRGBEdges(threshold_lgn=threshold_lgn, default_config_path=default_config_path)

    with multiprocessing.Pool(6) as pool:
        results = list(tqdm.tqdm(pool.imap(make_and_safe_crops, [(image_id, oads, ToEdges, crop_dir, save_in_one_file, False) for image_id in image_ids]), total=len(image_ids)))
    


    # for image_id in tqdm.tqdm(image_ids):
    #     make_and_safe_crops((image_id, oads, ToEdges, crop_dir, save_in_one_file, True))