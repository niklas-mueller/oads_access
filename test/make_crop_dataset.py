import time
import os
os.environ['MKL_NUM_THREADS'] = '5'
os.environ['OPENBLAS_NUM_THREADS'] = '5'
os.environ["NUMEXPR_NUM_THREADS"] = "5" 
from pathlib import Path
import multiprocessing
import tqdm
from oads_access.oads_access import OADS_Access



def iterate(args):
    # (image_id, index), c, oads, crop_dir = args
    c, oads, crop_dir, image_list = args
    os.makedirs(os.path.join(crop_dir, c), exist_ok=True)
    for x in tqdm.tqdm(image_list):
        image_id, index = x.split("_")

        if f'{c}_{image_id}_{index}.tiff' in os.listdir(os.path.join(crop_dir, c)):
            Path(os.path.join(crop_dir, c, f'{c}_{image_id}_{index}.tiff')).unlink()

        if f'{image_id}_{index}.tiff' in os.listdir(os.path.join(crop_dir, c)):
            continue        

        # print(c, image_id, index)
        img, _ = oads.load_crop_from_image(image_name=image_id, index=int(index))

        img.save(os.path.join(crop_dir, c, f'{image_id}_{index}.tiff'))

if __name__ == '__main__':
    home_path = os.path.expanduser('~')
    data_dir = f'{home_path}/projects/data/oads'

    oads = OADS_Access(basedir=data_dir)

    # print(oads.images_per_class.keys())#, oads.images_per_class['Tree'])

    crop_dir = os.path.join(data_dir, 'oads_arw', 'crops', 'ML')
    os.makedirs(crop_dir, exist_ok=True)

    # with multiprocessing.Pool(5) as pool:
        # pool.imap(lambda x: time.sleep(10), range(100))
        # results = pool.map(iterate, [(x.split('_'), c, oads, crop_dir) for x in image_list])
        # results = list(tqdm.tqdm(pool.imap(iterate, [(c, oads, crop_dir, image_list) for c, image_list in oads.images_per_class.items() if len(image_list) > len(os.listdir(os.path.join(crop_dir, c)))])))

    tupels = list(oads.images_per_class.items())
    c, image_list = tupels[19]

    # for c, image_list in tqdm.tqdm():
    if len(image_list) == len(os.listdir(os.path.join(crop_dir, c))):
        print(c)
    else:
        # print(len(image_list), len(os.listdir(os.path.join(crop_dir, c))))
        iterate((c, oads, crop_dir, image_list))

