import rawpy
from oads_access.oads_access import OADS_Access
from PIL import Image
import os
import multiprocessing
import tqdm

is_indoor = True

# oads = OADS_Access(basedir="/home/niklas/projects/data/oads")
basedir = '/mnt/c/Users/nikla/OADS Missing Images from Camera/Upload_to_drive/ARW'
# oads = OADS_Access(basedir=basedir)
tiff_dir =os.path.join(basedir, 'tiff')

os.makedirs(tiff_dir, exist_ok=True)
images = os.listdir(os.path.join(basedir))

def to_tiff(image_name):
    # image_name=image_name.split(".")[0]
    try:
        # image, _ = oads.load_image(image_name=image_name)
        with rawpy.imread(os.path.join(basedir, image_name)) as raw:
            if is_indoor:
                image = raw.postprocess(no_auto_bright=True, gamma=(10,10))
            else:
                image = raw.postprocess()
    except KeyError:
        print('keyerror')
        return
    image_name=image_name.split(".")[0]
    pil_image = Image.fromarray(image)
    filename = f"{image_name}.tiff"
    pil_image.save(fp=os.path.join(tiff_dir, filename))  

with multiprocessing.Pool(12) as pool:
    _ = list(tqdm.tqdm(pool.imap(to_tiff, images), total=len(images)))


print("Successfully converted to tiff!")