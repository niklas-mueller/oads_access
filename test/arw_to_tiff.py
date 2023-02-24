import rawpy
from oads_access.oads_access import OADS_Access
from PIL import Image
import os
import multiprocessing
import tqdm

# oads = OADS_Access(basedir="/home/niklas/projects/data/oads")
basedir = '/mnt/c/Users/nikla/OneDrive/PhD/Projects/OADS Indoor Images (Hongye)'
# oads = OADS_Access(basedir=basedir)
tiff_dir = "/mnt/c/Users/nikla/OneDrive/PhD/Projects/OADS Indoor Images (Hongye)/oads_arw/tiff"

os.makedirs(tiff_dir, exist_ok=True)
images = os.listdir("/mnt/c/Users/nikla/OneDrive/PhD/Projects/OADS Indoor Images (Hongye)/oads_arw/Camera")

def to_tiff(image_name):
    # image_name=image_name.split(".")[0]
    try:
        # image, _ = oads.load_image(image_name=image_name)
        with rawpy.imread(os.path.join(basedir, 'oads_arw', 'Camera', image_name)) as raw:
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