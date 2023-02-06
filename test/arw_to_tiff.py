from oads_access.oads_access import OADS_Access
from PIL import Image
import os
import multiprocessing
import tqdm

oads = OADS_Access(basedir="/home/niklas/projects/data/oads")

tiff_dir = "/home/niklas/projects/data/oads/oads_arw/tiff"

os.makedirs(tiff_dir, exists_ok=True)
images = os.listdir("/home/niklas/projects/data/oads/oads_arw/ARW")

def to_tiff(image_name):
    image_name=image_name.split(".")[0]
    try:
        image, _ = oads.load_image(image_name=image_name)
    except KeyError:
        return
    pil_image = Image.fromarray(image)
    filename = f"{image_name}.tiff"
    pil_image.save(fp=os.path.join(tiff_dir, filename))  

with multiprocessing.Pool(12) as pool:
    _ = list(tqdm.tqdm(pool.imap(to_tiff, images), total=len(images)))


print("Successfully converted to tiff!")