from oads_access.oads_access import OADS_Access
from PIL import Image
import os
import multiprocessing
import tqdm

oads = OADS_Access(basedir="/home/niklas/projects/data/oads")

# data = oads.get_data_list()

tiff_dir = "/home/niklas/projects/data/oads/oads_arw/tiff"

#os.makedirs(tiff_dir, exists_ok=True)
images = os.listdir("/home/niklas/projects/data/oads/oads_arw/ARW")

def to_tiff(args):
    index, image_name = args
    try:
        image, _ = oads.load_image(image_name=image_name.split(".")[0])
    except KeyError as e:
        return
    pil_image = Image.fromarray(image)
    filename = f"{index}.tiff"
    pil_image.save(fp=os.path.join(tiff_dir, filename))  

with multiprocessing.Pool() as pool:
    _ = list(tqdm.tqdm(pool.imap(to_tiff, enumerate(images)), total=len(images)))


# for index, image_name in enumerate(images):  
#     try:
#         image, _ = oads.load_image(image_name=image_name.split(".")[0])
#     except KeyError as e:
#         continue
#     pil_image = Image.fromarray(image[0])
#     filename = f"{index}.tiff"
#     pil_image.save(fp=os.path.join(tiff_dir, filename))  

print("Successfully converted to tiff!")