import datetime
import rawpy
from oads_access.oads_access import OADS_Access
from PIL import Image
import os
import multiprocessing
import tqdm


def to_tiff(args):
    image_name, basedir, tiff_dir, is_indoor = args
    # image_name=image_name.split(".")[0]
    try:
        # image, _ = oads.load_image(image_name=image_name)
        with rawpy.imread(os.path.join(basedir, image_name)) as raw:
            if is_indoor:
                image = raw.postprocess(no_auto_bright=True, gamma=(30,30))
            else:
                image = raw.postprocess()
    except KeyError:
        print('keyerror')
        return
    image_name=image_name.split(".")[0]
    pil_image = Image.fromarray(image)
    filename = f"{image_name}.tiff"
    # pil_image.save(fp=os.path.join(tiff_dir, filename))  

    # Also resize directly
    target_size = (2155, 1440)
    pil_image = pil_image.resize(target_size)
    pil_image.save(fp=os.path.join(tiff_dir, 'reduced', filename))


if __name__ == '__main__':
    is_indoor = False

    # oads = OADS_Access(basedir="/home/niklas/projects/data/oads")
    # basedir = '/mnt/c/Users/nikla/OADS Missing Images from Camera/Upload_to_drive/ARW'
    basedir = '/home/niklas/projects/data/oads/oads_arw'
    # oads = OADS_Access(basedir=basedir)
    tiff_dir =os.path.join(basedir, 'tiff')
    os.makedirs(tiff_dir, exist_ok=True)

    basedir = os.path.join(basedir, 'ARW')
    
    images = os.listdir(os.path.join(tiff_dir, 'reduced'))

    # existing_tiff_images = [x.split('.')[0] for x in os.listdir(tiff_dir)]
    # args = [(x, basedir, tiff_dir, is_indoor) for x in images if x.split('.')[0] not in existing_tiff_images]
    # args = [(x, basedir, tiff_dir, is_indoor) for x in images]
    args = []
    for x in images:
        
        dt = datetime.datetime.fromtimestamp(os.path.getmtime(os.path.join(tiff_dir, 'reduced', x)))
        if dt.day == 15 and dt.month == 6:
            # print(dt)
            args.append(
                (f"{x.split('.')[0]}.ARW", basedir, tiff_dir, is_indoor)
            )


    with multiprocessing.Pool(multiprocessing.cpu_count()-2) as pool:
        _ = list(tqdm.tqdm(pool.imap(to_tiff, args), total=len(args)))


    print("Successfully converted to tiff!")