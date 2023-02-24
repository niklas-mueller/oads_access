from PIL import Image
import os
import multiprocessing
import tqdm
import argparse

def resize(args):
    image_name, tiff_dir, resize_factor = args
    try:
        image = Image.open(os.path.join(tiff_dir, image_name))
    except KeyError:
        return
    image = image.reduce(resize_factor)
    image.save(fp=os.path.join(tiff_dir, str(resize_factor), image_name))  


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', help='Path to input directory.',
                        default='/home/niklas/projects/data/oads')
    args = parser.parse_args()
    # tiff_dir = "/home/niklas/projects/data/oads/oads_arw/tiff"

    resize_factor = 2

    tiff_dir = args.input_dir
    images = os.listdir(tiff_dir)
    new_folder = os.listdir(os.path.join(tiff_dir, str(resize_factor)))
    images = [x for x in images if x not in new_folder]

    os.makedirs(os.path.join(tiff_dir, str(resize_factor)), exist_ok=True)
    tiff_dir = [tiff_dir for _ in range(len(images))]
    resize_factor = [resize_factor for _ in range(len(images))]


    with multiprocessing.Pool(12) as pool:
        _ = list(tqdm.tqdm(pool.imap(resize, zip(images, tiff_dir, resize_factor)), total=len(images)))


    print("Successfully resized tiff!")