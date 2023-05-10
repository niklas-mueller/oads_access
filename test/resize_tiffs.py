from PIL import Image
import os
import multiprocessing
import tqdm
import argparse
import yaml

def resize(args):
    image_name, tiff_dir, target_size = args
    try:
        image = Image.open(os.path.join(tiff_dir, image_name))
    except KeyError:
        return
    image = image.resize(target_size)
    image.save(fp=os.path.join(tiff_dir, 'reduced', image_name))  


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', help='Path to input directory.',
                        default='/home/niklas/projects/data/oads')
    args = parser.parse_args()
    # tiff_dir = "/home/niklas/projects/data/oads/oads_arw/tiff"
    tiff_dir = '/mnt/c/Users/nikla/OADS Missing Images from Camera/Upload_to_drive/ARW/tiff'

    target_size = (2155, 1440)

    # with open('/home/niklas/projects/oads_experiment/both_raw_and_jpg.yml', 'r') as f:
    #     both_raw_and_jpg = yaml.load(stream=f, Loader=yaml.UnsafeLoader)
    # tiff_dir = args.input_dir
    images = os.listdir(tiff_dir)
    # new_folder = os.listdir(os.path.join(tiff_dir, 'reduced'))
    images = [x for x in images if x not in tiff_dir] # and x.split('.')[0] in both_raw_and_jpg]

    os.makedirs(os.path.join(tiff_dir, 'reduced'), exist_ok=True)
    tiff_dir = [tiff_dir for _ in range(len(images))]
    target_size = [target_size for _ in range(len(images))]



    with multiprocessing.Pool(12) as pool:
        _ = list(tqdm.tqdm(pool.imap(resize, zip(images, tiff_dir, target_size)), total=len(images)))


    print("Successfully resized tiff!")