import numpy as np
import cv2
import os
from timeit import default_timer as timer
from deepgaze.saliency_map import FasaSaliencyMapping 

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
from result_manager.result_manager import ResultManager


image_folder = "/home/niklas/projects/data/oads/oads_arw/tiff"

# image_names = ['0.tiff', '1.tiff']
image_names = os.listdir(image_folder)

result_manager = ResultManager(root='/home/niklas/projects/oads_access/results')

images_per_page = 2


bins = [4, 8, 16]
# tot_bins = 8
format = 'BGR2LAB'

fig, ax = plt.subplots(images_per_page, len(bins)+1 )

figs = []
# for each image the same operations are repeated
for index, image_name in enumerate(image_names):

    image = cv2.imread(os.path.join(image_folder, image_name))

    my_map = FasaSaliencyMapping(image.shape[0], image.shape[1])  # init the saliency object
    start = timer()
    for bin_index, tot_bins in enumerate(bins):

        image_salient = my_map.returnMask(image, tot_bins=tot_bins, format=format)  # get the mask from the original image
        image_salient = cv2.GaussianBlur(image_salient, (3,3), 1)  # applying gaussin blur to make it pretty

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_salient = cv2.cvtColor(image_salient, cv2.COLOR_BGR2RGB)
        ax[index%images_per_page][bin_index+1].imshow(image_salient)
        ax[index%images_per_page][bin_index+1].axis('off')
        ax[index%images_per_page][bin_index+1].set_title(f'{tot_bins} bins')

    end = timer()
    print(f"--- {(end - start)} Image {index} tot seconds ---")

    ax[index%images_per_page][0].imshow(image)
    ax[index%images_per_page][0].axis('off')
    ax[index%images_per_page][0].set_title(image_name)

    if index % images_per_page == images_per_page-1:
        figs.append(fig)
        fig, ax = plt.subplots(images_per_page, len(bins)+1 )

result_manager.save_pdf(figs=figs, filename='saliency_maps.pdf')
