from PIL import Image
import os
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from result_manager.result_manager import ResultManager
from oads_access.oads_access import OADS_Access
import numpy as np
import sys
sys.path.append('/home/niklas/projects/lgnpy')
sys.path.append('/home/niklas/projects/mouse_lgn')
from lgnpy.CEandSC.lgn_statistics import *
from lgn_statistics import loadmat
from imscatter import imscatter
from scipy.stats import binned_statistic_2d



def get_random_crops(img: Image.Image, n_crops:int, crop_size:tuple):
    crops = []
    size = img.size
    for index in range(n_crops):
        left = np.random.randint(low=0, high=size[0] - crop_size[0])
        upper = np.random.randint(low=0, high=size[1] - crop_size[1])
        right = left+crop_size[0]
        lower = upper+crop_size[1]
        box = (left, upper, right, lower)
        crop = img.crop(box)
        crops.append((crop, box, index))

    return crops


def get_bin_values(data, bins, min_count):
    # Get index of bin for each entry in data
    indices = np.digitize(x=data, bins=bins)

    index_array = np.array([i for i in range(len(data))])

    # Check whether each index would actually occur the number of times (min_count) otherwise update
    min_count = min(min_count, min([len(index_array[indices == i]) for i in range(len(bins))]))
    
    # For each bin, gather the indices of the data array, shuffle them and only take min_count many to balance the size of the bins
    # This index array can then be used to get the images corresponding to the CE/SC values that were chosen, now having a uniform distribution
    bin_indices = np.array([np.random.permutation(index_array[indices == i])[:min_count] for i in range(len(bins))])

    # Get the actual data points for each bin
    bin_values = np.array([data[bin_indices[i]] for i in range(len(bins))])
    return bin_indices, bin_values

def plot_images(images, fig=None, max_images:int=None, figsize=(15,10)):
    if max_images is None:
        max_images = len(images)

    if fig is None:
        fig = plt.figure(figsize=figsize)

    # if max_images % 2 == 1:
    #     n_rows = int(np.floor(np.sqrt(max_images)))
    #     n_cols = int(np.floor(np.sqrt(max_images)))
    # else:
    n_rows = int(np.floor(np.sqrt(max_images)))
    n_cols = int(np.ceil(max_images / n_rows))

    index = 0
    for _ in range(n_rows):
        for _ in range(n_cols):
            if index >= max_images:
                break
            ax = fig.add_subplot(n_rows,n_cols,index+1)
            ax.imshow(images[index])
            ax.axis('off')
            index += 1

    return fig

basedir = '/home/niklas/projects/data/oads'
image_names = os.listdir(os.path.join(basedir, 'oads_arw', 'tiff'))

number_crops_per_image = 60
crop_size = (1900, 1200)

threshold_lgn = loadmat('/home/niklas/projects/LGNstatistics-master/CEandSCmatlab/ThresholdLGN.mat')
threshold_lgn = threshold_lgn['ThresholdLGN']

result_manager = ResultManager(root='/home/niklas/projects/oads_access/analysis/results')

results = result_manager.load_result('lgn_statistics.pkl')
force_recompute = True or results is None

figs = []


if force_recompute:
    ce_results = []
    sc_results = []
    beta_results = []
    gamma_results = []
    filenames = []
    crop_indices = [[] for _ in range(len(image_names))]
    crop_boxes = [[] for _ in range(len(image_names))]

    print(f"Number of Images: {len(image_names)}")
    for image_index, image_name in enumerate(image_names):
        # fig, ax = plt.subplots(1,number_crops_per_image+1, figsize=(30,10))
        # if image_index > 5:
        #     break
        img = Image.open(os.path.join(basedir, 'oads_arw', 'tiff', image_name))
        size = img.size
        # ax[0].imshow(img)
        # ax[0].axis('off')

        # ... , we compute the variance map on the whole image and move the "fovea" around to compute the mean of the signal !!!!!!!!!!!
        crops = get_random_crops(img=img, n_crops=number_crops_per_image, crop_size=crop_size)
        crop_masks = []
        for crop_index, (crop, box, index) in enumerate(crops):
            mask = np.zeros(np.array(img).shape[:2], dtype=np.bool8)
            mask[box[0]:box[2],box[1]:box[3]] = True
            crop_masks.append(mask)
            crop_boxes[image_index].append(box)
            crop_indices[image_index].append(index)
            # ax[crop_index+1].imshow(crop)
            # ax[crop_index+1].axis('off')

        ce, sc, beta, gamma = lgn_statistics(im=np.array(img), file_name=image_name, threshold_lgn=threshold_lgn, crop_masks=crop_masks, compute_extra_statistics=True)
        ce_results.append(ce)
        sc_results.append(sc)

        # ce_extra_results.append(ce_extra)
        # sc_extra_results.append(sc_extra)
        beta_results.append(beta)
        gamma_results.append(gamma)


        # figs.append(fig)

    results = {'CE': ce_results, 'SC': sc_results, 'Beta': beta_results, 'Gamma': gamma_results, 'filenames': filenames, 'crop_boxes': crop_boxes, 'crop_indices': crop_indices}
    result_manager.save_result(result=results, filename='lgn_statistics.pkl', overwrite=True)

else:
    ce_results = results['CE']
    sc_results = results['SC']
    beta_results = results['Beta']
    gamma_results = results['Gamma']
    filenames = results['filenames']
    crop_indices = results['crop_indices']
    crop_boxes = results['crop_boxes']
#     results = result_manager.load_result('lgn_statistics.pkl')


#########################

print("Creating histograms")

# ce = np.array([np.mean(_x) for _x in ce_extra_results])
# sc = np.array([np.mean(_x) for _x in sc_extra_results])
ce = np.array([np.mean(_x[:,i,0]) for _x in ce_results for i in range(1, _x.shape[1])]) # (color, boxes , center/peripherie)
sc = np.array([np.mean(_x[:,i,0]) for _x in sc_results for i in range(1, _x.shape[1])]) # (color, boxes , center/peripherie)


ce_peri = np.array([np.mean(_x[:,i,1]) for _x in ce_results for i in range(1, _x.shape[1])]) # (color, boxes , center/peripherie)
sc_peri = np.array([np.mean(_x[:,i,1]) for _x in sc_results for i in range(1, _x.shape[1])]) # (color, boxes , center/peripherie)

fig, ax = plt.subplots(1,2, figsize=(15,10))
ax[0].hist((ce-ce_peri)/ce_peri)
ax[0].set_title('CE Crops')
ax[0].set_xlabel('(Center - Peripheri) / Peripheri')
ax[1].hist((sc-sc_peri)/sc_peri)
ax[1].set_title('SC Crops')
ax[1].set_xlabel('(Center - Peripheri) / Peripheri')

figs.append(fig)

#############
# Create the imscatter based on the 
#############

# beta = np.array([np.mean(_x) for _x in beta_results])
# gamma = np.array([np.mean(_x) for _x in gamma_results])

ce_lower = np.percentile(ce, q=5)
ce_upper = np.percentile(ce, q=95)
sc_lower = np.percentile(sc, q=5)
sc_upper = np.percentile(sc, q=95)

####
count_ce, division_ce = np.histogram(ce[(ce >= ce_lower) & (ce <= ce_upper)], bins=5)

min_count_ce = count_ce.min()
min_value_ce = division_ce.min()
max_value_ce = division_ce.max()

print(min_count_ce, min_value_ce, max_value_ce)

count_sc, division_sc = np.histogram(sc[(sc >= sc_lower) & (sc <= sc_upper)], bins=5)

min_count_sc = count_sc.min()
min_value_sc = division_sc.min()
max_value_sc = division_sc.max()

print(min_count_sc, min_value_sc, max_value_sc)

# ###################
# # Try to create a 2d histogram woth SC and CE

# fig, (ax, ax0) = plt.subplots(1,2, figsize=(10,5))
# # hist, xedges, yedges, mesh = ax.hist2d(ce[(ce >= ce_lower) & (ce <= ce_upper)], sc[(sc >= sc_lower) & (sc <= sc_upper)], bins=5)
# hist,_,_, bins_indices = binned_statistic_2d(ce[(ce >= ce_lower) & (ce <= ce_upper)], sc[(sc >= sc_lower) & (sc <= sc_upper)], values=None, statistic='count', bins=5, expand_binnumbers=True)
# index_array = np.array([[i for i in range(bins_indices.shape[1])] for _ in range(bins_indices.shape[0])])

# min_count = int(np.min(hist))

# balanced_bin_indices = []
# for bin_index in range(1, 6):
#     _sc_indices = index_array[0,bins_indices[0,:] == bin_index]
#     _ce_indices = index_array[1,bins_indices[1,:] == bin_index]
#     _cut = np.intersect1d(_sc_indices, _ce_indices)
#     _cut = np.random.permutation(_cut)
#     _cut = _cut[:min_count]
#     balanced_bin_indices.append(_cut)
# # balanced_bin_indices = [np.random.permutation(np.intersect1d(index_array[0,bins_indices[0,:] == i], index_array[1,bins_indices[1,:] == i]))[:min_count] for i in range(5)]
# index_overlap = np.array(balanced_bin_indices).flatten()
# values_overlap_ce = ce[index_overlap]
# values_overlap_sc = sc[index_overlap]

# bar = ax.imshow(hist)
# ax.set_xlabel('CE')
# ax.set_ylabel('SC')
# ax.set_title('50% CI')
# plt.colorbar(bar, ax=ax)

# ax0.hist2d(values_overlap_ce, values_overlap_sc, bins=5)
# ax0.set_xlabel('CE')
# ax0.set_ylabel('SC')
# # plt.colorbar(bar, ax=ax0)
# figs.append(fig)
# ###################


#####
# Create a uniform distribution for CE and SC individuall and then select the indices that occur in both -> leads to a non-uniform distribution afterwards
fig, ax = plt.subplots(2,2, figsize=(10,5))

indices_ce, values_ce = get_bin_values(data=ce, bins=division_ce, min_count=min_count_ce)
ax[0, 0].hist(ce, bins=division_ce, label='CE all')
ax[0, 0].hist(values_ce.flatten(), bins=division_ce, label='CE Uniform Selection')
ax[0, 0].set_xlabel('CE')
ax[0, 0].legend()

indices_sc, values_sc = get_bin_values(data=sc, bins=division_sc, min_count=min_count_sc)
ax[0, 1].hist(sc, bins=division_sc, label='SC all')
ax[0, 1].hist(values_sc.flatten(), bins=division_sc, label='SC Uniform Selection')
ax[0, 1].set_xlabel('SC')
ax[0, 1].legend()


## Plot the indices
index_overlap = np.intersect1d(indices_ce.flatten(), indices_sc.flatten())
print(index_overlap)
ax[1,0].bar(x=indices_ce.flatten(), height=1, label='CE')
ax[1,0].bar(x=index_overlap, height=0.5, label='Overlap CE+SC')
ax[1,0].set_title('CE Uniform Selection Images')
ax[1,0].set_xlabel('Index')
ax[1,0].set_yticklabels([])
ax[1,1].bar(x=indices_sc.flatten(), height=1, label='SC')
ax[1,1].bar(x=index_overlap, height=0.5, label='Overlap CE+SC')
ax[1,1].set_title('SC Uniform Selection Images')
ax[1,1].set_xlabel('Index')
ax[1,1].set_yticklabels([])

ax[1,0].legend()
ax[1,1].legend()

plt.tight_layout()
# plt.show()

figs.append(fig)


##########
## print image to use
fig, ax = plt.subplots(1, len(index_overlap), figsize=(15,5))

im_fig, (im_ax, im_ax_all) = plt.subplots(1,2, figsize=(15,10))

#
im_fig = plt.figure(figsize=(25,15))
gs = GridSpec(4, 8)

im_ax = im_fig.add_subplot(gs[1:4, 4:7])
ax_hist_sc = im_fig.add_subplot(gs[1:4, 7])
ax_hist_sc.hist(sc[index_overlap], bins=division_sc, label='SC Uniform', orientation = 'horizontal')

ax_hist_ce = im_fig.add_subplot(gs[0, 4:7])
ax_hist_ce.hist(ce[index_overlap], bins=division_ce, label='CE Uniform')

im_ax_all = im_fig.add_subplot(gs[1:4, 0:3])
ax_hist_sc_all = im_fig.add_subplot(gs[1:4,3])
ax_hist_sc_all.hist(sc, bins=division_sc, label='SC all', orientation = 'horizontal')

ax_hist_ce_all = im_fig.add_subplot(gs[0, 0:3])
ax_hist_ce_all.hist(ce, bins=division_ce, label='CE all')
##

crops = []
for index in index_overlap:
    image_index = int(index%len(image_names))
    img = Image.open(os.path.join(basedir, 'oads_arw', 'tiff', image_names[image_index]))
    crop = img.crop(crop_boxes[image_index][int(index%number_crops_per_image)])
    crops.append(crop)

    crop = crop.reduce(4)

    imscatter(ce[index], sc[index], np.array(crop), zoom=0.1, ax=im_ax)
    im_ax.scatter(ce[index], sc[index])

fig = plot_images(crops)
figs.append(fig)

im_ax.set_title('Crops by CE and SC - Uniform Selection')
im_ax.set_xlabel('CE')
im_ax.set_ylabel('SC')



for image_index, image_name in enumerate(image_names):
    for crop_index in range(number_crops_per_image):
        img = Image.open(os.path.join(basedir, 'oads_arw', 'tiff', image_names[image_index]))
        crop = img.crop(crop_boxes[image_index][crop_index])
        crop = crop.reduce(4)

        imscatter(ce[image_index*crop_index], sc[image_index*crop_index], np.array(crop), zoom=0.1, ax=im_ax_all)
        im_ax_all.scatter(ce[image_index*crop_index], sc[image_index*crop_index])

im_ax_all.set_title("Crops by CE and SC - All Crops")
im_ax_all.set_xlabel('CE')
im_ax_all.set_ylabel('SC')
figs.append(im_fig)

im_fig.tight_layout()
###########


result_manager.save_pdf(figs=figs, filename='oads_crop_statistics.pdf')