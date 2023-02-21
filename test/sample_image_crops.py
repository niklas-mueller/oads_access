import random
import time
import io
from itertools import combinations
from PIL import Image
import os
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from result_manager.result_manager import ResultManager
from oads_access.oads_access import OADS_Access
from oads_access.utils import get_random_crops, get_bin_values, imscatter, imscatter_all, plot_images, joint_plot
import numpy as np
from lgnpy.CEandSC.lgn_statistics import lgn_statistics
from lgnpy.run_LGNstatistics import loadmat
from scipy.stats import binned_statistic_2d
import torch
import multiprocessing
import tqdm

def get_sc_and_ce(args):
    image_name, image_index, number_crops_per_image, crop_size = args
    basedir = '/home/niklas/projects/data/oads'
    img_path = os.path.join(basedir, 'oads_arw', 'tiff', image_name)
    img = Image.open(img_path)
    old_size = os.stat(img_path).st_size

    crop_boxes = []
    crop_indices = []
    crop_jpeg_factors = []

    # ... , we compute the variance map on the whole image and move the "fovea" around to compute the mean of the signal !!!!!!!!!!!
    crops = get_random_crops(img=img, n_crops=number_crops_per_image, crop_size=crop_size)
    crop_masks = []
    
    for crop_index, (crop, box, index) in enumerate(crops):
        mask = np.zeros(np.array(img).shape[:2], dtype=np.bool8)
        mask[box[1]:box[3], box[0]:box[2]] = True
        crop_masks.append(mask)
        crop_boxes.append(box)
        crop_indices.append(index)
        crop = img.crop(box)
        np_crop = np.array(crop)
        old_crop_size = np_crop.itemsize * np_crop.size
        temp_crop = io.BytesIO()
        crop.save(temp_crop, format='jpeg')
        new_crop_size = temp_crop.getbuffer().nbytes
        crop_jpeg_factors.append((old_crop_size, new_crop_size))
        # ax[crop_index+1].imshow(crop)
        # ax[crop_index+1].axis('off')

    # This scales the viewing distance to the images used when viewing distance was =1
    (height, width) = np.array(img).shape[:2]
    viewing_dist = np.mean([height / 480, width / 640])
    config = {'viewing_dist': viewing_dist}
    ce, sc, beta, gamma = lgn_statistics(im=np.array(img), file_name=image_name, config=config, threshold_lgn=threshold_lgn, force_recompute=True, cache=False, 
                                            crop_masks=crop_masks, compute_extra_statistics=True)
    ce_results = ce
    sc_results = sc

    # ce_extra_results = ce_extra
    # sc_extra_results = sc_extra
    beta_results = beta
    gamma_results = gamma

    temp = io.BytesIO() # this is a file object
    img.save(temp, format="jpeg") 
    new_size = temp.getbuffer().nbytes

    # jpg_factor = old_size / new_size
    jpeg_factors = (old_size, new_size)

    return crop_boxes, crop_indices, crop_jpeg_factors, ce_results, sc_results, beta_results, gamma_results, jpeg_factors, image_name, image_index

if __name__ == '__main__':
    basedir = '/home/niklas/projects/data/oads'
    image_names = os.listdir(os.path.join(basedir, 'oads_arw', 'tiff'))

    # image_names = [image_name for image_name in image_names if os.path.exists(f'/home/niklas/projects/lgnpy/cache/results_{image_name}.npy')]

    number_crops_per_image = 30
    crop_size = (1920, 1080)

    threshold_lgn = loadmat('/home/niklas/projects/lgnpy/ThresholdLGN.mat')
    threshold_lgn = threshold_lgn['ThresholdLGN']

    result_manager = ResultManager(root='/home/niklas/projects/oads_access/results')

    results = result_manager.load_result('lgn_statistics.pkl')
    force_recompute = False or results is None

    figs = []


    if force_recompute:
        ce_results = np.ones((len(image_names),3,number_crops_per_image+1,2))
        sc_results = np.ones((len(image_names),3,number_crops_per_image+1,2))
        beta_results = np.ones((len(image_names),3,number_crops_per_image+1,2))
        gamma_results = np.ones((len(image_names),3,number_crops_per_image+1,2))
        jpeg_factors = [(0,0) for _ in range(len(image_names))]
        filenames = ['' for _ in range(len(image_names))]
        crop_indices = [[] for _ in range(len(image_names))]
        crop_boxes = [[] for _ in range(len(image_names))]
        crop_jpeg_factors = [[] for _ in range(len(image_names))]

        print(f"Number of Images: {len(image_names)}")

        args = [(image_name, image_index, number_crops_per_image, crop_size) for image_index, image_name in enumerate(image_names)]
        with multiprocessing.Pool() as pool:
            results = list(tqdm.tqdm(pool.imap(get_sc_and_ce, args), total=len(args)))

        for crop_box, crop_index, crop_jpeg_factor, ce_result, sc_result, beta_result, gamma_result, jpeg_factor, image_name, image_index in results:
            crop_boxes[image_index] = crop_box
            crop_indices[image_index] = crop_index
            crop_jpeg_factors[image_index] = crop_jpeg_factor
            ce_results[image_index] = ce_result
            sc_results[image_index] = sc_result
            beta_results[image_index] = beta_result
            gamma_results[image_index] = gamma_result
            jpeg_factors[image_index] = jpeg_factor
            filenames[image_index] = image_name

        results = {'CE': ce_results, 'SC': sc_results, 'Beta': beta_results, 'Gamma': gamma_results, "jpeg_factors":jpeg_factors, 'filenames': filenames, 'crop_boxes': crop_boxes, 'crop_indices': crop_indices, "crop_jpeg_factors": crop_jpeg_factors}
        result_manager.save_result(result=results, filename='lgn_statistics.pkl', overwrite=True)

    else:
        ce_results = results['CE']
        sc_results = results['SC']
        beta_results = results['Beta']
        gamma_results = results['Gamma']
        filenames = results['filenames']
        crop_indices = results['crop_indices']
        crop_boxes = results['crop_boxes']
        jpeg_factors = results['jpeg_factors']
        crop_jpeg_factors = results['crop_jpeg_factors']
    #     results = result_manager.load_result('lgn_statistics.pkl')


    #########################

    print("Creating histograms")

    # ce = np.array([np.mean(_x) for _x in ce_extra_results])
    # sc = np.array([np.mean(_x) for _x in sc_extra_results])
    ce_full = np.array([np.mean(_x[:,0,0]) for _x in ce_results]) # (color, boxes , center/peripherie)
    sc_full = np.array([np.mean(_x[:,0,0]) for _x in sc_results]) # (color, boxes , center/peripherie)
    ce_peri_full = np.array([np.mean(_x[:,0,1]) for _x in ce_results]) # (color, boxes , center/peripherie)
    sc_peri_full = np.array([np.mean(_x[:,0,1]) for _x in sc_results]) # (color, boxes , center/peripherie)

    ce_crops = np.array([np.mean(_x[:,i,0]) for _x in ce_results for i in range(1, _x.shape[1])]) # (color, boxes , center/peripherie)
    sc_crops = np.array([np.mean(_x[:,i,0]) for _x in sc_results for i in range(1, _x.shape[1])]) # (color, boxes , center/peripherie)
    ce_peri_crops = np.array([np.mean(_x[:,i,1]) for _x in ce_results for i in range(1, _x.shape[1])]) # (color, boxes , center/peripherie)
    sc_peri_crops = np.array([np.mean(_x[:,i,1]) for _x in sc_results for i in range(1, _x.shape[1])]) # (color, boxes , center/peripherie)

    fig, ax = plt.subplots(2,4, figsize=(10,5))
    ax[0, 0].hist(ce_crops)
    ax[0, 0].set_title('CE Center')
    ax[0, 1].boxplot(ce_crops)
    ax[0, 1].set_title('CE Center')
    ax[0, 2].hist(sc_crops)
    ax[0, 2].set_title('SC Center')
    ax[0, 3].boxplot(sc_crops)
    ax[0, 3].set_title('SC Center')
    ax[1, 0].hist(ce_peri_crops)
    ax[1, 0].set_title('CE Peri')
    ax[1, 1].boxplot(ce_peri_crops)
    ax[1, 1].set_title('CE Peri')
    ax[1, 2].hist(sc_peri_crops)
    ax[1, 2].set_title('SC Peri')
    ax[1, 3].boxplot(sc_peri_crops)
    ax[1, 3].set_title('SC Peri')

    fig.tight_layout()

    figs.append(fig)

    ##################

    # One example of the distribution of the full CE/SC vs. crop CE/SC
    fig, ax = plt.subplots(2,1, figsize=(25,15))
    n=100
    crop_dists = np.mean(ce_results[:n,:,1:,0], axis=1).T
    center_dists =np.array([np.mean(ce_results[i,:,0,0]) for i in range(n)])
    sorted_index = np.argsort(center_dists)
    ax[0].boxplot(crop_dists[:, sorted_index])
    ax[0].set_xticks(np.array([i for i, _ in enumerate(center_dists)])[::10])
    ax[0].set_xticklabels(center_dists[::10])
    # ax[0].set_xticklabels([])
    ax[0].set_xlabel('CE of full image, sorted by size')
    ax[0].set_ylabel('Distribution of CE across crops')
    crop_dists = np.mean(sc_results[:n,:,1:,0], axis=1).T
    center_dists =np.array([np.mean(sc_results[i,:,0,0]) for i in range(n)])
    sorted_index = np.argsort(center_dists)
    ax[1].boxplot(crop_dists[:, sorted_index])
    ax[1].set_xticks(np.array([i for i, _ in enumerate(center_dists)])[::10])
    ax[1].set_xticklabels(center_dists[::10])
    # ax[1].set_xticklabels([])
    ax[1].set_xlabel('SC of full image, sorted by size')
    ax[1].set_ylabel('Distribution of SC across crops')
    figs.append(fig)


    ###################

    # print(np.array(jpeg_factors).shape)
    fig, ax = plt.subplots(1,3, figsize=(10,5))
    ax[0].hist(np.array(jpeg_factors)[:,0])    
    ax[0].set_title('File size RAW')
    ax[1].hist(np.array(jpeg_factors)[:,1])    
    ax[1].set_title('File size JPEG')
    ax[2].hist(np.array(jpeg_factors)[:,1] / np.array(jpeg_factors)[:,0])    
    ax[2].set_title('JPEG Comp Factor (New FS / Old FS)')

    fig.tight_layout()
    figs.append(fig)

    """
    Now we want to select crops such that 

    (a) the jpeg compression factor distribution is uniform
    (b) CE/SC of center and periphery are decorrelated
    
    """

    jpeg_comp_rate = np.array(jpeg_factors)[:,1] / np.array(jpeg_factors)[:,0]

    fig, ax = plt.subplots(2,3, figsize=(10,5))

    
    factors = jpeg_comp_rate
    lower = np.percentile(factors, q=5)
    upper = np.percentile(factors, q=95)
    n_bins = 25
    count, bins = np.histogram(factors[(factors > lower) & (factors < upper)], bins=n_bins)
    indices, values = get_bin_values(data=factors, bins=bins, min_count=count.min())

    chosen_sc = np.array(sc_full)[indices.flatten()]
    chosen_ce = np.array(ce_full)[indices.flatten()]
    chosen_sc_peri = np.array(sc_peri_full)[indices.flatten()]
    chosen_ce_peri = np.array(ce_peri_full)[indices.flatten()]
    chosen_factors = np.array(factors)[indices.flatten()]

    ax[0, 0].scatter(sc_full, ce_full, c=factors)
    ax[0, 0].set_xlabel('SC')
    ax[0, 0].set_ylabel('CE')
    ax[0, 0].set_title('All points')

    ax[1, 0].scatter(sc_full, sc_peri_full, c=factors)
    ax[1, 0].set_xlabel('SC')
    ax[1, 0].set_ylabel('CE')
    ax[1, 0].set_title('All points')

    ax[0, 1].scatter(chosen_sc, chosen_ce, c=chosen_factors)
    ax[0, 1].set_xlabel('SC')
    ax[0, 1].set_ylabel('CE')
    # ax[0, 1].set_title(f'Uniform JPEG Distribution: {count.min()} number of points per bin -> {count.min() * n_bins} points')

    # figs.append(fig)


    ##########
    ### Find minimum correlation
    rng = np.random.default_rng()
    sample_size = 995
    min_corr = 1.0
    min_samples = ()
    # Compute single contributions to correlation
    # pick initial set of points
    start = time.time()
    print(len(chosen_ce))
    # combs = combinations(range(len(chosen_ce)), sample_size)
    combs = set()
    N = 10000
    while len(combs) < N:
        combs.add(tuple(sorted(random.sample(range(len(chosen_ce)), sample_size))))
    # np.random.shuffle(combs)
    for i, comb in enumerate(iter(combs)):
        # if i > 200000:
        #     break
        _center = chosen_ce[list(comb)]
        _peri = chosen_ce_peri[list(comb)]
        corr = np.corrcoef(_center, _peri)[1,0]
        if corr < min_corr:
            # min_samples = (_center, _peri)
            min_samples = list(comb)
            min_corr = corr
    end = time.time()
    print(min_corr)
    print(f'Combinations Time: {end-start}')

    ax[0, 2].scatter(chosen_sc[min_samples], chosen_ce[min_samples], c=chosen_factors[min_samples])
    # ax[0, 2].set_xlim((x_min, x_max))
    # ax[0, 2].set_ylim((y_min, y_max))
    ax[0, 2].set_xlabel('SC')
    ax[0, 2].set_ylabel('CE')
    ax[0, 2].set_title(f'{min_corr:.4f}')

    fig.tight_layout()
    figs.append(fig)
    

    # #############
    images = []
    titles = []
    tups = []
    for index in range(len(sc_full)):
        if os.path.exists(os.path.join(basedir, 'oads_arw', 'tiff', filenames[index])):
            if sc_full[index] > 1.4 or ce_full[index] > 0.005:
                img = Image.open(os.path.join(basedir, 'oads_arw', 'tiff', filenames[index]))
                images.append(img)
                titles.append(f'sc: {sc_full[index]:.4f}\nce: {ce_full[index]:.4f}')
                tups.append((sc_full[index], ce_full[index], os.path.join(basedir, 'oads_arw', 'tiff', filenames[index])))

    fig = plot_images(images=images, titles=titles)
    fig.tight_layout()
    figs.append(fig)


    fig, ax = plt.subplots(1,1, figsize=(25,25))
    imscatter_all(images=[x[2] for x in tups], xs=[x[0] for x in tups], ys=[x[1] for x in tups], xlabel='SC', ylabel='CE', ax=ax, zoom=0.01)
    figs.append(fig)


    fig, ax = plt.subplots(1,1,figsize=(10,5))
    img = np.array(images[0])
    img[-224:,-224:,:] = 255
    ax.imshow(img)
    ax.axis('off')
    # ax.imshow(np.ones((224,224)))
    figs.append(fig)

    fig.savefig('/home/niklas/projects/oads_access/results/oads_resolution_demo.png')
    exit(1)
    # #############
    jpeg_comp_rate_crops = (np.array(crop_jpeg_factors)[:,:,1] / np.array(crop_jpeg_factors)[:,:,0])#.flatten()

    XX,YY = np.meshgrid(np.arange(jpeg_comp_rate_crops.shape[1]),np.arange(jpeg_comp_rate_crops.shape[0]))
    crop_index_array = np.vstack((XX.ravel(), YY.ravel())).T
    jpeg_comp_rate_crops = jpeg_comp_rate_crops.flatten()

    print(f'crop_index_array: {crop_index_array.shape}')
    print(f'jpeg_comp_rate_crops: {jpeg_comp_rate_crops.shape}')

    fig, ax = plt.subplots(1,3, figsize=(30,15))

    
    factors_crops = jpeg_comp_rate_crops
    lower = np.percentile(factors_crops, q=1)
    upper = np.percentile(factors_crops, q=99)
    n_bins = 40
    count, bins = np.histogram(factors_crops[(factors_crops > lower) & (factors_crops < upper)], bins=n_bins)
    indices, values = get_bin_values(data=factors_crops, bins=bins, min_count=count.min())

    x_min, x_max = np.nanmin(sc_crops), np.nanmax(sc_crops)
    y_min, y_max = np.nanmin(ce_crops), np.nanmax(ce_crops)

    ax[0].scatter(sc_crops, ce_crops, c=factors_crops)
    ax[0].set_xlim((x_min, x_max))
    ax[0].set_ylim((y_min, y_max))
    ax[0].set_xlabel('SC')
    ax[0].set_ylabel('CE')
    ax[0].set_title('All crops')

    indices = indices.flatten()
    ax[1].scatter(np.array(sc_crops)[indices], np.array(ce_crops)[indices], c=factors_crops[indices])
    ax[1].set_xlim((x_min, x_max))
    ax[1].set_ylim((y_min, y_max))
    ax[1].set_xlabel('SC')
    ax[1].set_ylabel('CE')
    ax[1].set_title(f'Uniform JPEG Distribution: {count.min()} number of points per bin -> {count.min() * n_bins} ({len(np.array(sc_crops)[indices])}) points')

    print(indices.shape)
    np.random.shuffle(indices)
    indices = indices[:995]

    chosen_ce_crops = np.array(ce_crops)[indices]
    chosen_sc_crops = np.array(sc_crops)[indices]
    chosen_ce_crops_peri = np.array(ce_peri_crops)[indices]
    chosen_sc_crops_peri = np.array(sc_peri_crops)[indices]
    chosen_factors_crops = factors_crops[indices]

    ax[2].scatter(chosen_sc_crops, chosen_ce_crops, c=chosen_factors_crops)
    ax[2].set_xlim((x_min, x_max))
    ax[2].set_ylim((y_min, y_max))
    ax[2].set_xlabel('SC')
    ax[2].set_ylabel('CE')
    ax[2].set_title(f'Uniform JPEG Distribution: {len(indices)} points')

    figs.append(fig)

    ##########
    fig = joint_plot(xs=[sc_full, sc_crops], ys=[ce_full, ce_crops], cs=[factors, factors_crops], figsize=(25,20), axtitles=['Full Images', 'Crops'], xlabels=['SC', 'SC'], ylabels=['CE', 'CE'])
    figs.append(fig)
    ##########

    ce_corr = np.corrcoef(chosen_ce_crops, chosen_ce_crops_peri)[1,0]
    sc_corr = np.corrcoef(chosen_sc_crops, chosen_sc_crops_peri)[1,0]
    print(f'CE Center/Peri correlation: {ce_corr}\nSC Center/Peri correlation: {sc_corr}')


    ##########
    images = []
    titles = []
    tups = []
    for index in range(len(sc_crops)):
        if os.path.exists(os.path.join(basedir, 'oads_arw', 'tiff', filenames[crop_index_array[index][0]])):
            if sc_crops[index] > 1.6 or ce_crops[index] > 0.009:
                img = Image.open(os.path.join(basedir, 'oads_arw', 'tiff', filenames[crop_index_array[index][0]]))
                crop = img.crop(crop_boxes[crop_index_array[index][1]][crop_index_array[index][0]])
                images.append(crop)
                titles.append(f'sc: {sc_crops[index]:.4f}\nce: {ce_crops[index]:.4f}')
                tups.append((sc_crops[index], ce_crops[index], crop))

    fig = plot_images(images=images, titles=titles, max_images=50)
    fig.tight_layout()
    figs.append(fig)

    fig, ax = plt.subplots(1,1, figsize=(25,25))
    imscatter_all(images=[x[2] for x in tups], xs=[x[0] for x in tups], ys=[x[1] for x in tups], xlabel='SC', ylabel='CE', ax=ax, zoom=0.007)
    figs.append(fig)



   

    


    ##########
    # nan images
    # nan_images = []
    # print(f'len(sc_crops): {len(sc_crops)}')
    # for i in range(len(sc_crops)):
    #     if np.isnan(sc_crops[i]) or np.isnan(ce_crops[i]):
    #         nan_images.append(Image.open(os.path.join(basedir, filenames[crop_index_array[i][0]])))

    # print(f'Nan images: {nan_images}')

    # fig = plot_images(nan_images)

    # #############
    # # Create the imscatter based on the 
    # #############

    # # beta = np.array([np.mean(_x) for _x in beta_results])
    # # gamma = np.array([np.mean(_x) for _x in gamma_results])

    # ce_lower = np.percentile(ce, q=5)
    # ce_upper = np.percentile(ce, q=95)
    # sc_lower = np.percentile(sc, q=5)
    # sc_upper = np.percentile(sc, q=95)

    # ####
    # count_ce, division_ce = np.histogram(ce[(ce >= ce_lower) & (ce <= ce_upper)], bins=5)

    # min_count_ce = count_ce.min()
    # min_value_ce = division_ce.min()
    # max_value_ce = division_ce.max()

    # print(min_count_ce, min_value_ce, max_value_ce)

    # count_sc, division_sc = np.histogram(sc[(sc >= sc_lower) & (sc <= sc_upper)], bins=5)

    # min_count_sc = count_sc.min()
    # min_value_sc = division_sc.min()
    # max_value_sc = division_sc.max()

    # print(min_count_sc, min_value_sc, max_value_sc)

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


    # #####
    # # Create a uniform distribution for CE and SC individuall and then select the indices that occur in both -> leads to a non-uniform distribution afterwards
    # fig, ax = plt.subplots(2,2, figsize=(10,5))

    # indices_ce, values_ce = get_bin_values(data=ce, bins=division_ce, min_count=min_count_ce)
    # ax[0, 0].hist(ce, bins=division_ce, label='CE all')
    # ax[0, 0].hist(values_ce.flatten(), bins=division_ce, label='CE Uniform Selection')
    # ax[0, 0].set_xlabel('CE')
    # ax[0, 0].legend()

    # indices_sc, values_sc = get_bin_values(data=sc, bins=division_sc, min_count=min_count_sc)
    # ax[0, 1].hist(sc, bins=division_sc, label='SC all')
    # ax[0, 1].hist(values_sc.flatten(), bins=division_sc, label='SC Uniform Selection')
    # ax[0, 1].set_xlabel('SC')
    # ax[0, 1].legend()


    # ## Plot the indices
    # index_overlap = np.intersect1d(indices_ce.flatten(), indices_sc.flatten())
    # print(index_overlap)
    # ax[1,0].bar(x=indices_ce.flatten(), height=1, label='CE')
    # ax[1,0].bar(x=index_overlap, height=0.5, label='Overlap CE+SC')
    # ax[1,0].set_title('CE Uniform Selection Images')
    # ax[1,0].set_xlabel('Index')
    # ax[1,0].set_yticklabels([])
    # ax[1,1].bar(x=indices_sc.flatten(), height=1, label='SC')
    # ax[1,1].bar(x=index_overlap, height=0.5, label='Overlap CE+SC')
    # ax[1,1].set_title('SC Uniform Selection Images')
    # ax[1,1].set_xlabel('Index')
    # ax[1,1].set_yticklabels([])

    # ax[1,0].legend()
    # ax[1,1].legend()

    # plt.tight_layout()
    # # plt.show()

    # figs.append(fig)


    # ##########
    # ## print image to use
    # fig, ax = plt.subplots(1, len(index_overlap), figsize=(15,5))

    # im_fig, (im_ax, im_ax_all) = plt.subplots(1,2, figsize=(15,10))

    # #
    # im_fig = plt.figure(figsize=(25,15))
    # gs = GridSpec(4, 8)

    # im_ax = im_fig.add_subplot(gs[1:4, 4:7])
    # ax_hist_sc = im_fig.add_subplot(gs[1:4, 7])
    # ax_hist_sc.hist(sc[index_overlap], bins=division_sc, label='SC Uniform', orientation = 'horizontal')

    # ax_hist_ce = im_fig.add_subplot(gs[0, 4:7])
    # ax_hist_ce.hist(ce[index_overlap], bins=division_ce, label='CE Uniform')

    # im_ax_all = im_fig.add_subplot(gs[1:4, 0:3])
    # ax_hist_sc_all = im_fig.add_subplot(gs[1:4,3])
    # ax_hist_sc_all.hist(sc, bins=division_sc, label='SC all', orientation = 'horizontal')

    # ax_hist_ce_all = im_fig.add_subplot(gs[0, 0:3])
    # ax_hist_ce_all.hist(ce, bins=division_ce, label='CE all')
    # ##

    # crops = []
    # for index in index_overlap:
    #     image_index = int(index%len(image_names))
    #     img = Image.open(os.path.join(basedir, 'oads_arw', 'tiff', image_names[image_index]))
    #     crop = img.crop(crop_boxes[image_index][int(index%number_crops_per_image)])
    #     crops.append(crop)

    #     crop = crop.reduce(4)

    #     imscatter(ce[index], sc[index], np.array(crop), zoom=0.1, ax=im_ax)
    #     im_ax.scatter(ce[index], sc[index])

    # fig = plot_images(crops)
    # figs.append(fig)

    # im_ax.set_title('Crops by CE and SC - Uniform Selection')
    # im_ax.set_xlabel('CE')
    # im_ax.set_ylabel('SC')



    # for image_index, image_name in enumerate(image_names):
    #     for crop_index in range(number_crops_per_image):
    #         img = Image.open(os.path.join(basedir, 'oads_arw', 'tiff', image_names[image_index]))
    #         crop = img.crop(crop_boxes[image_index][crop_index])
    #         crop = crop.reduce(4)

    #         imscatter(ce[image_index*crop_index], sc[image_index*crop_index], np.array(crop), zoom=0.1, ax=im_ax_all)
    #         im_ax_all.scatter(ce[image_index*crop_index], sc[image_index*crop_index])

    # im_ax_all.set_title("Crops by CE and SC - All Crops")
    # im_ax_all.set_xlabel('CE')
    # im_ax_all.set_ylabel('SC')
    # figs.append(im_fig)

    # im_fig.tight_layout()
    # ###########


    result_manager.save_pdf(figs=figs, filename='oads_crop_statistics.pdf')