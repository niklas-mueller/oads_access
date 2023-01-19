import time
from oads_access.oads_access import OADS_Access
from result_manager.result_manager import ResultManager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image, TiffImagePlugin
import os
import sys
sys.path.append('../..')
from mouse_lgn.lgn_statistics import load_lgn_statistics
# from pytorch_utils.pytorch_utils import ToJpeg, ToOpponentChannel

basedir = '/home/niklas/projects/data/oads'

# oads = OADS_Access(basedir=basedir, file_formats='.ARW', n_processes=16)
result_manager = ResultManager(root='/home/niklas/projects/oads_access/analysis/results')

# print(oads.image_names['92823b8c86c343c3'])
image_names = os.listdir(os.path.join(basedir, 'oads_arw', 'tiff'))

# use_mean = False

means = []
stds = []
ces = []
scs = []
betas = []
gammas = []

means_center = []
stds_center = []
scs_center = []
ces_center = []
betas_center = []
gammas_center = []

means_periphery = []
stds_periphery = []
ces_periphery = []
scs_periphery = []
betas_periphery = []
gammas_periphery = []

ce, sc, beta, gamma, filenames = load_lgn_statistics(
    filepath='/home/niklas/projects/oads_access/analysis/LGNstatistics_OADS_test.mat', drop_colors=False)

def get_statistics(image, index=None, use_mean=False):
    # statistic_1 = 0
    # statistic_2 = 0

    # if use_mean:
    mean = image.mean()
    std = image.std()

    # else:
    _ce = ce[index][0] # second index (0) is color channel
    _sc = sc[index][0] # second index (0) is color channel
    _beta = beta[index][0] # second index (0) is color channel
    _gamma = gamma[index][0] # second index (0) is color channel

    return mean, std, _ce, _sc, _beta, _gamma


for image_index, image_name in enumerate(image_names):
    # img, obj = oads.load_image(image_name.split('.')[0])
    img = TiffImagePlugin.TiffImageFile(fp=os.path.join(basedir, 'oads_arw', 'tiff', image_name))

    # print(img.mean(), img.std())
    statistic_full = get_statistics(image=np.array(img), index=image_index)
    means.append(statistic_full[0])
    stds.append(statistic_full[1])
    ces.append(statistic_full[2])
    scs.append(statistic_full[3])
    betas.append(statistic_full[4])
    gammas.append(statistic_full[5])
    
    # Get a center and periphery estimate
    (height, width, _) = np.array(img).shape
    center = img.crop((width * 0.25, height*0.25, width * 0.75, height * 0.75))
    statistics_center = get_statistics(image=np.array(center))
    means_center.append(statistics_center[0])
    stds_center.append(statistics_center[1])
    ces_center.append(statistics_center[2])
    scs_center.append(statistics_center[3])
    betas_center.append(statistics_center[4])
    gammas_center.append(statistics_center[5])



fig, (ax, ax0) = plt.subplots(2,3, figsize=(20,10))

ax[0].scatter(means, stds, marker='.', color='b', label='Mean/Std Full')
ax[0].scatter(means_center, stds_center, marker='.', color='g', label='Mean/Std Center')
ax[0].set_xlabel('Mean')
ax[0].set_ylabel('Std')

ax[1].scatter(ces, scs, marker='+', color='b', label='CE/SC Full')
ax[1].scatter(ces_center, scs_center, marker='+', color='g', label='CE/SC Center')
ax[1].set_xlabel('CE')
ax[1].set_ylabel('SC')

ax[2].scatter(betas, gammas, marker='+', color='b', label='Beta/Gamma Full')
ax[2].scatter(betas_center, gammas_center, marker='+', color='g', label='Beta/Gamma Center')
ax[2].set_xlabel('Beta')
ax[2].set_ylabel('Gamma')

# Plot histogram of these
cmin = 0
cmax = len(image_names)

ax0[0].hist2d(x=means, y=stds, cmin=cmin, cmax=cmax)
ax0[0].set_xlabel('Mean')
ax0[0].set_ylabel('Std')
ax0[0].set_title('Full')

ax0[1].hist2d(x=ces, y=scs, cmin=cmin, cmax=cmax)
ax0[1].set_xlabel('CE')
ax0[1].set_ylabel('SC')
ax0[1].set_title('Full')

bar = ax0[2].hist2d(x=betas, y=gammas, cmin=cmin, cmax=cmax)
ax0[2].set_xlabel('Beta')
ax0[2].set_ylabel('Gamma')
ax0[2].set_title('Full')

# plt.colorbar(bar[3], cax=ax0[2])
# ax.scatter(means, stds, marker='.', color='b', label='Mean Full')
# ax.scatter(means_center, stds_center, marker='+', color='b', label='Mean Center')
# if use_mean:
#     ax.set_xlabel('Means')
#     ax.set_ylabel('Stds')
# else:
#     ax.set_xlabel('CE')
#     ax.set_ylabel('SC')

plt.legend()

result_manager.save_pdf(figs=[fig], filename='image_statistics_plot.pdf')
