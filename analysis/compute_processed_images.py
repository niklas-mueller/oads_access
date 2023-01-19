import time
from oads_access.oads_access import OADS_Access
from result_manager.result_manager import ResultManager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from pytorch_utils.pytorch_utils import ToJpeg, ToOpponentChannel

basedir = '/home/niklas/projects/data/oads'

oads = OADS_Access(basedir=basedir, file_formats='.ARW', n_processes=16)
result_manager = ResultManager(root='/home/niklas/projects/oads_access/analysis/results')

# print(oads.image_names['92823b8c86c343c3'])
img, obj = oads.load_image('0a0e1801856d4ded')

figs = []

fig, (ax, ax0) = plt.subplots(1, 2)
ax.imshow(img)
ax.axis('off')

ax0.imshow(ToJpeg()(img))
ax0.axis('off')
ax0.set_title('JPEG')

figs.append(fig)



##############

crop_box = (0,0,500,500)

fig, (ax, ax0, ax1) = plt.subplots(1,3)
_x = Image.fromarray(img).crop(crop_box)
ax.imshow(_x)
ax.set_xlabel(np.array(_x).shape)
# ax.axis('off')
ax.set_title('Crop')
figs.append(fig)

# fig, ax = plt.subplots()
_x = ToJpeg()(Image.fromarray(img).crop(crop_box))
ax0.imshow(_x)
ax0.set_xlabel(np.array(_x).shape)
# ax0.axis('off')
ax0.set_title("Crop -> JPEG")
# figs.append(fig)

_x = ToJpeg()(np.array(Image.fromarray(img).crop(crop_box)))
ax1.imshow(_x)
ax1.set_xlabel(np.array(_x).shape)
# ax1.axis('off')
ax1.set_title("JPEG -> Crop")


fig, (ax, ax0, ax1) = plt.subplots(3,3)
coc = ToOpponentChannel()(Image.fromarray(img).crop(crop_box))
# print(coc.shape)
########
_x = Image.fromarray(coc[:,:,0])
ax[0].imshow(_x, cmap='gray')
ax[0].set_xlabel(np.array(_x).shape)

# ax[0].axis('off')
ax[0].set_title("Crop -> COC")

_x = ToJpeg()(Image.fromarray(np.array(coc[:,:,0])))
ax[1].imshow(_x, cmap='gray')
ax[1].set_xlabel(np.array(_x).shape)

# ax[1].axis('off')
ax[1].set_title("Crop -> COC -> JPEG")

_x = ToOpponentChannel()(Image.fromarray(ToJpeg()(img)).crop(crop_box))[:,:,0]
ax[2].imshow(_x, cmap="gray")
ax[2].set_xlabel(np.array(_x).shape)

# ax[2].axis('off')
ax[2].set_title("JPEG -> CROP -> COC")
# ########
_x = Image.fromarray(coc[:,:,1])
ax0[0].imshow(_x, cmap='gray')
ax0[0].set_xlabel(np.array(_x).shape)

# ax0[0].axis('off')
ax0[0].set_title("Crop -> COC")

_x = ToJpeg()(Image.fromarray(np.array(coc[:,:,1])))
ax0[1].imshow(_x, cmap='gray')
ax0[1].set_xlabel(np.array(_x).shape)

# ax0[1].axis('off')
ax0[1].set_title("Crop -> COC -> JPEG")

_x = ToOpponentChannel()(Image.fromarray(ToJpeg()(img)).crop(crop_box))[:,:,1]
ax0[2].imshow(_x, cmap="gray")
ax0[2].set_xlabel(np.array(_x).shape)

# ax0[2].axis('off')
ax0[2].set_title("JPEG -> CROP -> COC")
# ########
_x = Image.fromarray(coc[:,:,2])
ax1[0].imshow(_x, cmap='gray')
ax1[0].set_xlabel(np.array(_x).shape)

# ax1[0].axis('off')
ax1[0].set_title("Crop -> COC")

_x = ToJpeg()(Image.fromarray(np.array(coc[:,:,2])))
ax1[1].imshow(_x, cmap='gray')
ax1[1].set_xlabel(np.array(_x).shape)

# ax1[1].axis('off')
ax1[1].set_title("Crop -> COC -> JPEG")

_x = ToOpponentChannel()(Image.fromarray(ToJpeg()(img)).crop(crop_box))[:,:,2]
ax1[2].imshow(_x, cmap="gray")
ax1[2].set_xlabel(np.array(_x).shape)

# ax1[2].axis('off')
ax1[2].set_title("JPEG -> CROP -> COC")
########

figs.append(fig)


fig, (ax, ax0, ax1) = plt.subplots(3,3)

coc = ToOpponentChannel()(img)
########
_x = Image.fromarray(img[:,:,0]).crop(crop_box)
ax[0].imshow(_x, cmap="gray")
ax[0].set_xlabel(np.array(_x).shape)

# ax[0].axis('off')
ax[0].set_title("COC -> Crop")

_x = ToJpeg()(Image.fromarray(img[:,:,0]).crop(crop_box))
ax[1].imshow(_x, cmap="gray")
ax[1].set_xlabel(np.array(_x).shape)

# ax[1].axis('off')
ax[1].set_title("COC -> Crop -> JPEG")

_x = Image.fromarray(ToJpeg()(Image.fromarray(img[:,:,0]))).crop(crop_box)
ax[2].imshow(_x, cmap="gray")
ax[2].set_xlabel(np.array(_x).shape)

# ax[2].axis('off')
ax[2].set_title("COC -> JPEG -> Crop")

########
_x = Image.fromarray(img[:,:,1]).crop(crop_box)
ax0[0].imshow(_x, cmap="gray")
ax0[0].set_xlabel(np.array(_x).shape)

# ax0[0].axis('off')

_x = ToJpeg()(Image.fromarray(img[:,:,1]).crop(crop_box))
ax0[1].imshow(_x, cmap="gray")
ax0[1].set_xlabel(np.array(_x).shape)

# ax0[1].axis('off')
# ax0[1].set_title("COC -> Crop")

_x = Image.fromarray(ToJpeg()(Image.fromarray(img[:,:,1]))).crop(crop_box)
ax0[2].imshow(_x, cmap="gray")
ax0[2].set_xlabel(np.array(_x).shape)

# ax0[2].axis('off')
# ax0[2].set_title("COC -> Crop")

########
_x = Image.fromarray(img[:,:,2]).crop(crop_box)
ax1[0].imshow(_x, cmap="gray")
ax1[0].set_xlabel(np.array(_x).shape)

# ax1[0].axis('off')

_x = ToJpeg()(Image.fromarray(img[:,:,2]).crop(crop_box))
ax1[1].imshow(_x, cmap="gray")
ax1[1].set_xlabel(np.array(_x).shape)

# ax1[1].axis('off')
# ax1[1].set_title("COC -> Crop")

_x = Image.fromarray(ToJpeg()(Image.fromarray(img[:,:,2]))).crop(crop_box)
ax1[2].imshow(_x, cmap="gray")
ax1[2].set_xlabel(np.array(_x).shape)

# ax1[2].axis('off')
# ax1[2].set_title("COC -> Crop")

########


figs.append(fig)


result_manager.save_pdf(figs=figs, filename='example_image.pdf')#, dpi=None)

# def get_image_intensity(tup):
#     (img, obj) = tup
#     pass

# start = time.time()
# distribution = oads.apply_per_image(get_image_intensity)
# end = time.time()

# print(f'time taken: {end-start}')

# print(distribution)