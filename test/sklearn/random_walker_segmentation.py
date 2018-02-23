import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from skimage.morphology import watershed
from skimage.feature import peak_local_max
import skimage.io
import numpy as np
import matplotlib.pyplot as plt

from skimage.segmentation import random_walker
from skimage.data import binary_blobs
from skimage.exposure import rescale_intensity
import skimage


# # Generate an initial image with two overlapping circles
x, y = np.indices((80, 80))
x1, y1, x2, y2,x3,y3 = 28, 28, 44, 52,20,68
r1, r2,r3 = 16, 20,13
mask_circle1 = (x - x1)**2 + (y - y1)**2 < r1**2
mask_circle2 = (x - x2)**2 + (y - y2)**2 < r2**2
mask_circle3 = (x - x3)**2 + (y - y3)**2 < r3**2
image = np.logical_or(mask_circle1, mask_circle3)
image = np.logical_or(image, mask_circle2)

# Load a single image and its associated masks
# id = '0a7d30b252359a10fd298b638b90cb9ada3acced4e0c0e5a3692013f432ee4e9'
# file = "/root/dataset/dataScienceBowl2018/stage1_train/{}/images/{}.png".format(id,id)
# mfile = "/root/dataset/dataScienceBowl2018/stage1_train/{}/masks/*.png".format(id)
# image = skimage.io.imread(file)
# masks = skimage.io.imread_collection(mfile).concatenate()
# height, width, _ = image.shape
# num_masks = masks.shape[0]
#
# # Make a ground truth array and summary label image
# y_true = np.zeros((num_masks, height, width), np.uint16)
# y_true[:,:,:] = masks[:,:,:] // 255  # Change ground truth mask to zeros and ones
#
# labels = np.zeros((height, width), np.uint16)
# labels[:,:] = np.sum(y_true, axis=0)  # Add up to plot all masks
#
# image = labels

# Generate noisy synthetic data
# data = skimage.img_as_float(binary_blobs(length=128, seed=1))
# sigma = 0.35
# data += np.random.normal(loc=0, scale=sigma, size=data.shape)
# data = rescale_intensity(data, in_range=(-sigma, 1 + sigma),
#                          out_range=(-1, 1))
data = image
# The range of the binary image spans over (-1, 1).
# We choose the hottest and the coldest pixels as markers.
markers = np.zeros(data.shape, dtype=np.uint)
markers[data < -0.95] = 1
markers[data > 0.95] = 2

# Run random walker algorithm
labels = random_walker(data, markers, beta=10, mode='bf')

# Plot results
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 3.2),
                                    sharex=True, sharey=True)
ax1.imshow(data, cmap='gray', interpolation='nearest')
ax1.axis('off')
ax1.set_adjustable('box-forced')
ax1.set_title('Noisy data')
ax2.imshow(markers, cmap='magma', interpolation='nearest')
ax2.axis('off')
ax2.set_adjustable('box-forced')
ax2.set_title('Markers')
ax3.imshow(labels, cmap='gray', interpolation='nearest')
ax3.axis('off')
ax3.set_adjustable('box-forced')
ax3.set_title('Segmentation')

fig.tight_layout()
plt.show()