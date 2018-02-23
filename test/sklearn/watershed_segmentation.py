import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from skimage.morphology import watershed
from skimage.feature import peak_local_max
import skimage.io


# Generate an initial image with two overlapping circles
# x, y = np.indices((80, 80))
# x1, y1, x2, y2,x3,y3 = 28, 28, 44, 52,20,68
# r1, r2,r3 = 16, 20,13
# mask_circle1 = (x - x1)**2 + (y - y1)**2 < r1**2
# mask_circle2 = (x - x2)**2 + (y - y2)**2 < r2**2
# mask_circle3 = (x - x3)**2 + (y - y3)**2 < r3**2
# image = np.logical_or(mask_circle1, mask_circle3)
# image = np.logical_or(image, mask_circle2)

# Load a single image and its associated masks
id = '6fe2df6de1d962b90146c822bcefc84d0d3d6926fdfbacd3acdc9de830ee5622'
file = "/root/dataset/dataScienceBowl2018/stage1_train/{}/images/{}.png".format(id,id)
mfile = "/root/dataset/dataScienceBowl2018/stage1_train/{}/masks/*.png".format(id)
image = skimage.io.imread(file)
masks = skimage.io.imread_collection(mfile).concatenate()
height, width, _ = image.shape
num_masks = masks.shape[0]

# Make a ground truth array and summary label image
y_true = np.zeros((num_masks, height, width), np.uint16)
y_true[:,:,:] = masks[:,:,:] // 255  # Change ground truth mask to zeros and ones

labels = np.zeros((height, width), np.uint16)
labels[:,:] = np.sum(y_true, axis=0)  # Add up to plot all masks
image = labels

# Now we want to separate the two objects in image
# Generate the markers as local maxima of the distance to the background
distance = ndi.distance_transform_edt(image)
local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)),
                            labels=image)
markers = ndi.label(local_maxi)[0]
labels = watershed(-distance, markers, mask=image)

fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True,
                         subplot_kw={'adjustable': 'box-forced'})
ax = axes.ravel()

ax[0].imshow(image, cmap=plt.cm.gray, interpolation='nearest')
ax[0].set_title('Overlapping objects')
ax[1].imshow(-distance, cmap=plt.cm.gray, interpolation='nearest')
ax[1].set_title('Distances')
ax[2].imshow(labels, cmap=plt.cm.spectral, interpolation='nearest')
ax[2].set_title('Separated objects')

for a in ax:
    a.set_axis_off()

fig.tight_layout()
plt.show()
