"""
==============
Blob Detection
==============

Blobs are bright on dark or dark on bright regions in an image. In
this example, blobs are detected using 3 algorithms. The image used
in this case is the Hubble eXtreme Deep Field. Each bright dot in the
image is a star or a galaxy.

Laplacian of Gaussian (LoG)
-----------------------------
This is the most accurate and slowest approach. It computes the Laplacian
of Gaussian images with successively increasing standard deviation and
stacks them up in a cube. Blobs are local maximas in this cube. Detecting
larger blobs is especially slower because of larger kernel sizes during
convolution. Only bright blobs on dark backgrounds are detected. See
:py:meth:`skimage.feature.blob_log` for usage.

Difference of Gaussian (DoG)
----------------------------
This is a faster approximation of LoG approach. In this case the image is
blurred with increasing standard deviations and the difference between
two successively blurred images are stacked up in a cube. This method
suffers from the same disadvantage as LoG approach for detecting larger
blobs. Blobs are again assumed to be bright on dark. See
:py:meth:`skimage.feature.blob_dog` for usage.

Determinant of Hessian (DoH)
----------------------------
This is the fastest approach. It detects blobs by finding maximas in the
matrix of the Determinant of Hessian of the image. The detection speed is
independent of the size of blobs as internally the implementation uses
box filters instead of convolutions. Bright on dark as well as dark on
bright blobs are detected. The downside is that small blobs (<3px) are not
detected accurately. See :py:meth:`skimage.feature.blob_doh` for usage.
"""

from math import sqrt
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray

import sys
import Queue
import cv2
import numpy
import matplotlib.pyplot as plt

max_sigma_p=0
if len(sys.argv) < 2 or len(sys.argv) > 3:
    sys.exit(1)
if len(sys.argv) == 2:
    max_sigma_p=30
#image = data.hubble_deep_field()[0:500, 0:500]
filename = sys.argv[1]
max_sigma_p= int(sys.argv[2])
image = cv2.imread(filename + ".png")
#image = cv2.imread("DNA.bmp")

sp = image.shape
width = sp[1]
height = sp[0]

image_gray = rgb2gray(image)

#cv2.imshow("image_gray", image_gray)
cv2.imwrite("image_gray.png", image_gray)

#blobs_log = blob_log(image_gray, max_sigma=30, num_sigma=20, threshold=.2)
blobs_log = blob_log(image_gray, max_sigma=max_sigma_p, num_sigma=10, threshold=.2)

# Compute radii in the 3rd column.
blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)


q = Queue.Queue()
background = image[4, 4]
for blob_log in blobs_log:
    y, x, r = blob_log
    #cv2.circle(image, (int(x), int(y)), int(r), (255, 255, 0), 2, -1)

    q.put([int(y), int(x)]) 
    while (not q.empty()):
        temp_x, temp_y = q.get()
        r_, g_, b_ = image[temp_x, temp_y] 
        image[temp_x, temp_y] = [0, 0, 0]
        direction = [[-1, 0], [0, -1], [-1, -1], [0, 1], [1, 0], [1, 1], [1, -1], [-1, 1]]
        for neighbor in direction:
            neighbor_x = temp_x + neighbor[0]
            neighbor_y = temp_y + neighbor[1]
            #if neighbor_x > 0 and neighbor_x < height and neighbor_y > 0 and neighbor_y < width:
            if neighbor_x > 0 and neighbor_x < width and neighbor_y > 0 and neighbor_y < height:
                _r, _g, _b = image[neighbor_x, neighbor_y]
                if _r != 0 and _g != 0 and _b != 0:
                    if abs(int(_r - r_)) < 30 and abs(int(_g - g_)) < 30 and abs(int(_b - b_)) < 30: 
                        q.put([neighbor_x, neighbor_y])

#cv2.circle(image, (100, 100), int(100), (255, 255, 0), 2, -1)
#q.put((2, 3))
#top = q.get()
#print top[0], top[1]
#cv2.imwrite("DNA.png", image)
cv2.imwrite("image_" + filename +  ".png", image)
#cv2.imshow("image", image)
#cv2.waitKey(10)

#blobs_dog = blob_dog(image_gray, max_sigma=30, threshold=.1)
#blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)

#blobs_doh = blob_doh(image_gray, max_sigma=30, threshold=.01)

#blobs_list = [blobs_log, blobs_dog, blobs_doh]
#colors = ['yellow', 'lime', 'red']
#titles = ['Laplacian of Gaussian', 'Difference of Gaussian',
          #'Determinant of Hessian']
#sequence = zip(blobs_list, colors, titles)

#fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True,
                         #subplot_kw={'adjustable': 'box-forced'})
#ax = axes.ravel()

#for idx, (blobs, color, title) in enumerate(sequence):
    #ax[idx].set_title(title)
    #ax[idx].imshow(image, interpolation='nearest')
    #for blob in blobs:
        #y, x, r = blob
        #c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
        #ax[idx].add_patch(c)
    #ax[idx].set_axis_off()

#plt.tight_layout()
#plt.show()
