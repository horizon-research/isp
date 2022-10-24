import rawpy
import imageio
import numpy as np
import sys
from PIL import Image
from scipy import signal
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear

#a3 = np.array([[np.arange(0,3,1), np.arange(3,6,1), np.arange(6,9,1), np.arange(9,12,1)],
#               [np.arange(100,103,1), np.arange(103,106,1), np.arange(106,109,1), np.arange(109,112,1)],
#               [np.arange(200,203,1), np.arange(203,206,1), np.arange(206,209,1), np.arange(209,212,1)],
#               [np.arange(300,303,1), np.arange(303,306,1), np.arange(306,309,1), np.arange(309,312,1)],
#               [np.arange(400,403,1), np.arange(403,406,1), np.arange(406,409,1), np.arange(409,412,1)]])
#print(a3.shape)
#print(a3)
#print(a3[:,:,0].flatten().shape)
#tmp=np.stack((a3[:,:,0].flatten(), a3[:,:,1].flatten(), a3[:,:,2].flatten()))
#print(tmp.shape)
#tt = np.stack((tmp[0].reshape(5, 4), tmp[1].reshape(5, 4), tmp[2].reshape(5, 4)), axis=2)
#print(tt)
#print(tt.shape)
#sys.exit()

np.set_printoptions(threshold=sys.maxsize)

image = sys.argv[1]
raw = rawpy.imread(image)

print(raw.color_matrix)
print(raw.color_desc)
#print(raw.rgb_xyz_matrix)
print(raw.raw_type)
#print(raw.white_level)
print(raw.raw_pattern)
#print(raw.num_colors)
print(raw.camera_whitebalance)
print(raw.black_level_per_channel)
#print(raw.camera_white_level_per_channel)
#sys.exit()

#try:
#    thumb = raw.extract_thumb()
#except rawpy.LibRawNoThumbnailError:
#  print('no thumbnail found')
#except rawpy.LibRawUnsupportedThumbnailError:
#  print('unsupported thumbnail')
#else:
#  if thumb.format == rawpy.ThumbFormat.JPEG:
#    with open('thumb.jpg', 'wb') as f:
#      f.write(thumb.data)
#  elif thumb.format == rawpy.ThumbFormat.BITMAP:
#    imageio.imsave('thumb.tiff', thumb.data)
#sys.exit()

# raw_img is a np array
raw_img = raw.raw_image
ishape = raw_img.shape
print(ishape)
raw_color_index = raw.raw_colors

# subtract black level
black = np.reshape(raw.black_level_per_channel, (2, 2))
black = np.tile(black, (ishape[0]//2, ishape[1]//2))
gs_img = (raw_img - black) / (raw.white_level - black)
#sys.exit()
#Image.fromarray((gs_img * 256).astype(np.uint8)).save("gs.png")

# apply wb gains; could either do here or multiply the wb matrix after demosaic; equivalent for bilinear filtering
#gs_img = np.where(raw_color_index == 0, gs_img * raw.camera_whitebalance[0], gs_img)
#gs_img = np.where(raw_color_index == 1, gs_img * raw.camera_whitebalance[1], gs_img)
#gs_img = np.where(raw_color_index == 3, gs_img * raw.camera_whitebalance[1], gs_img)
#gs_img = np.where(raw_color_index == 2, gs_img * raw.camera_whitebalance[2], gs_img)

## create bayer-domain raw image that can be displayed as RGB image
##https://stackoverflow.com/questions/19766757/replacing-numpy-elements-if-condition-is-met
#r_channel = np.where(raw_color_index == 0, gs_img, 0)
#g_channel = np.where(raw_color_index == 1, gs_img, 0)
#g_channel = np.where(raw_color_index == 3, gs_img, g_channel)
#b_channel = np.where(raw_color_index == 2, gs_img, 0)
#
##https://hausetutorials.netlify.app/posts/2019-12-20-numpy-reshape/
#color_img = np.stack((r_channel, g_channel, b_channel)).transpose((1, 2, 0))
#color_img = (color_img * 256).astype(np.uint8)
#Image.fromarray(color_img, 'RGB').save("bayer.png")

# demosaic
# https://colour-demosaicing.readthedocs.io/en/latest/generated/colour_demosaicing.demosaicing_CFA_Bayer_bilinear.html#colour_demosaicing.demosaicing_CFA_Bayer_bilinear
demosaic_img = demosaicing_CFA_Bayer_bilinear(gs_img, 'RGGB') #iphone
#demosaic_img = demosaicing_CFA_Bayer_bilinear(gs_img, 'BGGR') #pixel
#color_img = (demosaic_img * 256).astype(np.uint8)
#Image.fromarray(color_img, 'RGB').save("demosaic.png")
#sys.exit()

# wb and cc
wb_mat = np.array([[raw.camera_whitebalance[0], 0, 0], [0, raw.camera_whitebalance[1], 0], [0, 0, raw.camera_whitebalance[2]]])
# From photo
cc_mat = raw.color_matrix[0:3, 0:3]
# Tree
#cc_mat = np.array([[1.6640625, -0.6796875, 0.0078125], [-0.1484375, 1.3046875, -0.1484375], [0.109375, -0.7421875, 1.640625]])
# Person
#cc_mat = np.array([[1.6328125, -0.6171875, -0.015625], [-0.171875, 1.3828125, -0.2109375], [0.125, -0.8046875, 1.6796875]])

#https://www.pythoninformer.com/python-libraries/numpy/index-and-slice/
flat_img = np.stack((demosaic_img[:,:,0].flatten(), demosaic_img[:,:,1].flatten(), demosaic_img[:,:,2].flatten()))
#print(flat_img[:, 1000:1005])

# wb
flat_img = np.matmul(wb_mat, flat_img)
#print(flat_img[:, 1000:1005])

# cc
flat_img = np.clip(np.matmul(cc_mat, flat_img), 0, 1)
print(flat_img[:, 1000:1005])

cor_color_img = np.stack((flat_img[0].reshape(ishape[0], ishape[1]),
                          flat_img[1].reshape(ishape[0], ishape[1]),
                          flat_img[2].reshape(ishape[0], ishape[1])), axis=2)

# apply gamma
#cor_color_img = cor_color_img ** (1/2.2)
i = cor_color_img < 0.0031308
j = np.logical_not(i)
cor_color_img[i] = 323 / 25 * cor_color_img[i]
cor_color_img[j] = 211 / 200 * cor_color_img[j] ** (5 / 12) - 11 / 200

print(cor_color_img.shape)
color_img = (cor_color_img * 256).astype(np.uint8)
Image.fromarray(color_img, 'RGB').save("color.png")
