import rawpy
import imageio
import numpy as np
import sys
from PIL import Image
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear

np.set_printoptions(threshold=sys.maxsize)

def print_metadata(raw):
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

def extract_thumb(raw):
  try:
      thumb = raw.extract_thumb()
  except rawpy.LibRawNoThumbnailError:
    print('no thumbnail found')
  except rawpy.LibRawUnsupportedThumbnailError:
    print('unsupported thumbnail')
  else:
    if thumb.format == rawpy.ThumbFormat.JPEG:
      with open('thumb.jpg', 'wb') as f:
        f.write(thumb.data)
    elif thumb.format == rawpy.ThumbFormat.BITMAP:
      imageio.imsave('thumb.tiff', thumb.data)

# subtract black level and normalize
def subtract_bl_norm(raw, raw_img):
  black = np.reshape(raw.black_level_per_channel, (2, 2))
  black = np.tile(black, (raw_img.shape[0]//2, raw_img.shape[1]//2))
  gs_img = (raw_img - black) / (raw.white_level - black)

  # save raw bayer-domain image as a gray-scale image
  #Image.fromarray((gs_img * 256).astype(np.uint8)).save("gs.png")
  return gs_img

# apply wb gains; could either do here or multiply the wb matrix after demosaic; equivalent for bilinear filtering
def apply_wb_gain(raw, gs_img):
  raw_color_index = raw.raw_colors
  gs_img = np.where(raw_color_index == 0, gs_img * raw.camera_whitebalance[0], gs_img)
  gs_img = np.where(raw_color_index == 1, gs_img * raw.camera_whitebalance[1], gs_img)
  gs_img = np.where(raw_color_index == 3, gs_img * raw.camera_whitebalance[1], gs_img)
  gs_img = np.where(raw_color_index == 2, gs_img * raw.camera_whitebalance[2], gs_img)
  return gs_img

# create bayer-domain raw image that can be displayed as RGB image
def gen_bayer_rgb_img(raw, gs_img):
  raw_color_index = raw.raw_colors

  ##https://stackoverflow.com/questions/19766757/replacing-numpy-elements-if-condition-is-met
  r_channel = np.where(raw_color_index == 0, gs_img, 0)
  g_channel = np.where(raw_color_index == 1, gs_img, 0)
  g_channel = np.where(raw_color_index == 3, gs_img, g_channel)
  b_channel = np.where(raw_color_index == 2, gs_img, 0)
  
  #https://hausetutorials.netlify.app/posts/2019-12-20-numpy-reshape/
  color_img = np.stack((r_channel, g_channel, b_channel)).transpose((1, 2, 0))
  Image.fromarray((color_img * 256).astype(np.uint8), 'RGB').save("bayer.png")

# demosaic
def demosaic(gs_img):
  #https://colour-demosaicing.readthedocs.io/en/latest/generated/colour_demosaicing.demosaicing_CFA_Bayer_bilinear.html#colour_demosaicing.demosaicing_CFA_Bayer_bilinear

  #demosaic_img = demosaicing_CFA_Bayer_bilinear(gs_img, 'RGGB') #iphone
  demosaic_img = demosaicing_CFA_Bayer_bilinear(gs_img, 'BGGR') #pixel

  #Image.fromarray((demosaic_img * 256).astype(np.uint8), 'RGB').save("demosaic.png")

  return demosaic_img

# white balance and color correction 
# using the method where we first apply a rotation matrix to normalize to the capture white point and then apply the correction matrix, which does the real chromatic adaptation
# see: https://www.spiedigitallibrary.org/journals/optical-engineering/volume-59/issue-11/110801/Color-conversion-matrices-in-digital-cameras-a-tutorial/10.1117/1.OE.59.11.110801.full?SSO=1
def apply_wb_cc(raw, demosaic_img, raw_img):
  #https://www.pythoninformer.com/python-libraries/numpy/index-and-slice/
  # form a Nx3 array from the image pixels
  flat_img = np.stack((demosaic_img[:,:,0].flatten(), demosaic_img[:,:,1].flatten(), demosaic_img[:,:,2].flatten()))
  #print(flat_img[:, 1000:1005])

  # wb
  wb_mat = np.array([[raw.camera_whitebalance[0], 0, 0], [0, raw.camera_whitebalance[1], 0], [0, 0, raw.camera_whitebalance[2]]])
  flat_img = np.matmul(wb_mat, flat_img)
  #print(flat_img[:, 1000:1005])

  # cc
  # From photo
  #cc_mat = raw.color_matrix[0:3, 0:3]
  # Tree
  #cc_mat = np.array([[1.6640625, -0.6796875, 0.0078125], [-0.1484375, 1.3046875, -0.1484375], [0.109375, -0.7421875, 1.640625]])
  # Person
  cc_mat = np.array([[1.6328125, -0.6171875, -0.015625], [-0.171875, 1.3828125, -0.2109375], [0.125, -0.8046875, 1.6796875]])

  flat_img = np.clip(np.matmul(cc_mat, flat_img), 0, 1)
  print(flat_img[:, 1000:1005])

  color_img = np.stack((flat_img[0].reshape(raw_img.shape[0], raw_img.shape[1]),
                            flat_img[1].reshape(raw_img.shape[0], raw_img.shape[1]),
                            flat_img[2].reshape(raw_img.shape[0], raw_img.shape[1])), axis=2)

  return color_img

# apply gamma and save image
def apply_gamma(color_img):
  #color_img = color_img ** (1/2.2)

  i = color_img < 0.0031308
  j = np.logical_not(i)
  color_img[i] = 323 / 25 * color_img[i]
  color_img[j] = 211 / 200 * color_img[j] ** (5 / 12) - 11 / 200

  print(color_img.shape)
  color_img = (color_img * 256).astype(np.uint8)
  Image.fromarray(color_img, 'RGB').save("color.png")

def main():
  raw = rawpy.imread(sys.argv[1])
  # raw_img is a np array
  raw_img = raw.raw_image

  print_metadata(raw)
  extract_thumb(raw)
  gs_img = subtract_bl_norm(raw, raw_img)
  #gs_img = apply_wb_gain(raw, gs_img)
  #gen_bayer_rgb_img(raw, gs_img)
  demosaic_img = demosaic(gs_img)
  color_img = apply_wb_cc(raw, demosaic_img, raw_img)
  apply_gamma(color_img)

if __name__ == "__main__":
  main()
