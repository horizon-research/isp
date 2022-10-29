import rawpy
import imageio
import numpy as np
import sys
import argparse
from PIL import Image
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear
from scipy import interpolate

np.set_printoptions(threshold=sys.maxsize)

def parseArg():
  parser = argparse.ArgumentParser()
  parser.add_argument('-sg', action='store_true', help='save bayer-domain image as gray-scale image')
  parser.add_argument('-sb', action='store_true', help='save bayer-domain image as RGB image')
  parser.add_argument('-sd', action='store_true', help='save demosaic-ed image')
  parser.add_argument('-sc', action='store_true', help='save final RGB image')
  parser.add_argument('-et', action='store_true', help='extract thumb JPEG file if exists')

  parser.add_argument('-im', type=str, required=True, help='input image path')
  parser.add_argument('-lm', type=str, required=False, help='optional lens shading correction map path')
  parser.add_argument('-cm', type=str, required=False, help='optional color correction matrix math')

  args = parser.parse_args()
  return args

class image_signal_processing:
  def __init__(self):
    print("Read image")

    self.args = parseArg()

    self.raw = rawpy.imread(self.args.im)
    # raw_img is a np array
    self.raw_img = self.raw.raw_image
    print(self.raw_img.shape)

    if (self.args.lm):
      self.lens_sm = np.array(imageio.imread(self.args.lm))
      print(self.lens_sm.shape)

  def create_cfa_indices(self):
    # raw.raw_colors is a numerical mask; we instead generate a char mask so that we can check R/G/B by names

    # this is based on spatial raster order on the CFA; 01/32 on iPhone; 23/10 on Pixel
    cfa_pattern_id = np.array(self.raw.raw_pattern)
    # this is baesd on numerical order above; RGBG on iPhone; RGBG on Pixel; 
    color_desc = np.frombuffer(self.raw.color_desc, dtype=np.byte)

    # this tile is based on the raster order
    # https://stackoverflow.com/questions/14639496/how-to-create-a-numpy-array-of-arbitrary-length-strings
    tile_pattern = np.array([[chr(color_desc[cfa_pattern_id[0, 0]]), chr(color_desc[cfa_pattern_id[0, 1]])],
                             [chr(color_desc[cfa_pattern_id[1, 0]]), chr(color_desc[cfa_pattern_id[1, 1]])]], dtype=object)
    self.cfa_pattern_rgb = np.array(tile_pattern, copy=True) # make a deep copy

    # generate GR and GB (for lens shading correction later)
    for i in range(2):
      for j in range(2):
        if (tile_pattern[i,j] == 'G'):
          tile_pattern[i,j] = 'G' + tile_pattern[i,(j+1)%2]
    print(tile_pattern)

    self.raw_color_index = np.tile(tile_pattern, (self.raw.raw_image.shape[0]//2, self.raw.raw_image.shape[1]//2))

  def extract_metadata(self):
    print("Extract metadata")

    #https://letmaik.github.io/rawpy/api/rawpy.RawPy.html
    ###print(self.raw.color_matrix)
    ###print(self.raw.camera_whitebalance)

    #print(self.raw.color_desc)
    #print(self.raw.rgb_xyz_matrix)
    #print(self.raw.raw_type)
    #print(self.raw.white_level)
    #print(self.raw.raw_pattern)
    #print(self.raw.num_colors)
    #print(self.raw.black_level_per_channel)
    #print(self.raw.camera_white_level_per_channel)

    self.create_cfa_indices()

    if (self.args.et):
      self.extract_thumb()

  def extract_thumb(self):
    print("Extract thumbnail")

    try:
        thumb = self.raw.extract_thumb()
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
  def subtract_bl_norm(self):
    print("Subtract blacklevel")

    black = np.reshape(self.raw.black_level_per_channel, (2, 2))
    black = np.tile(black, (self.raw_img.shape[0]//2, self.raw_img.shape[1]//2))
    self.gs_img = (self.raw_img - black) / (self.raw.white_level - black)

  # apply wb gains; could either do here or multiply the wb matrix after demosaic; equivalent for bilinear filtering
  def apply_wb_gain(self):
    self.gs_img = np.where(self.raw_color_index == 'R', self.gs_img * self.raw.camera_whitebalance[0], self.gs_img)
    self.gs_img = np.where(self.raw_color_index == 'GR', self.gs_img * self.raw.camera_whitebalance[1], self.gs_img)
    self.gs_img = np.where(self.raw_color_index == 'GB', self.gs_img * self.raw.camera_whitebalance[1], self.gs_img)
    self.gs_img = np.where(self.raw_color_index == 'B', self.gs_img * self.raw.camera_whitebalance[2], self.gs_img)

  # create bayer-domain raw image that can be displayed as RGB image
  def gen_bayer_rgb_img(self):
    ##https://stackoverflow.com/questions/19766757/replacing-numpy-elements-if-condition-is-met
    r_channel = np.where(self.raw_color_index == 'R', self.gs_img, 0)
    g_channel = np.where(((self.raw_color_index == 'GR') | (self.raw_color_index == 'GB')), self.gs_img, 0)
    b_channel = np.where(self.raw_color_index == 'B', self.gs_img, 0)

    #https://hausetutorials.netlify.app/posts/2019-12-20-numpy-reshape/
    self.bayer_color_img = np.stack((r_channel, g_channel, b_channel), axis=2)

  # https://developer.android.com/reference/android/hardware/camera2/CaptureResult#STATISTICS_LENS_SHADING_CORRECTION_MAP
  def lens_shading_correction(self):
    if (not self.args.lm):
      return

    print("Lens shading correction")

    x = np.append(np.arange(0, self.raw_img.shape[0] - 1,
        (self.raw_img.shape[0] - 1)/(self.lens_sm.shape[0] - 1)), [self.raw_img.shape[0] - 1])
    y = np.append(np.arange(0, self.raw_img.shape[1] - 1,
        (self.raw_img.shape[1] - 1)/(self.lens_sm.shape[1] - 1)), [self.raw_img.shape[1] - 1])

    # When on a regular grid with x.size = m and y.size = n, if z.ndim == 2, then z must have shape (n, m)
    f = interpolate.interp2d(y, x, self.lens_sm[:,:,0], kind='quintic')
    lens_sm_r = f(np.arange(0, self.raw_img.shape[1], 1), np.arange(0, self.raw_img.shape[0], 1))
    f = interpolate.interp2d(y, x, self.lens_sm[:,:,1], kind='quintic')
    lens_sm_g_red = f(np.arange(0, self.raw_img.shape[1], 1), np.arange(0, self.raw_img.shape[0], 1))
    f = interpolate.interp2d(y, x, self.lens_sm[:,:,2], kind='quintic')
    lens_sm_g_blue = f(np.arange(0, self.raw_img.shape[1], 1), np.arange(0, self.raw_img.shape[0], 1))
    f = interpolate.interp2d(y, x, self.lens_sm[:,:,3], kind='quintic')
    lens_sm_b = f(np.arange(0, self.raw_img.shape[1], 1), np.arange(0, self.raw_img.shape[0], 1))

    #print(self.gs_img[0:5,0:5])
    self.gs_img = np.where(self.raw_color_index == 'R', self.gs_img * lens_sm_r, self.gs_img)
    self.gs_img = np.where(self.raw_color_index == 'GR', self.gs_img * lens_sm_g_red, self.gs_img)
    self.gs_img = np.where(self.raw_color_index == 'GB', self.gs_img * lens_sm_g_blue, self.gs_img)
    self.gs_img = np.where(self.raw_color_index == 'B', self.gs_img * lens_sm_b, self.gs_img)
    #print(self.gs_img[0:5,0:5])

  # demosaic
  def demosaic(self):
    print("Demosaicing")

    #https://colour-demosaicing.readthedocs.io/en/latest/generated/colour_demosaicing.demosaicing_CFA_Bayer_bilinear.html#colour_demosaicing.demosaicing_CFA_Bayer_bilinear

    cfa_pattern_char = "".join(self.cfa_pattern_rgb.flatten())

    # this expects spatial raster order on the CFA; RGGB for iPhone; BGGR for Pixel
    self.demosaic_img = demosaicing_CFA_Bayer_bilinear(self.gs_img, cfa_pattern_char)

  def extractCCM(self):
    with open(self.args.cm) as ccm_file:
      for line in ccm_file:
        return np.asarray(line.split()).astype(np.float).reshape(3, 3)

  # white balance and color correction
  # using the method where we first apply a rotation matrix to normalize to the
  # capture white point and then apply the correction matrix, which does the
  # real chromatic adaptation. see:
  # https://www.spiedigitallibrary.org/journals/optical-engineering/volume-59/issue-11/110801/Color-conversion-matrices-in-digital-cameras-a-tutorial/10.1117/1.OE.59.11.110801.full?SSO=1
  def apply_wb_cc(self):
    print("White balance and color correction")

    #https://www.pythoninformer.com/python-libraries/numpy/index-and-slice/
    # form a Nx3 array from the image pixels
    flat_img = np.stack((self.demosaic_img[:,:,0].flatten(),
                         self.demosaic_img[:,:,1].flatten(),
                         self.demosaic_img[:,:,2].flatten()))
    #print(flat_img[:, 1000:1005])

    # wb
    wb_mat = np.array([[self.raw.camera_whitebalance[0], 0, 0],
                       [0, self.raw.camera_whitebalance[1], 0],
                       [0, 0, self.raw.camera_whitebalance[2]]])
    flat_img = np.matmul(wb_mat, flat_img)
    #print(flat_img[:, 1000:1005])

    # cc
    if (self.args.cm):
      cc_mat = self.extractCCM()
    else:
      cc_mat = self.raw.color_matrix[0:3, 0:3]

    flat_img = np.clip(np.matmul(cc_mat, flat_img), 0, 1)
    #print(flat_img[:, 1000:1005])

    self.color_img = np.stack((flat_img[0].reshape(self.raw_img.shape[0], self.raw_img.shape[1]),
                               flat_img[1].reshape(self.raw_img.shape[0], self.raw_img.shape[1]),
                               flat_img[2].reshape(self.raw_img.shape[0], self.raw_img.shape[1])), axis=2)

  def tone_mapping(self):
    print("Tone Mapping")

    # simple mapper
    i = self.color_img < 0.5
    j = np.logical_not(i)
    self.color_img[i] = self.color_img[i] * (1 / 0.5)
    self.color_img[i] = self.color_img[i] ** (1/1.2)
    self.color_img[j] = 1

    #print(self.color_img[self.color_img > 1])

  # apply gamma and save image
  def apply_gamma(self):
    print("Apply gamma")

    #self.color_img = self.color_img ** (1/2.2)

    i = self.color_img < 0.0031308
    j = np.logical_not(i)
    self.color_img[i] = 323 / 25 * self.color_img[i]
    self.color_img[j] = 211 / 200 * self.color_img[j] ** (5 / 12) - 11 / 200

  def save_images(self):
    print("Save images")

    if (self.args.sg):
      # save raw bayer-domain image as a gray-scale image
      Image.fromarray((self.gs_img * 256).astype(np.uint8)).save("gs.png")

    if (self.args.sb):
      self.gen_bayer_rgb_img()
      Image.fromarray((self.bayer_color_img * 256).astype(np.uint8), 'RGB').save("bayer.png")

    if (self.args.sd):
      Image.fromarray((self.demosaic_img * 256).astype(np.uint8), 'RGB').save("demosaic.png")

    if (self.args.sc):
      Image.fromarray((self.color_img * 256).astype(np.uint8), 'RGB').save("color.png")

def main():
  parseArg()
  isp = image_signal_processing()

  isp.extract_metadata()
  isp.subtract_bl_norm()
  isp.lens_shading_correction()
  #isp.apply_wb_gain()
  isp.demosaic()
  isp.apply_wb_cc()
  isp.tone_mapping()
  isp.apply_gamma()
  isp.save_images()

if __name__ == "__main__":
  main()
