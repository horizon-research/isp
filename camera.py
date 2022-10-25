import rawpy
import imageio
import numpy as np
import sys
from PIL import Image
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear
#from scipy.interpolate import interp1d

np.set_printoptions(threshold=sys.maxsize)

class image_signal_processing:
  def __init__(self):
    print("Read image")

    self.raw = rawpy.imread(sys.argv[1])
    # raw_img is a np array
    self.raw_img = self.raw.raw_image

  def create_cfa_indices(self):
    # raw.raw_colors is a numerical mask; we instead generate a char mask so that we can check R/G/B by names
    color_desc = np.frombuffer(self.raw.color_desc, dtype=np.byte) # e.g., RGBG
    cfa_pattern_id = np.array(self.raw.raw_pattern) # e.g., 2310
    cfa_pattern_rgb = np.array([[color_desc[cfa_pattern_id[0, 0]], color_desc[cfa_pattern_id[0, 1]]],
                                [color_desc[cfa_pattern_id[1, 0]], color_desc[cfa_pattern_id[1, 1]]]])
    self.raw_color_index = np.tile(cfa_pattern_rgb, (self.raw.raw_image.shape[0]//2, self.raw.raw_image.shape[1]//2))

  def extract_metadata(self):
    print("Extract metadata")

    #https://letmaik.github.io/rawpy/api/rawpy.RawPy.html
    print(self.raw.color_matrix)
    #print(self.raw.color_desc)
    #print(self.raw.rgb_xyz_matrix)
    #print(self.raw.raw_type)
    #print(self.raw.white_level)
    #print(self.raw.raw_pattern)
    #print(self.raw.num_colors)
    print(self.raw.camera_whitebalance)
    #print(self.raw.black_level_per_channel)
    #print(self.raw.camera_white_level_per_channel)

    self.create_cfa_indices()

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
    self.gs_img = np.where(self.raw_color_index == 0, self.gs_img * self.raw.camera_whitebalance[0], self.gs_img)
    self.gs_img = np.where(self.raw_color_index == 1, self.gs_img * self.raw.camera_whitebalance[1], self.gs_img)
    self.gs_img = np.where(self.raw_color_index == 3, self.gs_img * self.raw.camera_whitebalance[1], self.gs_img)
    self.gs_img = np.where(self.raw_color_index == 2, self.gs_img * self.raw.camera_whitebalance[2], self.gs_img)

  # create bayer-domain raw image that can be displayed as RGB image
  def gen_bayer_rgb_img(self):
    print("Generate RGB image in Bayer domain")

    ##https://stackoverflow.com/questions/19766757/replacing-numpy-elements-if-condition-is-met
    r_channel = np.where(self.raw_color_index == ord('R'), self.gs_img, 0)
    g_channel = np.where(self.raw_color_index == ord('G'), self.gs_img, 0)
    b_channel = np.where(self.raw_color_index == ord('B'), self.gs_img, 0)

    #https://hausetutorials.netlify.app/posts/2019-12-20-numpy-reshape/
    self.bayer_color_img = np.stack((r_channel, g_channel, b_channel), axis=2)

  # demosaic
  def demosaic(self):
    print("Demosaicing")

    #https://colour-demosaicing.readthedocs.io/en/latest/generated/colour_demosaicing.demosaicing_CFA_Bayer_bilinear.html#colour_demosaicing.demosaicing_CFA_Bayer_bilinear

    # RGGB for iPhone; BGGR for Pixel
    cfa_pattern_char = bytes(self.raw_color_index[0:2,0:2].flatten()).decode()

    self.demosaic_img = demosaicing_CFA_Bayer_bilinear(self.gs_img, cfa_pattern_char)

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
    # From photo
    cc_mat = self.raw.color_matrix[0:3, 0:3]
    # Tree
    #cc_mat = np.array([[1.6640625, -0.6796875, 0.0078125], [-0.1484375, 1.3046875, -0.1484375], [0.109375, -0.7421875, 1.640625]])
    # Person
    #cc_mat = np.array([[1.6328125, -0.6171875, -0.015625], [-0.171875, 1.3828125, -0.2109375], [0.125, -0.8046875, 1.6796875]])

    flat_img = np.clip(np.matmul(cc_mat, flat_img), 0, 1)
    print(flat_img[:, 1000:1005])

    self.color_img = np.stack((flat_img[0].reshape(self.raw_img.shape[0], self.raw_img.shape[1]),
                               flat_img[1].reshape(self.raw_img.shape[0], self.raw_img.shape[1]),
                               flat_img[2].reshape(self.raw_img.shape[0], self.raw_img.shape[1])), axis=2)

  def tone_mapping(self):
    print("Tone Mapping")

    #x = [0, 0.8, 0.9, 1]
    #y = [0, 0.95, 0.975, 1]
    #tm = interp1d(x, y, kind='cubic')

    # first Reinhard curve
    #X = self.color_img[:,:,0] * 0.4124564 + self.color_img[:,:,1] * 0.3575761 + self.color_img[:,:,2] * 0.1804375
    #luminance = self.color_img[:,:,0] * 0.2126729 + self.color_img[:,:,1] * 0.7151522 + self.color_img[:,:,2] * 0.0721750
    #Z = self.color_img[:,:,0] * 0.0193339 + self.color_img[:,:,1] * 0.1191920 + self.color_img[:,:,2] * 0.9503041
    #l_avg = np.average(luminance)
    #l_adj = luminance / (9.6 * l_avg + 0.0001)
    #l_out = l_adj / (1 + l_adj)
    #print(luminance[0:3,0:3])
    #print(l_out[0:3,0:3])
    #R = X * 3.2404542 + l_out * -1.5371385 + Z * -0.4985314
    #G = X * -0.9692660 + l_out * 1.8760108 + Z * 0.0415560
    #B = X * 0.0556434 + l_out * -0.2040259 + Z * 1.0572252
    #self.color_img = np.stack((R, G, B), axis=2)

    # simple mapper
    i = self.color_img < 0.5
    j = np.logical_not(i)
    self.color_img[i] = self.color_img[i] * (1 / 0.5)
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

    # save raw bayer-domain image as a gray-scale image
    #Image.fromarray((self.gs_img * 256).astype(np.uint8)).save("gs.png")
    #Image.fromarray((self.bayer_color_img * 256).astype(np.uint8), 'RGB').save("bayer.png")
    #Image.fromarray((self.demosaic_img * 256).astype(np.uint8), 'RGB').save("demosaic.png")
    Image.fromarray((self.color_img * 256).astype(np.uint8), 'RGB').save("color.png")

def main():
  isp = image_signal_processing()

  isp.extract_metadata()
  #isp.extract_thumb()
  isp.subtract_bl_norm()
  #isp.apply_wb_gain()
  #isp.gen_bayer_rgb_img()
  isp.demosaic()
  isp.apply_wb_cc()
  isp.tone_mapping()
  isp.apply_gamma()
  isp.save_images()

if __name__ == "__main__":
  main()
