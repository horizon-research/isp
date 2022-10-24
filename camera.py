import rawpy
import imageio
import numpy as np
import sys
from PIL import Image
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear

np.set_printoptions(threshold=sys.maxsize)

class image_signal_processing:
  def __init__(self):
    self.raw = rawpy.imread(sys.argv[1])
    # raw_img is a np array
    self.raw_img = self.raw.raw_image

  def create_cfa_index(self):
    # raw.raw_colors is a numerical mask; we instead generate a char mask so that we can check R/G/B by names
    color_desc = np.frombuffer(self.raw.color_desc, dtype=np.byte) # e.g., RGBG
    cfa_pattern = np.array(self.raw.raw_pattern) # e.g., 2310
    cfa_pattern_rgb = np.array([[color_desc[cfa_pattern[0, 0]], color_desc[cfa_pattern[0, 1]]],
                               [color_desc[cfa_pattern[1, 0]], color_desc[cfa_pattern[1, 1]]]])
    self.raw_color_index = np.tile(cfa_pattern_rgb, (self.raw.raw_image.shape[0]//2, self.raw.raw_image.shape[1]//2))
  
  def extract_metadata(self):
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

    self.create_cfa_index()
    
  def extract_thumb(self):
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
    black = np.reshape(self.raw.black_level_per_channel, (2, 2))
    black = np.tile(black, (self.raw_img.shape[0]//2, self.raw_img.shape[1]//2))
    self.gs_img = (self.raw_img - black) / (self.raw.white_level - black)
  
    # save raw bayer-domain image as a gray-scale image
    #Image.fromarray((gs_img * 256).astype(np.uint8)).save("gs.png")
  
  # apply wb gains; could either do here or multiply the wb matrix after demosaic; equivalent for bilinear filtering
  def apply_wb_gain(self):
    self.gs_img = np.where(self.raw_color_index == 0, self.gs_img * self.raw.camera_whitebalance[0], self.gs_img)
    self.gs_img = np.where(self.raw_color_index == 1, self.gs_img * self.raw.camera_whitebalance[1], self.gs_img)
    self.gs_img = np.where(self.raw_color_index == 3, self.gs_img * self.raw.camera_whitebalance[1], self.gs_img)
    self.gs_img = np.where(self.raw_color_index == 2, self.gs_img * self.raw.camera_whitebalance[2], self.gs_img)
  
  # create bayer-domain raw image that can be displayed as RGB image
  def gen_bayer_rgb_img(self):
    ##https://stackoverflow.com/questions/19766757/replacing-numpy-elements-if-condition-is-met
    r_channel = np.where(self.raw_color_index == ord('R'), self.gs_img, 0)
    g_channel = np.where(self.raw_color_index == ord('G'), self.gs_img, 0)
    b_channel = np.where(self.raw_color_index == ord('B'), self.gs_img, 0)
    
    #https://hausetutorials.netlify.app/posts/2019-12-20-numpy-reshape/
    color_img = np.stack((r_channel, g_channel, b_channel), axis=2)
    Image.fromarray((color_img * 256).astype(np.uint8), 'RGB').save("bayer.png")
  
  # demosaic
  def demosaic(self):
    #https://colour-demosaicing.readthedocs.io/en/latest/generated/colour_demosaicing.demosaicing_CFA_Bayer_bilinear.html#colour_demosaicing.demosaicing_CFA_Bayer_bilinear
  
    #self.demosaic_img = demosaicing_CFA_Bayer_bilinear(self.gs_img, 'RGGB') #iphone
    self.demosaic_img = demosaicing_CFA_Bayer_bilinear(self.gs_img, 'BGGR') #pixel
  
    #Image.fromarray((self.demosaic_img * 256).astype(np.uint8), 'RGB').save("demosaic.png")
  
  # white balance and color correction 
  # using the method where we first apply a rotation matrix to normalize to the
  # capture white point and then apply the correction matrix, which does the
  # real chromatic adaptation. see:
  # https://www.spiedigitallibrary.org/journals/optical-engineering/volume-59/issue-11/110801/Color-conversion-matrices-in-digital-cameras-a-tutorial/10.1117/1.OE.59.11.110801.full?SSO=1
  def apply_wb_cc(self):
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
    #cc_mat = self.raw.color_matrix[0:3, 0:3]
    # Tree
    #cc_mat = np.array([[1.6640625, -0.6796875, 0.0078125], [-0.1484375, 1.3046875, -0.1484375], [0.109375, -0.7421875, 1.640625]])
    # Person
    cc_mat = np.array([[1.6328125, -0.6171875, -0.015625], [-0.171875, 1.3828125, -0.2109375], [0.125, -0.8046875, 1.6796875]])
  
    flat_img = np.clip(np.matmul(cc_mat, flat_img), 0, 1)
    #print(flat_img[:, 1000:1005])
  
    self.color_img = np.stack((flat_img[0].reshape(self.raw_img.shape[0], self.raw_img.shape[1]),
                               flat_img[1].reshape(self.raw_img.shape[0], self.raw_img.shape[1]),
                               flat_img[2].reshape(self.raw_img.shape[0], self.raw_img.shape[1])), axis=2)
  
  # apply gamma and save image
  def apply_gamma(self):
    #self.color_img = self.color_img ** (1/2.2)
  
    i = self.color_img < 0.0031308
    j = np.logical_not(i)
    self.color_img[i] = 323 / 25 * self.color_img[i]
    self.color_img[j] = 211 / 200 * self.color_img[j] ** (5 / 12) - 11 / 200
  
    print(self.color_img.shape)
    self.color_img = (self.color_img * 256).astype(np.uint8)
    Image.fromarray(self.color_img, 'RGB').save("color.png")

def main():
  isp = image_signal_processing()

  isp.extract_metadata()
  isp.extract_thumb()
  isp.subtract_bl_norm()
  #isp.apply_wb_gain()
  isp.gen_bayer_rgb_img()
  sys.exit()
  isp.demosaic()
  isp.apply_wb_cc()
  isp.apply_gamma()

if __name__ == "__main__":
  main()
