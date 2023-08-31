import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageEnhance
import tomograph
import argparse
import glob
import filter
import uuid,os
from skimage.io import imread
from skimage.transform import rescale
from math import log10, sqrt

#average
def average_filter(imagepath,savedir):
  img = cv2.imread(imagepath)

  kernel = np.ones((5, 5), np.float32) / 25
  dst = cv2.filter2D(img, -1, kernel)

  plt.subplot(121), plt.imshow(img), plt.title('Original')
  plt.xticks([]), plt.yticks([])
  plt.subplot(122), plt.imshow(dst), plt.title('Averaging')
  plt.xticks([]), plt.yticks([])

  image_name = str(uuid.uuid4())
  if not os.path.exists(savedir):
    os.mkdir(savedir)
  cv2.imwrite(os.path.join(savedir,"{}.jpg".format(image_name)),dst)


#enhance
def enhance_filter(imagepath,savedir):
  img = cv2.imread(imagepath)

  im = Image.open('C:\\Users\\nurcan\\Desktop\\filters\\images.jpg')
  factor = 2
  im_out = ImageEnhance.Color(im).enhance(factor)

  image_name = str(uuid.uuid4())
  if not os.path.exists(savedir):
    os.mkdir(savedir)

  im_out.save(os.path.join(savedir,"{}.jpg".format(image_name)), im_out)

  image_name = str(uuid.uuid4())
  if not os.path.exists(savedir):
    os.mkdir(savedir)
  cv2.imwrite(os.path.join(savedir,"{}.jpg".format(image_name)),img)

 # plt.imsave(os.path.join(savedir,"{}.jpg".format(image_name)), laplacian, cmap='gray', format='png')


#keskinleştirme
def  keskinlestirme_filter(imagepath,savedir):
  img = cv2.imread(imagepath)

  grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  #edge_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
  sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
  imgf = cv2.filter2D(img, -1, sharpen_kernel)

  image_name = str(uuid.uuid4())
  if not os.path.exists(savedir):
    os.mkdir(savedir)
  cv2.imwrite(os.path.join(savedir,"{}.jpg".format(image_name)),imgf)


#canny
def canny_filter(imagepath,savedir):
  img = cv2.imread(imagepath)

  def auto_canny(image, sigma=0.33):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = Canny_detector(image, lower, upper)
    return edged

  def Canny_detector(img, weak_th=None, strong_th=None):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('black', img)

    img = cv2.GaussianBlur(img, (5, 5), 1.4)

    gx = cv2.Sobel(np.float32(img), cv2.CV_64F, 1, 0, 3)
    gy = cv2.Sobel(np.float32(img), cv2.CV_64F, 0, 1, 3)

    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    mag_max = np.max(mag)
    if not weak_th:
      weak_th = mag_max * 0.1
    if not strong_th:
      strong_th = mag_max * 0.5

    height, width = img.shape

    for i_x in range(width):
      for i_y in range(height):

        grad_ang = ang[i_y, i_x]
        grad_ang = abs(grad_ang - 180) if abs(grad_ang) > 180 else abs(grad_ang)

        if grad_ang <= 22.5:
          neighb_1_x, neighb_1_y = i_x - 1, i_y
          neighb_2_x, neighb_2_y = i_x + 1, i_y

        elif grad_ang > 22.5 and grad_ang <= (22.5 + 45):
          neighb_1_x, neighb_1_y = i_x - 1, i_y - 1
          neighb_2_x, neighb_2_y = i_x + 1, i_y + 1

        elif grad_ang > (22.5 + 45) and grad_ang <= (22.5 + 90):
          neighb_1_x, neighb_1_y = i_x, i_y - 1
          neighb_2_x, neighb_2_y = i_x, i_y + 1

        elif grad_ang > (22.5 + 90) and grad_ang <= (22.5 + 135):
          neighb_1_x, neighb_1_y = i_x - 1, i_y + 1
          neighb_2_x, neighb_2_y = i_x + 1, i_y - 1

        elif grad_ang > (22.5 + 135) and grad_ang <= (22.5 + 180):
          neighb_1_x, neighb_1_y = i_x - 1, i_y
          neighb_2_x, neighb_2_y = i_x + 1, i_y

        if width > neighb_1_x >= 0 and height > neighb_1_y >= 0:
          if mag[i_y, i_x] < mag[neighb_1_y, neighb_1_x]:
            mag[i_y, i_x] = 0
            continue

        if width > neighb_2_x >= 0 and height > neighb_2_y >= 0:
          if mag[i_y, i_x] < mag[neighb_2_y, neighb_2_x]:
            mag[i_y, i_x] = 0

    ids = np.zeros_like(img)

    for i_x in range(width):
      for i_y in range(height):

        grad_mag = mag[i_y, i_x]

        if grad_mag < weak_th:
          mag[i_y, i_x] = 0
        elif strong_th > grad_mag >= weak_th:
          ids[i_y, i_x] = 1
        else:
          ids[i_y, i_x] = 2
          mag[i_y, i_x] = 255

    for i in range(width):
      for j in range(height):

        try:
          if (ids[i, j] == 1):

            if ((ids[i + 1, j - 1] == 2) or (ids[i + 1, j] == 2) or (ids[i + 1, j + 1] == 2)
                    or (ids[i, j - 1] == 2) or (ids[i, j + 1] == 2)
                    or (ids[i - 1, j - 1] == 2) or (ids[i - 1, j] == 2) or (ids[i - 1, j + 1] == 2)):

              mag[i, j] = 255
              ids[i, j] = 2
            else:
              mag[i, j] = 0

        except IndexError as e:
          pass

    for i in range(width, 0, -1):
      for j in range(height, 0, -1):

        try:
          if (ids[i, j] == 1):

            if ((ids[i + 1, j - 1] == 2) or (ids[i + 1, j] == 2) or (ids[i + 1, j + 1] == 2)
                    or (ids[i, j - 1] == 2) or (ids[i, j + 1] == 2)
                    or (ids[i - 1, j - 1] == 2) or (ids[i - 1, j] == 2) or (ids[i - 1, j + 1] == 2)):

              mag[i, j] = 255
              ids[i, j] = 2
            else:
              mag[i, j] = 0

        except IndexError as e:
          pass

    for i in range(width):
      for j in range(height, 0, -1):

        try:
          if (ids[i, j] == 1):

            if ((ids[i + 1, j - 1] == 2) or (ids[i + 1, j] == 2) or (ids[i + 1, j + 1] == 2)
                    or (ids[i, j - 1] == 2) or (ids[i, j + 1] == 2)
                    or (ids[i - 1, j - 1] == 2) or (ids[i - 1, j] == 2) or (ids[i - 1, j + 1] == 2)):

              mag[i, j] = 255
              ids[i, j] = 2
            else:
              mag[i, j] = 0

        except IndexError as e:
          pass

    for i in range(width, 0, -1):
      for j in range(height):

        try:
          if (ids[i, j] == 1):

            if ((ids[i + 1, j - 1] == 2) or (ids[i + 1, j] == 2) or (ids[i + 1, j + 1] == 2)
                    or (ids[i, j - 1] == 2) or (ids[i, j + 1] == 2)
                    or (ids[i - 1, j - 1] == 2) or (ids[i - 1, j] == 2) or (ids[i - 1, j + 1] == 2)):

              mag[i, j] = 255
              ids[i, j] = 2
            else:
              mag[i, j] = 0

        except IndexError as e:
          pass

    return mag

  img = cv2.imread("C:\\Users\\nurcan\\Desktop\\filters\\images.jpg")


  def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if (mse == 0):

      return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    print('\n')
    print(f"MSE: {mse}")
    print(f"PSNR: {psnr}")

  canny_output = "C:\\Users\\nurcan\\Desktop\\filters\\images.jpg"

  for i in range(1, 10):
    # image_path = "C:\\Users\\nurcan\\Desktop\\filter\\1.jpg"
    # img = cv2.imread(image_path)
    imgf = auto_canny(img)
    #os.chdir(canny_output)


  image_name = str(uuid.uuid4())
  if not os.path.exists(savedir):
    os.mkdir(savedir)
  cv2.imwrite(os.path.join(savedir, "{}.jpg".format(image_name)), imgf)


#prewitt
def prewitt_filter(imagepath,savedir):
  img = cv2.imread(imagepath)

  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img_gaussian = cv2.GaussianBlur(gray, (3, 3), 0)
  kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
  kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
  img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx)
  img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)
  prewitt = img_prewittx + img_prewitty

  image_name = str(uuid.uuid4())
  if not os.path.exists(savedir):
    os.mkdir(savedir)
  cv2.imwrite(os.path.join(savedir,"{}.jpg".format(image_name)),prewitt)


#sobel
def sobel_filter(imagepath,savedir):
  img = cv2.imread(imagepath)

  Gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  Horizontal = cv2.Sobel(Gray, 0, 1, 0, cv2.CV_64F)
  Vertical = cv2.Sobel(Gray, 0, 0, 1, cv2.CV_64F)
  Bitwise_Or = cv2.bitwise_or(Horizontal, Vertical)


  image_name = str(uuid.uuid4())
  if not os.path.exists(savedir):
    os.mkdir(savedir)
  cv2.imwrite(os.path.join(savedir,"{}.jpg".format(image_name)),Bitwise_Or)


#laplacian
def laplacian_filter(imagepath,savedir):
  img = cv2.imread(imagepath)

  gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  blur_img = cv2.GaussianBlur(gray_img, (3, 3), 5)
  laplacian = cv2.Laplacian(blur_img, cv2.CV_64F)
  plt.figure()
  plt.title('Shapes')

  image_name = str(uuid.uuid4())
  if not os.path.exists(savedir):
    os.mkdir(savedir)
  plt.imsave(os.path.join(savedir,"{}.jpg".format(image_name)), laplacian, cmap='gray', format='png')
  plt.imshow(laplacian, cmap='gray')


#gabor
def gabor_filter(imagepath,savedir):
  img = cv2.imread(imagepath)

#g_kernel = cv2.getGaborKernel((30,30), 18.0, np.pi/30, 31.0, 5.5, 9, ktype=cv2.CV_32F)
  img = cv2.imread('C:\\Users\\nurcan\\Desktop\\filters\\images.jpg')
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  g_kernel = cv2.getGaborKernel((30,30), 18.0, np.pi/30, 31.0, 5.5, 9, ktype=cv2.CV_32F)
  filtered_img = cv2.filter2D(img, cv2.CV_8UC3, g_kernel)

  h, w = g_kernel.shape[:2]
  g_kernel = cv2.resize(g_kernel, (3*w, 3*h), interpolation=cv2.INTER_CUBIC)


  image_name = str(uuid.uuid4())
  if not os.path.exists(savedir):
    os.mkdir(savedir)
    cv2.imwrite(os.path.join(savedir, "{}.jpg".format(image_name)), filtered_img)


#radon
class Params:
  def __init__(self, image_path, theta, detector_quantity, span):
    self.image_path = image_path
    self.theta = np.deg2rad(float(theta))
    self.detector_quantity = int(detector_quantity)
    self.span = np.deg2rad(span)

def make_image_square(image_original):
  diagonal = np.sqrt(2) * max(image_original.shape)
  pad = [int(np.ceil(diagonal - s)) for s in image_original.shape]
  new_center = [(s + p) // 2 for s, p in zip(image_original.shape, pad)]
  old_center = [s // 2 for s in image_original.shape]
  pad_before = [nc - oc for oc, nc in zip(old_center, new_center)]
  pad_width = [(pb, p - pb) for pb, p in zip(pad_before, pad)]
  return np.pad(image_original, pad_width, mode='constant', constant_values=0)

def radon_filter(self, savedir):
  params = Params("C:\\Users\\nurcan\\Desktop\\filters\\images.jpg", 4, 100, 180)
  image_original = imread(params.image_path, as_gray=True)
  image_rescaled = rescale(image_original, scale=0.4)
  image_padded = make_image_square(image_rescaled)
  emitter_angles = tomograph.generate_angles(params.theta)
  sinogram = tomograph.radon(image_padded, emitter_angles, params.detector_quantity, params.span)
  sinogram_filtered = filter.filter_sinogram(sinogram, "ramp")
  image_reconstructed_filtered = tomograph.inverse_radon(sinogram_filtered, image_padded.shape[0], emitter_angles,
                                                         params.detector_quantity, params.span)
  image_reconstructed = tomograph.inverse_radon(sinogram, image_padded.shape[0], emitter_angles,
                                                params.detector_quantity, params.span)

  fig, ax = plt.subplots(2, 2)
  ax[0, 0].set_title("Original image")
  ax[0, 1].set_title("Sinogram")
  ax[1, 0].set_title("Reconstructed image")
  ax[1, 1].set_title("Filtered and reconstructed image")
  ax[0, 1].set_xlabel("Detector Index")
  ax[0, 1].set_ylabel("Projection step")
  ax[0, 0].imshow(image_padded, cmap="gray")
  ax[0, 1].imshow(sinogram, cmap="gray")
  ax[1, 0].imshow(image_reconstructed, cmap="gray")
  ax[1, 1].imshow(image_reconstructed_filtered, cmap="gray")


  image_name = str(uuid.uuid4())
  if not os.path.exists(savedir):
    os.mkdir(savedir)
    plt.tight_layout(0, -0.8, 0)
    plt.savefig(os.path.join(savedir, "{}.png".format(image_name)))
    #plt.show()


#minfilter
def minimum_filter(imagepath,savedir,n):
  img = cv2.imread(imagepath)

  size = (n, n)
  shape = cv2.MORPH_RECT
  kernel = cv2.getStructuringElement(shape, size)

  imgResult = cv2.erode(img, kernel)
  #cv2.namedWindow('Result with n ' + str(n), cv2.WINDOW_NORMAL)  # Adjust the window length
  #cv2.imshow('Result with n ' + str(n), imgResult)

  image_name = str(uuid.uuid4())
  if not os.path.exists(savedir):
    os.mkdir(savedir)
  cv2.imwrite(os.path.join(savedir,"{}.jpg".format(image_name)),imgResult)


#maxfilter
def maximum_filter(imagepath,savedir,n):
  img = cv2.imread(imagepath)

  size = (n,n)
  shape = cv2.MORPH_RECT
  kernel = cv2.getStructuringElement(shape, size)

  imgResult = cv2.dilate(img, kernel)

  #cv2.namedWindow('Result with n ' + str(n), cv2.WINDOW_NORMAL) # Adjust the window length
  #cv2.imshow('Result with n ' + str(n), imgResult)

  image_name = str(uuid.uuid4())
  if not os.path.exists(savedir):
    os.mkdir(savedir)
  cv2.imwrite(os.path.join(savedir,"{}.jpg".format(image_name)),imgResult)


#sharpen
def sharpen_filter(imagepath,savedir):
  img = cv2.imread(imagepath)

  sharpenKernel = np.array(([[0, -1, 0], [-1, 9, -1], [0, -1, 0]]), np.float32) / 9
  # meanBlurKernel = np.ones((3, 3), np.float32)/9

  # gaussianBlur = cv2.filter2D(src=img, kernel=gaussianBlurKernel, ddepth=-1)
  # meanBlur = cv2.filter2D(src=img, kernel=meanBlurKernel, ddepth=-1)
  sharpen = cv2.filter2D(src=img, kernel=sharpenKernel, ddepth=-1)

  horizontalStack = np.concatenate((img, sharpen), axis=1)
  image_name = str(uuid.uuid4())
  if not os.path.exists(savedir):
    os.mkdir(savedir)
  cv2.imwrite(os.path.join(savedir,"{}.jpg".format(image_name)),sharpen)


#removenoise
def remove_noise_filter(imagepath,savedir):
  img = cv2.imread(imagepath)

  kernel = np.ones((5, 5), np.uint8)
  erosion = cv2.erode(img, kernel, iterations=1)
  dilation = cv2.dilate(img, kernel, iterations=1)
  opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
  closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

  image_name = str(uuid.uuid4())
  if not os.path.exists(savedir):
    os.mkdir(savedir)
  cv2.imwrite(os.path.join(savedir,"{}.jpg".format(image_name)),closing)


#medianfilter
def median_filter(imagepath,savedir):
  img = cv2.imread(imagepath)
  img = cv2.imread('C:\\Users\\nurcan\\Desktop\\filters\\images.jpg')
  median = cv2.medianBlur(img, 5)
  compare = np.concatenate((img, median), axis=1)

  #cv2.imshow('img', compare)
  image_name = str(uuid.uuid4())
  if not os.path.exists(savedir):
    os.mkdir(savedir)
  cv2.imwrite(os.path.join(savedir, "{}.jpg".format(image_name)), compare)


#edgedetectbinary
def edge_binary_filter(imagepath,savedir):
  img = cv2.imread(imagepath)
  image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  se = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
  bg = cv2.morphologyEx(image, cv2.MORPH_DILATE, se)
  out_gray = cv2.divide(image, bg, scale=255)
  out_binary = cv2.threshold(out_gray, 255, 255, cv2.THRESH_OTSU)[1]

  image_name = str(uuid.uuid4())
  if  not os.path.exists(savedir):
    os.mkdir(savedir)
  cv2.imwrite(os.path.join(savedir,"{}.jpg".format(image_name)),out_binary)



if __name__ == "__main__":
    #edge_binary_filter("C:\\Users\\nurcan\\Desktop\\filters\\images.jpg", "edge_binary_filter")
    #remove_noise_filter("C:\\Users\\nurcan\\Desktop\\filters\\images.jpg","remove_noise_filter")
    #sharpen_filter("C:\\Users\\nurcan\\Desktop\\filters\\images.jpg", "sharpen_filter")
    #maximum_filter("C:\\Users\\nurcan\\Desktop\\filters\\images.jpg", "max_filter",n=5)
    #minimum_filter("C:\\Users\\nurcan\\Desktop\\filters\\images.jpg", "min_filter",n=5)
    #laplacian_filter("C:\\Users\\nurcan\\Desktop\\filters\\images.jpg", "laplacian_filter")
    #sobel_filter("C:\\Users\\nurcan\\Desktop\\filters\\images.jpg", "sobel_filter")
    #prewitt_filter("C:\\Users\\nurcan\\Desktop\\filters\\images.jpg", "prewitt_filter")
    #canny_filter("C:\\Users\\nurcan\\Desktop\\filters\\images.jpg", "canny_filter")
    #keskinlestirme_filter("C:\\Users\\nurcan\\Desktop\\filters\\images.jpg", "keskinlestirme_filter")
    #enhance_filter("C:\\Users\\nurcan\\Desktop\\filters\\images.jpg", "enhance_filter")
    #average_filter("C:\\Users\\nurcan\\Desktop\\filters\\images.jpg", "average_filter")
    #median_filter("C:\\Users\\nurcan\\Desktop\\filters\\images.jpg", "median_filter")
    #radon_filter("C:\\Users\\nurcan\\Desktop\\filters\\images.jpg", "radon_filter")
    gabor_filter("C:\\Users\\nurcan\\Desktop\\filters\\images.jpg", "gabor_filter")


