import cv
import numpy
import numpy.lib.stride_tricks
import scipy.ndimage.interpolation

import mask
import feature_vector

class Image(object):
  sub_images = None
  neighborhood_sub_images = None
  radial_coefficients = None
  circular_coefficients = None

  def __init__(self, file_data, **kwargs):

    # Load in the data by file name or by data structure.
    if type(file_data) is str:
      self.image = numpy.asarray(cv.LoadImageM(file_data, cv.CV_LOAD_IMAGE_GRAYSCALE))
    elif type(file_data) is numpy.ndarray:
      self.image = file_data.copy()
    else:
      raise ValueError("Constructor argument must be path to image file or Numpy ndarray.")

    # Extract out our coefficients.
    self.radial_coefficients, self.circular_coefficients = self.get_coefficients(**kwargs)

    # Construct our vector of features.
    self.feature_vector = feature_vector.FeatureVector(self)

  def get_sub_images(self, size=60):

    # Subimage structure shape.
    structure_shape = (self.image.shape[0]-size+1,self.image.shape[1]-size+1)

    # Create the empty array for our sub-images.
    self.sub_images = numpy.empty(structure_shape, dtype=object)

    # Extract out our sub-images of shape p-by-p.
    sub_images = numpy.lib.stride_tricks.as_strided(self.image,
      shape=(self.image.shape[0]-size+1,self.image.shape[1]-size+1,size,size),
      strides=(1,self.image.shape[1],1,1))

    # Capture our sub_images as an array of objects
    sub_images = numpy.array([[None]+[sub_img for row in sub_images
      for sub_img in row]])[:,1:].reshape(structure_shape)

    # Stride through our image and extract out p-by-p subimages.
    return numpy.vectorize(Image)(sub_images)

  def get_coefficients(self, **kwargs):

    # Create a [k,x,y]-list containing our radial and circular masks.
    masks = [mask.radial(**kwargs), mask.circular(**kwargs)]

    # Prepare a data structure to store our coefficients.
    coefficients = [numpy.empty(m.shape[:1] + self.image.shape, dtype=numpy.complex) for m in masks]
    
    for mask_type,mask_coefficients in zip(masks, coefficients):
      for m, coefficient in zip(mask_type, mask_coefficients):

        # Create storage arrays for our real and imaginary components.
        real_coef = cv.CreateMat(self.image.shape[0], self.image.shape[1], cv.CV_64FC1)
        imag_coef = cv.CreateMat(self.image.shape[0], self.image.shape[1], cv.CV_64FC1)

        # Filter our image using our masks.
        cv.Filter2D(cv.fromarray(self.image.astype(numpy.float64)), real_coef, cv.fromarray(m.real.copy()))
        cv.Filter2D(cv.fromarray(self.image.astype(numpy.float64)), imag_coef, cv.fromarray(m.real.copy()))

        # Store our coefficients.
        coefficient[:] = numpy.asarray(real_coef) + 1j * numpy.asarray(imag_coef)

    return coefficients
