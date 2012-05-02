import numpy

import util

class FeatureVector(object):
  """ Our primary feature vector which consists of radial magnitudes, radial
      angles and circular features. """

  k = None
  l = None
  radial_magnitudes = None
  radial_angles = None
  circular_features = None

  def __init__(self, image):
    self.k = image.radial_coefficients.shape[0] 
    self.l = image.circular_coefficients.shape[0] 

    self.radial_magnitudes = RadialMagnitudes(image)
    self.radial_angles = RadialAngles(image)
    self.circular_features = CircularFeatures(image)

  def __sub__(self, other):
    w_m = w_a = self.k - 1
    w_c = 2 * self.l - 1
    w_t = w_m + w_a + w_c
    
    return (w_m / w_t) * (self.radial_magnitudes - other.radial_magnitudes) + \
           (w_a / w_t) * (self.radial_angles - other.radial_angles) + \
           (w_c / w_t) * (self.circular_features - other.circular_features)

class RadialMagnitudes(object):
  data = None

  def __init__(self, image):
    self.data = numpy.abs(image.radial_coefficients)

  def __sub__(self, other):
    return 1/2. * numpy.sum(self.data-other.data, axis=0)

class RadialAngles(object):
  r_k = None
  data = None

  def __init__(self, image):
    self.r_k = numpy.arctan2(image.radial_coefficients.imag, image.radial_coefficients.real)
    self.data = numpy.mod(self.r_k[1:,:,:] -
        numpy.array([ k * numpy.ones(self.r_k.shape[1:]) for k in range(2,self.r_k.shape[0]+1)]) *
        numpy.repeat(numpy.expand_dims(self.r_k[0,:,:], axis=0), self.r_k.shape[0]-1,axis=0), 2*numpy.pi)
    
  def __sub__(self, other):
    w = 1/numpy.array([ k * numpy.ones(self.data.shape[1:]) for k in range(2,self.data.shape[0]+2)])
    wt = numpy.repeat(numpy.expand_dims(numpy.sum(w, axis=0), axis=0), w.shape[0], axis=0)
    return numpy.sum(w/(wt * numpy.pi) * util.phi(self.data, other.data), axis=0)

class CircularFeatures(object):
  data = None

  def __init__(self, image):
    # Flatten our circular coefficients to a (x,y,2*k) matrix.
    data = numpy.array([image.circular_coefficients.real,
      image.circular_coefficients.imag]).transpose(2,3,1,0)

    self.data = data.reshape(data.shape[:2]+(numpy.product(data.shape[2:]),))

  def __sub__(self, other):
    return 1/2. * numpy.sum(self.data-other.data, axis=2)
