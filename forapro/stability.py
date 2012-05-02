import functools
import numpy

import util

def td_tm_stable(reference_image, image_neighborhood, td=5000, tm=30000):

  # Calculate the difference between the reference image and neighboring images in terms of
  # the radial magnitudes feature vector.
  rm_dist = numpy.vectorize(lambda neighbor: reference_image.feature_vector.radial_magnitudes - neighbor.feature_vector.radial_magnitudes, otypes='O')(image_neighborhood)

  # Extract out the center pixel for each neighborhood image.
  rm_dist = numpy.vectorize(lambda x: x[x.shape[0]/2, x.shape[1]/2])(rm_dist)

  r_k = reference_image.radial_coefficients[:, reference_image.radial_coefficients.shape[1]/2, reference_image.radial_coefficients.shape[2]/2]

  return (numpy.max(rm_dist) < td) & (numpy.sum(numpy.abs(r_k)) > tm) 

def ta_tm_stable(reference_image, image_neighborhood, ta=numpy.deg2rad(10), tm=0.01):

  # Extract out the r_k vectors for each image in the neighborhood.
  neighborhood_rk = util.get_vectorized_attr(image_neighborhood,
      ['feature_vector', 'radial_angles', 'r_k'])

  # Extract out the canonical orientation of our neighborhood. Note we're taking center pixel.
  neighborhood_r1 = numpy.vectorize(lambda x: x[x.shape[0]/2, x.shape[1]/2], otypes='d')(util.get_vectorized_index(neighborhood_rk, [0]))

  # Extract out the canonical orientation of our reference image.
  reference_r1 = reference_image.feature_vector.radial_angles.r_k[0]
  reference_r1 = reference_r1[reference_r1.shape[0]/2, reference_r1.shape[1]/2]

  # Calculate phi between our reference image and neighborhood.
  phi = numpy.vectorize(util.phi)(neighborhood_r1, reference_r1)

  # Return the phi result.
  return (numpy.max(phi, axis=0) < ta) & (numpy.abs(reference_r1) >= tm)
