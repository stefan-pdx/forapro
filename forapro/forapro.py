import numpy
import scipy.spatial.distance

import image
import stability

class Forapro(object):
  target_image = None
  query_image = None

  def __init__(self, query_image_path, target_image_path):
    self.target_image = image.Image(target_image_path, k=5, l=5, lambda_r=30)
    self.query_image = image.Image(query_image_path, k=5, l=5, lambda_r=30)

  #TODO: normalized cross-correlation
  #TODO: Hough transforms

  def subtemplates_features_candidates(self, nc=3):
    # Extract out sub-images from the query image that are going to be
    # potential templates.
    self.query_image.sub_images = self.query_image.get_sub_images()

    # Build an (x,y) coordinate system for sub-images. Note we collapse it down
    # to m**2-by-m**2 so that we can pass it into cdist.
    sub_image_index = numpy.mgrid[0:self.query_image.sub_images.shape[0],0:self.query_image.sub_images.shape[1]].transpose(1,2,0).reshape(*(numpy.prod(self.query_image.sub_images.shape), 2))

    # Calculate the distance between any two pixels.
    distance_matrix = scipy.spatial.distance.cdist(sub_image_index, sub_image_index).reshape(2*self.query_image.sub_images.shape)

    # Build a "neighborhood" matrix. This is a (m,m,m,m,1) matrix
    # where (p0_y, p0_x, p1_y, p1_x) gives you a boolean value
    # whether p1 is considered a neighbor of p0
    neighborhoods = numpy.where(distance_matrix < 10, self.query_image.sub_images, None)

    # Create a list of stable subtemplates that we can use.
    stable_templates = []

    # Iterate through each sub-image and determine if it's stable.
    for sub_image_row, neighborhood_row in zip(self.query_image.sub_images, neighborhoods):
      for sub_image, neighborhood in zip(sub_image_row, neighborhood_row):

        # Check to see if the pixel is stable given it's neighbors
        neighbors = neighborhood.flatten()[numpy.where(neighborhood.flatten())]

        if stability.ta_tm_stable(sub_image, neighbors) & stability.td_tm_stable(sub_image, neighbors):
          stable_templates.append(sub_image)

    # For each stable template, calculate the "matching distance images".
    mdi = numpy.array([self.target_image.feature_vector - template.feature_vector for template in stable_templates])

    # Take the center value for each image.
    mdi = mdi[:, mdi.shape[1]/2, mdi.shape[2]/2]

    # Return stable teamples, top nc templates, and the matching distance images
    return stable_templates, stable_templates[numpy.argmax(mdi)], mdi
