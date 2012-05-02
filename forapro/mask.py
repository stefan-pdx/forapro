import numpy

def radial(lambda_r=10, n=10, k=5, **kwargs):
  # Distance from center of image.
  y_dist,x_dist = [numpy.cumsum(numpy.ones(2*lambda_r))  - (2*lambda_r+1)/ 2.0 for _ in range(2)]

  # Determine the x/y distances for each point in the mask.
  x,y = numpy.meshgrid(x_dist, y_dist)
  r = numpy.sqrt(x*x+y*y)

  # Extract out our radial masks.
  masks = numpy.array([numpy.sqrt(numpy.max(r * (lambda_r - r), 0)) * numpy.exp(1j * k * numpy.arctan2(y,x)) for k in range(1,k+1)])

  # Normalize the weights
  masks /= numpy.abs(masks)

  return numpy.array(masks)

def circular(lambda_r=10, l=5, **kwargs):
  # Distance from center of image.
  y_dist,x_dist = [numpy.cumsum(numpy.ones(2*lambda_r))  - (2*lambda_r+1) / 2.0 for _ in range(2)]

  # Determine the x/y distances for each point in the mask.
  x,y = numpy.meshgrid(x_dist, y_dist)

  # Iterate through the difference pixel distances and build masks.
  masks = [1/(2*numpy.pi*numpy.sqrt(x*x+y*y)) * numpy.exp(l * numpy.sqrt(x*x+y*y) * 1j / lambda_r) for l in range(1,l+1)]

  # Account for r=0.
  masks = [numpy.where(x*x+y*y>0, mask, 0.73) for mask in masks]

  # Normalize the weights
  masks = numpy.array(masks)
  masks /= numpy.abs(masks)

  return masks
