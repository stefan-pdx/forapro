import numpy

def phi(a_arr, b_arr, axis=0):
  return numpy.min(numpy.array([numpy.mod(a_arr-b_arr, 2*numpy.pi),
    2*numpy.pi - numpy.mod(a_arr-b_arr, 2*numpy.pi)]), axis=axis)

def get_vectorized_attr(obj, field_list):
  """ Recursively iterate through obj to full out the fields listed in field_list.
      Provides a way to dive into numpy object arrays. """

  obj = numpy.vectorize(lambda x: getattr(x,field_list[0]), otypes='O')(obj)

  for field in field_list[1:]:
    obj = numpy.vectorize(lambda x: getattr(x,field), otypes='O')(obj)

  return obj

def get_vectorized_index(obj, indices):
  """ Recursively iterate through obj to full out the indices provided.
      Provides a way to dive into numpy object arrays. """

  obj = numpy.vectorize(lambda x: x[indices[0]], otypes='O')(obj)

  for index in indices[1:]:
    obj = numpy.vectorize(lambda x: x[index], otypes='O')(obj)

  return obj
