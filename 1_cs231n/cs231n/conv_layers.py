import numpy as np


def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) where x[i] is the ith input.
  We multiply this against a weight matrix of shape (D, M) where
  D = \prod_i d_i

  Inputs:
  x - Input data, of shape (N, d_1, ..., d_k)
  w - Weights, of shape (D, M)
  b - Biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None
  out = x.reshape([x.shape[0], -1]).dot(w) + b
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None
  db = dout.sum(axis=0)
  dw = x.reshape([x.shape[0], -1]).T.dot(dout)
  dx = dout.dot(w.T).reshape(x.shape)
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  out = np.maximum(0, x)
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  dx = dout * (x > 0)
  return dx

def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width WW.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  strd, pad = conv_param['stride'], conv_param['pad']
  x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
  N, C, H, W = x.shape
  F, _, HH, WW = w.shape
  H_prime = np.int(1 + (H+2*pad-HH)/strd)
  W_prime = np.int(1 + (W+2*pad-WW)/strd)
  out = np.zeros(shape=(N, F, H_prime, W_prime))

  for ni in range(N):
    for fi in range(F):
      for row in range(0, H+2*pad-HH+1, strd):
        for col in range(0, W+2*pad-WW+1, strd):
          x_padded_local = x_padded[ni, :, row:row+HH, col:col+WW]
          filter_local = w[fi, :, :, :]
          out[ni, fi, np.int(row/strd), np.int(col/strd)] = np.sum(np.multiply(x_padded_local, filter_local)) + b[fi]

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  x, w, b, conv_param = cache
  N, C, H, W = x.shape
  F, _, HH, WW = w.shape
  strd, pad = conv_param['stride'], conv_param['pad']
  x_padded = np.pad(x, ((0,0), (0,0), (pad, pad), (pad, pad)), mode='constant')
  _, _, H_prime, W_prime = dout.shape
  dx = np.zeros(shape = x.shape)
  dx_padded = np.zeros(shape = x_padded.shape)
  dw = np.zeros(shape = w.shape)
  db = np.zeros(shape = b.shape)

  for ni in range(N):
    for fi in range(F):
      for row in range(0, H+2*pad-HH+1, strd):
        for col in range(0, W+2*pad-WW+1, strd):
          dx_padded[ni, :, row:row+HH, col:col+WW] += dout[ni, fi, np.int(row/strd), np.int(col/strd)] * w[fi, :, :, :]
          dw[fi, :, :, :] += dout[ni, fi, np.int(row/strd), np.int(col/strd)] * x_padded[ni, :, row:row+HH, col:col+WW]
          db[fi] += dout[ni, fi, np.int(row/strd), np.int(col/strd)]
  
  dx = dx_padded[:,:,pad:pad+H,pad:pad+W]
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  p_height = pool_param['pool_height']
  p_width = pool_param['pool_width']
  strd = pool_param['stride']
  N, C, H, W = x.shape
  H_prime = np.int(1 + (H-p_height)/strd)
  W_prime = np.int(1 + (W-p_width)/strd)
  out = np.zeros(shape = (N,C,H_prime,W_prime))

  for ni in range(N):
    for ci in range(C):
      for hi in range(0, H_prime):
        for wi in range(0, W_prime):
          out[ni, ci, hi, wi] = np.max(x[ni, ci, strd*hi:strd*hi+p_height, strd*wi:strd*wi+p_width])

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  x, pool_param = cache
  p_height = pool_param['pool_height']
  p_width = pool_param['pool_width']
  strd = pool_param['stride']
  N, C, H, W = x.shape
  H_prime = np.int(1 + (H-p_height)/strd)
  W_prime = np.int(1 + (W-p_width)/strd)
  dx = np.zeros(shape = x.shape)

  for ni in range(N):
    for ci in range(C):
      for hi in range(0, H_prime):
        for wi in range(0, W_prime):
          x_local = x[ni, ci, strd*hi:strd*hi+p_height, strd*wi:strd*wi+p_width]
          org_indx = np.unravel_index(x_local.argmax(), x_local.shape)
          dx[ni, ci, strd*hi:strd*hi+p_height, strd*wi:strd*wi+p_width][org_indx] = dout[ni, ci, hi, wi]

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx

def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx
