"""utils.py"""

import os
import math
import random
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def activation_fn(fn_type):
  """Simple switcher for choosing activation functions."""
  if fn_type == 'none':
    fn = lambda x: x
  elif fn_type == 'relu':
    fn = nn.ReLU()
  elif fn_type in ['leaky-relu', 'leaky_relu']:
    fn = nn.LeakyReLU()
  elif fn_type == 'tanh':
    fn = nn.Tanh()
  elif fn_type == 'sigmoid':
    fn = nn.Sigmoid()
  elif fn_type == 'softmax':
    fn = nn.Softmax()
  else:
    raise NotImplementedError(
        'Activation function implemented: ' + str(fn_type))

  return fn


def build_topk_mask(x, dim=1, k=2):
  """
  Simple functional version of KWinnersMask/KWinners since
  autograd function apparently not currently exportable by JIT

  Sourced from Jeremy's RSM code
  """
  res = torch.zeros_like(x)
  _, indices = torch.topk(x, k=k, dim=dim, sorted=False)
  return res.scatter(dim, indices, 1)


def truncated_normal_(tensor, mean=0, std=1):
  size = tensor.shape
  tmp = tensor.new_empty(size + (4,)).normal_()
  valid = (tmp < 2) & (tmp > -2)
  ind = valid.max(-1, keepdim=True)[1]
  tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
  tensor.data.mul_(std).add_(mean)
  return tensor


def xavier_truncated_normal_(tensor, gain=1.0):
  gain = 1.0
  fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(tensor)
  std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
  return truncated_normal_(tensor, mean=0.0, std=std)


def initialize_parameters(m, weight_init='xavier_uniform_', bias_init='zeros_'):
  """Initialize nn.Module parameters."""
  if not isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
    return

  weight_init_fn = get_initializer_by_name(weight_init)

  if m.weight is not None and weight_init_fn is not None:
    weight_init_fn(m.weight)

  bias_init_fn = get_initializer_by_name(bias_init)

  if m.bias is not None:
    bias_init_fn(m.bias)


def get_initializer_by_name(init_type):
  # Handle custom initializers
  if init_type == 'truncated_normal_':
    return lambda x: truncated_normal_(x, mean=0.0, std=0.03)

  if init_type == 'xavier_truncated_normal_':
    return lambda x: xavier_truncated_normal_(x)

  return getattr(torch.nn.init, init_type, None)


def reduce_max(x, dim=0, keepdim=False):
  """
  Performs `torch.max` over multiple dimensions of `x`
  """
  axes = sorted(dim)
  maxed = x
  for axis in reversed(axes):
    maxed, _ = maxed.max(axis, keepdim)
  return maxed

def get_top_k(x, k, mask_type="pass_through", topk_dim=0, scatter_dim=0):
  """Finds the top k values in a tensor, returns them as a tensor.

  Accepts a tensor as input and returns a tensor of the same size. Values
  in the top k values are preserved or converted to 1, remaining values are
  floored to 0 or -1.

      Example:
          >>> a = torch.tensor([1, 2, 3])
          >>> k = 1
          >>> ans = get_top_k(a, k)
          >>> ans
          torch.tensor([0, 0, 3])

  Args:
      x: (tensor) input.
      k: (int) how many top k examples to return.
      mask_type: (string) Options: ['pass_through', 'hopfield', 'binary']
      topk_dim: (int) Which axis do you want to grab topk over? ie. batch = 0
      scatter_dim: (int) Make it the same as topk_dim to scatter the values
  """

  # Initialize zeros matrix
  zeros = torch.zeros_like(x)

  # find top k vals, indicies
  vals, idx = torch.topk(x, k, dim=topk_dim)

  # Scatter vals onto zeros
  top_ks = zeros.scatter(scatter_dim, idx, vals)

  if mask_type != "pass_through":
    # pass_through does not convert any values.

    if mask_type == "binary":
      # Converts values to 0, 1
      top_ks[top_ks > 0.] = 1.
      top_ks[top_ks < 1.] = 0.

    elif mask_type == "hopfield":
      # Converts values to -1, 1
      top_ks[top_ks >= 0.] = 1.
      top_ks[top_ks < 1.] = -1.

    else:
      raise Exception('Valid options: "pass_through", "hopfield" (-1, 1), or "binary" (0, 1)')

  return top_ks

def add_image_noise_flat(image, label=None, minval=0., noise_type='sp_binary', noise_factor=0.2):
  """If the image is flat (batch, size) then use this version. It reshapes and calls the add_imagie_noise()"""
  image_shape = image.shape.as_list()
  image = tf.reshape(image, (-1, image_shape[1], 1, 1))
  image = add_image_noise(image, label, minval, noise_type, noise_factor)
  image = tf.reshape(image, (-1, image_shape[1]))
  return image


def add_image_noise(image, label=None, minval=0., noise_type='sp_binary', noise_factor=0.2):
  image_shape = image.shape.as_list()
  image_size = np.prod(image_shape[1:])

  if noise_type == 'sp_float' or noise_type == 'sp_binary':
    noise_mask = np.zeros(image_size)
    noise_mask[:int(noise_factor * image_size)] = 1
    noise_mask = tf.convert_to_tensor(noise_mask, dtype=tf.float32)
    noise_mask = tf.random_shuffle(noise_mask)
    noise_mask = tf.reshape(noise_mask, [-1, image_shape[1], image_shape[2], image_shape[3]])

    noise_image = tf.random_uniform(image_shape, minval, 1.0)
    if noise_type == 'sp_binary':
      noise_image = tf.sign(noise_image)
    noise_image = tf.multiply(noise_image, noise_mask)  # retain noise in positions of noise mask

    image = tf.multiply(image, (1 - noise_mask))  # zero out noise positions
    corrupted_image = image + noise_image  # add in the noise
  else:
    if noise_type == 'none':
      raise RuntimeWarning("Add noise has been called despite noise_type of 'none'.")
    else:
      raise NotImplementedError("The noise_type '{0}' is not supported.".format(noise_type))

  if label is None:
    return corrupted_image

  return corrupted_image, label


def add_image_salt_noise_flat(image, label=None, noise_val=0., noise_factor=0., mode='add'):
  """If the image is flat (batch, size) then use this version. It reshapes and calls the add_image_noise()"""
  image_shape = image.shape
  image = torch.reshape(image, (-1, image_shape[1], 1, 1))
  image = add_image_salt_noise(image, label, noise_val, noise_factor, mode)
  image = torch.reshape(image, (-1, image_shape[1]))
  return image

def add_image_salt_pepper_noise_flat(image, label=None, salt_val=1., pepper_val=0., noise_factor=0.):
  """If the image is flat (batch, size) then use this version. It reshapes and calls the add_image_noise()"""
  image_shape = image.shape
  image = torch.reshape(image, (-1, image_shape[1], 1, 1))
  image = add_image_salt_noise(image, label, salt_val, noise_factor, 'replace')
  image = add_image_salt_noise(image, label, pepper_val, noise_factor, 'replace')
  image = torch.reshape(image, (-1, image_shape[1]))
  return image

def add_image_salt_noise(image, label=None, noise_val=0., noise_factor=0., mode='add'):
  """ Add salt noise.

  :param image:
  :param label:
  :param noise_val: value of 'salt' (can be +ve or -ve, must be non zero to have an effect)
  :param noise_factor: the proportion of the image
  :param mode: 'replace' = replace existing value, 'add' = noise adds to the existing value
  :return:
  """

  device = image.device

  image_shape = image.shape
  image_size = np.prod(image_shape[1:])

  # random shuffle of chosen number of active bits
  noise_mask = np.zeros(image_size, dtype=np.float32)
  noise_mask[:int(noise_factor * image_size)] = 1
  np.random.shuffle(noise_mask)

  noise_mask = np.reshape(noise_mask, [-1, image_shape[1], image_shape[2], image_shape[3]])
  noise_mask = torch.from_numpy(noise_mask).to(device)

  if mode == 'replace':
    image = image * (1 - noise_mask)  # image: zero out noise positions

  image = image + (noise_mask * noise_val)  # image: add in the noise at the chosen value

  if label is None:
    return image

  return image, label


def square_image_shape_from_1d(filters):
  """
  Make 1d tensor as square as possible. If the length is a prime, the worst case, it will remain 1d.
  Assumes and retains first dimension as batches.
  """
  height = int(math.sqrt(filters))

  while height > 1:
    width_remainder = filters % height
    if width_remainder == 0:
      break
    else:
      height = height - 1

  width = filters // height
  area = height * width
  lost_pixels = filters - area

  shape = [-1, height, width, 1]

  return shape, lost_pixels


def get_padding(kernel_size, stride=1, dilation=1):
  """Calculate symmetric padding for a convolution"""
  padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
  return padding


def get_same_padding(x, k, s, d):
  """Calculate asymmetric TensorFlow-like 'SAME' padding for a convolution"""
  return max((math.ceil(x / s) - 1) * s + (k - 1) * d + 1 - x, 0)


def is_static_pad(kernel_size, stride=1, dilation=11):
  """Can SAME padding for given args be done statically?"""
  return stride == 1 and (dilation * (kernel_size - 1)) % 2 == 0

def pad_same(x, k, s, d=(1, 1), value=0):
  """Dynamically pad input x with 'SAME' padding for conv with specified args"""
  ih, iw = x.size()[-2:]
  pad_h, pad_w = get_same_padding(ih, k[0], s[0], d[0]), get_same_padding(iw, k[1], s[1], d[1])
  padding = [0, 0, 0, 0]
  if pad_h > 0 or pad_w > 0:
    padding = [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
    x = F.pad(x, padding, value=value)
  return x, padding


def get_padding_value(padding, kernel_size):
  """Get TF-compatible padding."""
  dynamic = False
  if isinstance(padding, str):
    # for any string padding, the padding will be calculated for you, one of three ways
    padding = padding.lower()
    if padding == 'same':
      # TF compatible 'SAME' padding, has a performance and GPU memory allocation impact
      if is_static_pad(kernel_size):
        # static case, no extra overhead
        padding = get_padding(kernel_size)
      else:
        # dynamic 'SAME' padding, has runtime/GPU memory overhead
        padding = 0
        dynamic = True
    elif padding == 'valid':
      # 'VALID' padding, same as padding=0
      padding = 0
    else:
      # Default to PyTorch style 'same'-ish symmetric padding
      padding = get_padding(kernel_size)
  return padding, dynamic


def conv2d_same(x, weight, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1):
  """Perform a Conv2D operation using SAME padding."""
  stride = (stride, stride) if isinstance(stride, int) else stride

  x, _ = pad_same(x, weight.shape[-2:], stride, dilation)
  return F.conv2d(x, weight=weight, bias=bias, stride=stride, padding=padding, dilation=dilation,
                  groups=groups)


def conv_transpose2d_same(x, weight, bias=None, stride=(1, 1), padding=(0, 0), output_padding=(0, 0),
                          dilation=(1, 1), groups=1, output_shape=None):
  """Perform a ConvTranspose2D operation using SAME padding."""
  stride = (stride, stride) if isinstance(stride, int) else stride

  padded_x, padding = pad_same(x, weight.shape[-2:], stride, dilation)

  # Note: This is kind of hacky way to figure out the correct padding for the
  # transpose operation, depending on the stride
  if stride[0] == 1 and stride[1] == 1:
    x = padded_x
    padding = [padding[0] + padding[1], padding[2] + padding[3]]
  else:
    padding = [padding[0], padding[2]]

  if output_shape is not None:
    out_h = output_shape[2]
    out_w = output_shape[3]

    # Compute output shape
    h = (x.shape[2] - 1) * stride[0] + weight.shape[2] - 2 * padding[0]
    w = (x.shape[3] - 1) * stride[1] + weight.shape[3] - 2 * padding[1]

    output_padding = (out_h - h, out_w - w)

  return F.conv_transpose2d(x, weight=weight, bias=bias, stride=stride, padding=padding, output_padding=output_padding,
                            dilation=dilation, groups=groups)


def max_pool2d_same(x, kernel_size, stride, padding=(0, 0), dilation=(1, 1), ceil_mode=False):
  """Perform MaxPool2D operation using SAME padding."""
  kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
  stride = (stride, stride) if isinstance(stride, int) else stride

  x, _= pad_same(x, kernel_size, stride, value=-float('inf'))
  return F.max_pool2d(x, kernel_size, stride, padding, dilation, ceil_mode)
