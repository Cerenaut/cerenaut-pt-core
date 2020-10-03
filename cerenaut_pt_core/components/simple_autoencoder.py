"""FastNN module."""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from cerenaut_pt_core import utils


class SimpleAutoencoder(nn.Module):
  """A simple encoder-decoder network."""

  def __init__(self, input_shape, config, output_shape=None):
    super(SimpleAutoencoder, self).__init__()

    self.input_shape = list(input_shape)
    self.input_size = np.prod(self.input_shape[1:])

    if output_shape is None:
      self.output_shape = self.input_shape
      self.output_size = self.input_size
    else:
      self.output_shape = list(output_shape)
      self.output_size = np.prod(self.output_shape[1:])

    self.input_shape[0] = -1
    self.output_shape[0] = -1

    self.config = config

    self.build()

  def build(self):
    """Build the network architecture."""
    self.encoder = nn.Linear(self.input_size, self.config['num_units'], bias=self.config['use_bias'])
    self.decoder = nn.Linear(self.config['num_units'], self.output_size, bias=self.config['use_bias'])

    self.encoder_nonlinearity = utils.activation_fn(self.config['encoder_nonlinearity'])
    self.decoder_nonlinearity = utils.activation_fn(self.config['decoder_nonlinearity'])

    self.reset_parameters()

  def reset_parameters(self):
    self.apply(lambda m: utils.initialize_parameters(m, weight_init='xavier_truncated_normal_', bias_init='zeros_'))

  def add_noise(self, inputs):
    """Adds noise (salt, salt+pepper) to the inputs"""
    noise_type = self.config.get('noise_type', None)
    noise_mode = self.config.get('noise_mode', 'add')

    if self.training:
      noise_val = self.config.get('train_with_noise', 0.0)
      noise_factor = self.config.get('train_with_noise_pp', 0.0)
    else:
      noise_val = self.config.get('test_with_noise', 0.0)
      noise_factor = self.config.get('test_with_noise_pp', 0.0)

    if noise_type == 's':  # salt noise
      return utils.add_image_salt_noise_flat(inputs, noise_val=noise_val, noise_factor=noise_factor, mode=noise_mode)

    if noise_type == 'sp':  # salt + pepper noise
      # Inspired by denoising AE.
      # Add salt+pepper noise to mimic missing/extra bits in PC space.
      # Use a fairly high rate of noising to mitigate few training iters.
      return utils.add_image_salt_pepper_noise_flat(inputs,
                                                    salt_val=noise_val,
                                                    pepper_val=-noise_val,
                                                    noise_factor=noise_factor)

    return inputs

  def encode(self, inputs):
    inputs = torch.flatten(inputs, start_dim=1)

    encoding = self.encoder(inputs)
    encoding = self.encoder_nonlinearity(encoding)

    if self.config['hidden_dropout'] > 0:
      encoding = F.dropout(encoding, p=self.config['hidden_dropout'], training=self.training)

    return encoding

  def decode(self, encoding):
    decoding = self.decoder(encoding)
    decoding = self.decoder_nonlinearity(decoding)

    decoding = torch.reshape(decoding, self.output_shape)

    return decoding

  def forward(self, x):  # pylint: disable=arguments-differ
    x = torch.flatten(x, start_dim=1)

    # Normalize the inputs
    if self.config.get('norm_inputs'):
      x = (x - x.min()) / (x.max() - x.min())

    # Optionally add noise to the inputs
    x = self.add_noise(x)

    if self.config['input_dropout'] > 0:
      x = F.dropout(x, p=self.config['input_dropout'], training=self.training)

    encoding = self.encode(x)
    decoding = self.decode(encoding)

    return encoding, decoding
