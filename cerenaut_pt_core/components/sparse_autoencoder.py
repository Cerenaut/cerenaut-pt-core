"""SparseAutoencoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import utils


class SparseAutoencoder(nn.Module):
  """A convolutional k-Sparse autoencoder."""

  def __init__(self, input_shape, config, output_shape=None):
    super(SparseAutoencoder, self).__init__()

    self.input_shape = list(input_shape)
    self.input_size = np.prod(self.input_shape[1:])
    self.config = config

    if output_shape is None:
      self.output_shape = self.input_shape
      self.output_size = self.input_size
    else:
      self.output_shape = list(output_shape)
      self.output_size = np.prod(self.output_shape[1:])

    self.input_shape[0] = -1
    self.output_shape[0] = -1

    self.build()

    self.encoder_nonlinearity = utils.activation_fn(self.config['encoder_nonlinearity'])
    self.decoder_nonlinearity = utils.activation_fn(self.config['decoder_nonlinearity'])

  def reset_parameters(self):
    # self.apply(lambda m: utils.initialize_parameters(m, weight_init='xavier_normal_', bias_init='zeros_'))

    # Similar initialization to TF implementation of ConvAEs
    def custom_weight_init(m):
      if not isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        return

      if m.weight is not None:
        utils.truncated_normal_(m.weight, mean=0.0, std=0.03)

      if m.bias is not None:
        torch.nn.init.zeros_(m.bias)

    self.apply(custom_weight_init)

  def build(self):
    self.encoder = nn.Conv2d(self.input_shape[1], self.config['filters'],
                             kernel_size=self.config['kernel_size'],
                             stride=self.config['stride'],
                             bias=self.config['use_bias'],
                             padding=self.config['encoder_padding'])

    self.decoder = nn.ConvTranspose2d(self.config['filters'], self.input_shape[1],
                                      kernel_size=self.config['kernel_size'],
                                      stride=self.config['stride'],
                                      bias=self.config['use_bias'],
                                      padding=self.config['decoder_padding'])

    self.reset_parameters()

  def encode(self, inputs, stride=None):
    stride = stride if stride is not None else self.config['stride']

    if self.config['encoder_padding'] == 'same':
      encoding = utils.conv2d_same(inputs, self.encoder.weight,
                                   bias=self.encoder.bias,
                                   stride=(stride, stride))
    else:
      encoding = F.conv2d(inputs, self.encoder.weight,
                          bias=self.encoder.bias,
                          stride=stride,
                          padding=self.config['encoder_padding'])

    encoding = self.encoder_nonlinearity(encoding)

    return encoding

  def filter(self, encoding):
    """Build filtering/masking for specified encoding."""
    encoding_nhwc = encoding.permute(0, 2, 3, 1)  # NCHW => NHWC

    top_k_input = encoding_nhwc

    # Find the "winners". The top k elements in each batch sample. this is
    # what top_k does.
    # ---------------------------------------------------------------------
    k = int(self.config['sparsity'])

    if not self.training:
      k = int(k * self.config['sparsity_output_factor'])

    top_k_mask = utils.build_topk_mask(top_k_input, dim=-1, k=k)

    # Retrospectively add batch-sparsity per cell: pick the top-k (for now k=1 only).
    # ---------------------------------------------------------------------
    if self.training and self.config['use_lifetime_sparsity']:
      batch_max = utils.reduce_max(top_k_input, dim=[0, 1, 2], keepdim=True)  # input shape: batch,cells, output shape: cells
      batch_mask_bool = top_k_input >= batch_max # inhibit cells (mask=0) until spike has decayed
      batch_mask = batch_mask_bool.float()

      either_mask = torch.max(top_k_mask, batch_mask) # logical OR, i.e. top-k or top-1 per cell in batch
    else:
      either_mask = top_k_mask

    filtered_encoding_nhwc = encoding_nhwc * either_mask  # Apply mask 3 to output 2
    filtered_encoding = filtered_encoding_nhwc.permute(0, 3, 1, 2)  # NHWC => NCHW

    return filtered_encoding

  def decode(self, encoding, stride=None):
    stride = stride if stride is not None else self.config['stride']

    decoder_weight = self.decoder.weight

    if self.config['use_tied_weights']:
      decoder_weight = self.encoder.weight

    if self.config['decoder_padding'] == 'same':
      decoding = utils.conv_transpose2d_same(encoding, decoder_weight,
                                             bias=self.decoder.bias,
                                             stride=stride,
                                             output_shape=self.output_shape)
    else:
      decoding = F.conv_transpose2d(encoding, decoder_weight,
                                    bias=self.decoder.bias,
                                    stride=stride,
                                    padding=self.config['decoder_padding'])

    decoding = self.decoder_nonlinearity(decoding)
    decoding = torch.reshape(decoding, self.output_shape)

    return decoding

  def forward(self, x, stride=None):  # pylint: disable=arguments-differ
    encoding = self.encode(x, stride)

    if self.config['sparsity'] > 0:
      encoding = self.filter(encoding)

    decoding = self.decode(encoding, stride)

    return encoding, decoding
