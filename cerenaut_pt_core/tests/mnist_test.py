"""Component Integration Test."""

from __future__ import print_function

import json
import argparse

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision import datasets, transforms

from cerenaut_pt_core.components.sparse_autoencoder import SparseAutoencoder


def train(args, model, device, train_loader, optimizer, epoch, writer):
  """Trains the model for one epoch."""
  model.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)
    
    optimizer.zero_grad()
    _, output = model(data)
    loss = F.mse_loss(output, data)
    loss.backward()
    optimizer.step()

    writer.add_image('train/inputs', torchvision.utils.make_grid(data), batch_idx)
    writer.add_image('train/outputs', torchvision.utils.make_grid(output), batch_idx)
    writer.add_scalar('train/loss', loss, batch_idx)

    if batch_idx % args.log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
          epoch, batch_idx * len(data), len(train_loader.dataset),
          100. * batch_idx / len(train_loader), loss.item()))

      if args.dry_run:
        break


def test(model, device, test_loader, writer):
  """Evaluates the trained model."""
  model.eval()
  test_loss = 0

  with torch.no_grad():
    for batch_idx, (data, target) in enumerate(test_loader):
      data, target = data.to(device), target.to(device)
      _, output = model(data)

      writer.add_image('test/inputs', torchvision.utils.make_grid(data), batch_idx)
      writer.add_image('test/outputs', torchvision.utils.make_grid(output), batch_idx)

      test_loss += F.mse_loss(output, data, reduction='sum').item()  # sum up batch loss

    test_loss /= len(test_loader.dataset)

    writer.add_scalar('test/avg_loss', test_loss, 0)

    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))


def main():
  # Training settings
  parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
  parser.add_argument('--config', type=str, default='test_configs/sae.json', metavar='N',
                      help='Model configuration (default: test_configs/sae.json')
  parser.add_argument('--epochs', type=int, default=1, metavar='N',
                      help='Number of training epochs (default: 1)')
  parser.add_argument('--no-cuda', action='store_true', default=False,
                      help='disables CUDA training')
  parser.add_argument('--dry-run', action='store_true', default=False,
                      help='quickly check a single pass')
  parser.add_argument('--seed', type=int, default=1, metavar='S',
                      help='random seed (default: 1)')
  parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                      help='how many batches to wait before logging training status')
  parser.add_argument('--save-model', action='store_true', default=False,
                      help='For Saving the current Model')

  args = parser.parse_args()

  torch.manual_seed(args.seed)

  use_cuda = not args.no_cuda and torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")

  with open(args.config) as config_file:
    config = json.load(config_file)

  kwargs = {'batch_size': config['batch_size']}

  if use_cuda:
    kwargs.update({
        'num_workers': 1,
        'pin_memory': True,
        'shuffle': True
    })

  writer = SummaryWriter()

  transform = transforms.Compose([
      transforms.ToTensor()
  ])

  train_dataset = datasets.MNIST('./data', train=True, download=True,
                                 transform=transform)
  test_dataset = datasets.MNIST('./data', train=False,
                                transform=transform)

  train_loader = torch.utils.data.DataLoader(train_dataset, **kwargs)
  test_loader = torch.utils.data.DataLoader(test_dataset, **kwargs)

  input_shape = [-1, 1, 28, 28]

  if config['model'] == 'sae':
    model = SparseAutoencoder(input_shape, config['model_config']).to(device)
  else:
    raise NotImplementedError('Model not supported: ' + str(config['model']))

  optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

  for epoch in range(1, args.epochs + 1):
    train(args, model, device, train_loader, optimizer, epoch, writer)
    test(model, device, test_loader, writer)

  if args.save_model:
    torch.save(model.state_dict(), "mnist_sae.pt")


if __name__ == '__main__':
    main()
