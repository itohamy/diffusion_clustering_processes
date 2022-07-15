import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
import random
import numpy as np
from absl import app
from matplotlib import pyplot as plt
from PIL import Image


def main(argv):
  
  B = 3 # batch_size
  nc = 1
  img_sz = 28

  train_dataset, nlabels = get_dataset(is_train=True, uniform_dequantization=False)
  batch_sampler = MultipleDatasetsBatchSampler(train_dataset.targets, B)
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=10, pin_memory=True)
  train_iter = iter(train_loader)

  x_next, y_next = next(train_iter)
  print(x_next.size())
  print(y_next.size())
  print('y: ', y_next)

  x_reshaped = torch.reshape(x_next, (B, -1, nc, img_sz, img_sz))
  y_reshaped = torch.reshape(y_next, (B, -1))
  print(x_reshaped.size())
  print(y_reshaped.size())

  print('---')
  print(y_reshaped[0])
  print('\n')
  print(y_reshaped[1])
  print('\n')
  print(y_reshaped[2])
  img1 = Image.fromarray((np.squeeze(np.moveaxis(x_reshaped[2,0].detach().cpu().numpy(), 0, -1)) * 255).astype(np.uint8))
  img1.save("img1.jpeg")
  img2 = Image.fromarray((np.squeeze(np.moveaxis(x_reshaped[2,1].detach().cpu().numpy(), 0, -1)) * 255).astype(np.uint8))
  img2.save("img2.jpeg")
  img3 = Image.fromarray((np.squeeze(np.moveaxis(x_reshaped[2,2].detach().cpu().numpy(), 0, -1)) * 255).astype(np.uint8))
  img3.save("img3.jpeg")
  img4 = Image.fromarray((np.squeeze(np.moveaxis(x_reshaped[2,3].detach().cpu().numpy(), 0, -1)) * 255).astype(np.uint8))
  img4.save("img4.jpeg")
  img5 = Image.fromarray((np.squeeze(np.moveaxis(x_reshaped[2,4].detach().cpu().numpy(), 0, -1)) * 255).astype(np.uint8))
  img5.save("img5.jpeg")
  print('\n')

  return

  for x_next, y_next in train_loader:
      print(x_next.size())
      print(y_next.size())

      print('y: ', y_next)

      x_reshaped = torch.reshape(x_next, (B, -1, nc, img_sz, img_sz))
      y_reshaped = torch.reshape(y_next, (B, -1))
      print(x_reshaped.size())
      print(y_reshaped.size())

      print('---')
      print(y_reshaped[0])
      print('\n')
      print(y_reshaped[1])
      print('\n')
      print(y_reshaped[2])
      img1 = Image.fromarray((np.squeeze(np.moveaxis(x_reshaped[2,0].detach().cpu().numpy(), 0, -1)) * 255).astype(np.uint8))
      img1.save("img1.jpeg")
      img2 = Image.fromarray((np.squeeze(np.moveaxis(x_reshaped[2,1].detach().cpu().numpy(), 0, -1)) * 255).astype(np.uint8))
      img2.save("img2.jpeg")
      img3 = Image.fromarray((np.squeeze(np.moveaxis(x_reshaped[2,2].detach().cpu().numpy(), 0, -1)) * 255).astype(np.uint8))
      img3.save("img3.jpeg")
      img4 = Image.fromarray((np.squeeze(np.moveaxis(x_reshaped[2,3].detach().cpu().numpy(), 0, -1)) * 255).astype(np.uint8))
      img4.save("img4.jpeg")
      img5 = Image.fromarray((np.squeeze(np.moveaxis(x_reshaped[2,4].detach().cpu().numpy(), 0, -1)) * 255).astype(np.uint8))
      img5.save("img5.jpeg")
      print('\n')

      break


def get_dataset(is_train=True, uniform_dequantization=False, transform=None):

  data_name = 'MNIST'
  deterministic = False
  size = 28
  data_dir = 'data/MNIST'

  transform = transforms.Compose([
      t for t in [
          transforms.Resize(size),
          transforms.CenterCrop(size),
          (not deterministic) and transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
          (not deterministic) and
          transforms.Lambda(lambda x: x + 1. / 128 * torch.rand(x.size())),
      ] if t is not False
  ]) if transform == None else transform

  if data_name == 'CIFAR10':
      dataset = datasets.CIFAR10(root=data_dir,
                                  train=is_train,
                                  download=True,
                                  transform=transform)
      nlabels = 10

  elif data_name == 'MNIST':
      dataset = datasets.MNIST(data_dir,
                                transform=transforms.Compose([
                                  transforms.Resize(size),
                                  transforms.CenterCrop(size),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5, ), (0.5, ))
                              ]),
                                train=is_train,
                                download=True)
      nlabels = 10
  
  else:
      raise NotImplemented

  return dataset, nlabels


class MultipleDatasetsBatchSampler():
    """
    MultipleDatasetsBatchSampler:
     - Draw a small dataset size: N ~ p(N)
     - Draw labels: C ~ Categorial(pi) where pi ~ Dir(np.ones(K))
     - Draw data points from the dataset according to the labels in C.
    Returns batches of size [B=batch_sz, N, D] where N and D are fixed values for one certain batch.
    Taken from https://github.com/adambielski/siamese-triplet
    """

    def __init__(self, labels, B):
        self.N_min = 5
        self.N_max = 50
        self.B = B      
        self.labels = labels  # shape: [number of points in the whole dataset,], contains the labels of each point in the dataset according to the order the in the dataset.
        self.labels_set = list(set(self.labels.numpy()))
        self.K = len(self.labels_set)
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])

        self.used_label_indices_count = {label: 0 for label in self.labels_set} # save the last index that was used in "label_to_indices" per label.
        self.count = 0
        self.n_dataset = len(self.labels)

    def __iter__(self):
      self.count = 0

      while self.count + self.B < self.n_dataset:
        N = random.randint(self.N_min, self.N_max)
        print('N: ', N)
        indices = []  # will be in shape: [B*N] and will include B sets of indices, each set is of size N.

        for i in range(self.B):
          # For each B_i, create indices that will form the datapoints of the small dataset in entry i in the batch:
          dirichlet_sample = np.random.dirichlet(np.ones(self.K))
          multinomial_sample = np.random.multinomial(N, dirichlet_sample) # shape [K,] contains the count of datapoints to sample per label. Sum(multinomial_sample) = N.
          print('multinomial_sample: ', multinomial_sample)

          # # For tests:
          # if i==0:
          #   multinomial_sample = np.array([0,1,0,0,0,1,1,0,0,2]) # shape [K,] contains the count of datapoints to sample per label. Sum(multinomial_sample) = N.
          # elif i==1:
          #   multinomial_sample = np.array([1,1,0,2,0,0,1,0,0,0]) # shape [K,] contains the count of datapoints to sample per label. Sum(multinomial_sample) = N.
          # elif i==2:
          #   multinomial_sample = np.array([0,1,0,0,1,0,2,0,1,0]) # shape [K,] contains the count of datapoints to sample per label. Sum(multinomial_sample) = N.

          B_i_indices = []
          for j in range(multinomial_sample.shape[0]):
            n = multinomial_sample[j]
            lbl = self.labels_set[j]

            if self.used_label_indices_count[lbl] + n > len(self.label_to_indices[lbl]):
              np.random.shuffle(self.label_to_indices[lbl])
              self.used_label_indices_count[lbl] = 0

            B_i_indices.extend(self.label_to_indices[lbl][self.used_label_indices_count[lbl]:self.used_label_indices_count[lbl] + n])
            self.used_label_indices_count[lbl] += n

          np.random.shuffle(B_i_indices)
          indices.extend(B_i_indices)
        
        yield indices
        break
        self.count += self.B * N

    #def __len__(self):
    #    return self.n_dataset // self.B


if __name__ == "__main__":
  app.run(main)