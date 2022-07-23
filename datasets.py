# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Return training and evaluation/test datasets from config files."""
import jax
import tensorflow as tf
import tensorflow_datasets as tfds

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
import random
import numpy as np


def get_data_scaler(config):
  """Data normalizer. Assume data are always in [0, 1]."""
  if config.data.centered:
    # Rescale to [-1, 1]
    return lambda x: x * 2. - 1.
  else:
    return lambda x: x


def get_data_inverse_scaler(config):
  """Inverse data normalizer."""
  if config.data.centered:
    # Rescale [-1, 1] to [0, 1]
    return lambda x: (x + 1.) / 2.
  else:
    return lambda x: x


def crop_resize(image, resolution):
  """Crop and resize an image to the given resolution."""
  crop = tf.minimum(tf.shape(image)[0], tf.shape(image)[1])
  h, w = tf.shape(image)[0], tf.shape(image)[1]
  image = image[(h - crop) // 2:(h + crop) // 2,
          (w - crop) // 2:(w + crop) // 2]
  image = tf.image.resize(
    image,
    size=(resolution, resolution),
    antialias=True,
    method=tf.image.ResizeMethod.BICUBIC)
  return tf.cast(image, tf.uint8)


def resize_small(image, resolution):
  """Shrink an image to the given resolution."""
  h, w = image.shape[0], image.shape[1]
  ratio = resolution / min(h, w)
  h = tf.round(h * ratio, tf.int32)
  w = tf.round(w * ratio, tf.int32)
  return tf.image.resize(image, [h, w], antialias=True)


def central_crop(image, size):
  """Crop the center of an image to the given size."""
  top = (image.shape[0] - size) // 2
  left = (image.shape[1] - size) // 2
  return tf.image.crop_to_bounding_box(image, top, left, size, size)


# ------------- create dataset using pytorch, implement new batchSampler-----------------------

def get_dataset(config, is_train=True, uniform_dequantization=False):

  data_name = config.data.dataset
  deterministic = False
  size = config.data.image_size
  data_dir = config.data.data_dir

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
  
  elif data_name == 'FASHIONMNIST':
      dataset = datasets.FashionMNIST(data_dir,
                                transform=transforms.Compose([
                                  transforms.Resize(size),
                                  transforms.CenterCrop(size),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5, ), (0.5, ))
                              ]),
                                train=is_train,
                                download=True)
      nlabels = 10

  elif data_name == 'STL':
      dataset = datasets.STL10(data_dir, 'unlabeled', 
                              train=is_train, 
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                              ]))
      nlabels = len(dataset.classes)
  
  elif data_name == 'LSUN':
      if lsun_categories is None:
          lsun_categories = 'train'
      dataset = datasets.LSUN(data_dir, lsun_categories, transform, train=is_train)
      nlabels = len(dataset.classes)
  
  else:
      raise NotImplemented

  return dataset, nlabels


class MultipleDatasetsBatchSampler():
    """
    MultipleDatasetsBatchSampler:
     - Draw a small dataset size: N ~ p(N)
     - Draw labels: C ~ Cat(Dir(1,...,1))
     - Draw data points from the dataset according to the labels in C.
    Returns batches of size X=[B * N, D], Y=[B * N] where N and D are fixed values for one certain batch. (B = Batch size)
    When loading the data, after the right reshape, we can get the batch: X=[B, N, D], Y=[B, N]
    Taken from https://github.com/adambielski/siamese-triplet
    """

    def __init__(self, config, labels, is_train=True):
        self.N_min = config.data.N_min
        self.N_max = config.data.N_max
        self.K_min = config.data.K_min

        if is_train:
          self.B = config.training.batch_size
        else:
          self.B = config.evaluate.batch_size

        self.labels = labels  # shape: [number of points in the whole dataset,], contains the labels of each point in the dataset according to the order the in the dataset.
        self.labels_set = list(set(self.labels.numpy()))
        self.nlabels = len(self.labels_set)  # Number of labels we have in the dataset.
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
        K_max = min(N, self.nlabels)
        K = random.randint(self.K_min, K_max)
        K_labels = np.random.choice(self.nlabels, K, replace=False)  # Draw the original K label values that will participate each dataset in the batch.
        indices = []  # will be in shape: [B*N] and will include B sets of indices, each set is of size N.
        
        # For each B_i, create indices that will form the datapoints of the small dataset in entry i in the batch:
        for i in range(self.B):

          dirichlet_sample = np.random.dirichlet(np.ones(K))
          multinomial_sample = np.random.multinomial(N, dirichlet_sample) # shape [K,] contains the count of datapoints to sample per label. Sum(multinomial_sample) = N.

          B_i_indices = []
          for j in range(multinomial_sample.shape[0]):
            n = multinomial_sample[j]
            lbl = K_labels[j]

            if self.used_label_indices_count[lbl] + n > len(self.label_to_indices[lbl]):
              np.random.shuffle(self.label_to_indices[lbl])
              self.used_label_indices_count[lbl] = 0

            B_i_indices.extend(self.label_to_indices[lbl][self.used_label_indices_count[lbl]:self.used_label_indices_count[lbl] + n])
            self.used_label_indices_count[lbl] += n

          np.random.shuffle(B_i_indices)
          indices.extend(B_i_indices)
        
        yield indices
        self.count += self.B * N



# ------------- create dataset using tensorflow -----------------------

def get_dataset_tensorflow(config, uniform_dequantization=False, evaluation=False):
  """Create data loaders for training and evaluation.

  Args:
    config: A ml_collection.ConfigDict parsed from config files.
    uniform_dequantization: If `True`, add uniform dequantization to images.
    evaluation: If `True`, fix number of epochs to 1.

  Returns:
    train_ds, eval_ds, dataset_builder.
  """
  # Compute batch size for this worker.
  batch_size = config.training.batch_size if not evaluation else config.eval.batch_size

  # Reduce this when image resolution is too large and data pointer is stored
  shuffle_buffer_size = 10000
  prefetch_size = tf.data.experimental.AUTOTUNE
  num_epochs = None if not evaluation else 1

  # Create dataset builders for each dataset.
  if config.data.dataset == 'CIFAR10':
    dataset_builder = tfds.builder('cifar10')
    train_split_name = 'train'
    eval_split_name = 'test'

    def resize_op(img):
      img = tf.image.convert_image_dtype(img, tf.float32)
      return tf.image.resize(img, [config.data.image_size, config.data.image_size], antialias=True)

  elif config.data.dataset == 'FASHIONMNIST':
    dataset_builder = tfds.builder('fashion_mnist')
    train_split_name = 'train'
    eval_split_name = 'test'

    def resize_op(img):
      img = tf.image.convert_image_dtype(img, tf.float32)
      return tf.image.resize(img, [config.data.image_size, config.data.image_size], antialias=True)

  elif config.data.dataset == 'SVHN':
    dataset_builder = tfds.builder('svhn_cropped')
    train_split_name = 'train'
    eval_split_name = 'test'

    def resize_op(img):
      img = tf.image.convert_image_dtype(img, tf.float32)
      return tf.image.resize(img, [config.data.image_size, config.data.image_size], antialias=True)

  elif config.data.dataset == 'CELEBA':
    dataset_builder = tfds.builder('celeb_a')
    train_split_name = 'train'
    eval_split_name = 'validation'

    def resize_op(img):
      img = tf.image.convert_image_dtype(img, tf.float32)
      img = central_crop(img, 140)
      img = resize_small(img, config.data.image_size)
      return img

  elif config.data.dataset == 'LSUN':
    dataset_builder = tfds.builder(f'lsun/{config.data.category}')
    train_split_name = 'train'
    eval_split_name = 'validation'

    if config.data.image_size == 128:
      def resize_op(img):
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = resize_small(img, config.data.image_size)
        img = central_crop(img, config.data.image_size)
        return img

    else:
      def resize_op(img):
        img = crop_resize(img, config.data.image_size)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img

  elif config.data.dataset in ['FFHQ', 'CelebAHQ']:
    dataset_builder = tf.data.TFRecordDataset(config.data.tfrecords_path)
    train_split_name = eval_split_name = 'train'

  else:
    raise NotImplementedError(
      f'Dataset {config.data.dataset} not yet supported.')

  # Customize preprocess functions for each dataset.
  if config.data.dataset in ['FFHQ', 'CelebAHQ']:
    def preprocess_fn(d):
      sample = tf.io.parse_single_example(d, features={
        'shape': tf.io.FixedLenFeature([3], tf.int64),
        'data': tf.io.FixedLenFeature([], tf.string)})
      data = tf.io.decode_raw(sample['data'], tf.uint8)
      data = tf.reshape(data, sample['shape'])
      data = tf.transpose(data, (1, 2, 0))
      img = tf.image.convert_image_dtype(data, tf.float32)
      if config.data.random_flip and not evaluation:
        img = tf.image.random_flip_left_right(img)
      if uniform_dequantization:
        img = (tf.random.uniform(img.shape, dtype=tf.float32) + img * 255.) / 256.
      return dict(image=img, label=None)

  else:
    def preprocess_fn(d):
      """Basic preprocessing function scales data to [0, 1) and randomly flips."""
      img = resize_op(d['image'])
      if config.data.random_flip and not evaluation:
        img = tf.image.random_flip_left_right(img)
      if uniform_dequantization:
        img = (tf.random.uniform(img.shape, dtype=tf.float32) + img * 255.) / 256.

      return dict(image=img, label=d.get('label', None))

  def create_dataset(dataset_builder, split):
    dataset_options = tf.data.Options()
    dataset_options.experimental_optimization.map_parallelization = True
    dataset_options.experimental_threading.private_threadpool_size = 48
    dataset_options.experimental_threading.max_intra_op_parallelism = 1
    read_config = tfds.ReadConfig(options=dataset_options)
    if isinstance(dataset_builder, tfds.core.DatasetBuilder):
      dataset_builder.download_and_prepare()
      ds = dataset_builder.as_dataset(
        split=split, shuffle_files=True, read_config=read_config)
    else:
      ds = dataset_builder.with_options(dataset_options)
    ds = ds.repeat(count=num_epochs)
    ds = ds.shuffle(shuffle_buffer_size)
    ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds.prefetch(prefetch_size)

  train_ds = create_dataset(dataset_builder, train_split_name)
  eval_ds = create_dataset(dataset_builder, eval_split_name)
  return train_ds, eval_ds, dataset_builder
