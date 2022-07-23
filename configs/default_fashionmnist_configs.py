import ml_collections
import torch


def get_default_configs():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  config.training.batch_size = 32 #128
  training.n_iters = 1300001
  training.snapshot_freq = 50000
  training.log_freq = 50
  training.eval_freq = 100
  ## store additional checkpoints for preemption in cloud computing environments
  training.snapshot_freq_for_preemption = 10000
  ## produce samples at each snapshot.
  training.snapshot_sampling = True
  training.likelihood_weighting = False
  training.continuous = True
  training.reduce_mean = False

  # sampling
  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.n_steps_each = 1
  sampling.noise_removal = True
  sampling.probability_flow = False
  sampling.snr = 0.16

  # evaluation
  config.eval = evaluate = ml_collections.ConfigDict()
  evaluate.begin_ckpt = 9
  evaluate.end_ckpt = 26
  evaluate.batch_size = 32
  evaluate.enable_sampling = True #False
  evaluate.num_samples = 50000
  evaluate.enable_loss = True
  evaluate.enable_bpd = True #False
  evaluate.bpd_dataset = 'test'

  # data
  config.data = data = ml_collections.ConfigDict()
  data.dataset = 'FASHIONMNIST'
  data.image_size = 32 #28
  data.random_flip = True
  data.centered = False
  data.uniform_dequantization = False
  data.num_channels = 1
  data.N_min = 10  # the minimum size of a dataset (as an entry in the batch)
  data.N_max = 64  # the maximum size of a dataset (as an entry in the batch)
  data.K_min = 2  # the minimum number of clusters within one dataset in each entry in the batch.
  data.data_dir = 'data/FASHIONMNIST'
  data.nworkers = 15
  data.nlabels = 10

  # model
  config.model = model = ml_collections.ConfigDict()
  model.sigma_min = 0.01
  model.sigma_max = 50
  model.num_scales = 1000
  model.beta_min = 0.1
  model.beta_max = 20.
  model.dropout = 0.1
  model.embedding_type = 'fourier'

  # optimization
  config.optim = optim = ml_collections.ConfigDict()
  optim.weight_decay = 0
  optim.optimizer = 'Adam'
  optim.lr = 2e-4
  optim.beta1 = 0.9
  optim.eps = 1e-8
  optim.warmup = 5000
  optim.grad_clip = 1.

  # Feature Extraction
  config.encoder = encoder = ml_collections.ConfigDict()
  encoder.type = 'resnet18'  # options: {'resnet18', 'resnet50'}
  encoder.output_dim = 128

  # Mapping discrete labels to continouos space:
  config.mapping_to_cont = mapping_to_cont = ml_collections.ConfigDict()
  mapping_to_cont.mu = 1.0
  mapping_to_cont.sigma = 0.1

  config.seed = 42
  config.device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
  #config.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

  return config