from RMSprop_perceptron import SGD_builder, RMS_builder, ADAM_builder
from keras_cnn_mnist import CNN_factory
from utils import multi_runs_mnist

batches_size = [16, 32, 64, 128]
names = [f'data/CNN-batchsize{s}.csv' for s in batches_size]
multi_runs_mnist(10, 10, names, CNN_factory, batches_size=batches_size)