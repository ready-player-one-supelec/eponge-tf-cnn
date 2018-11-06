import tensorflow as tf
from tensorflow.keras.utils import to_categorical
mnist = tf.keras.datasets.mnist
from tensorflow.keras.callbacks import Callback
import csv
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


class Tester(Callback):
    def __init__(self, log_per_batch, x_test, y_test, filepath=""):
        self.seen = 0
        self.display = log_per_batch
        self.errors = []
        self.x_test = x_test
        self.y_test = y_test

    def on_train_begin(self, logs={}):
        self.errors.append((0, 1 - self.model.evaluate(self.x_test, self.y_test)[1]))

    def on_batch_end(self, batch, logs={}):
        self.seen += logs.get('size', 0)
        if self.seen % self.display == 0:
            # you can access loss, accuracy in self.params['metrics']
            self.errors.append((self.seen, 1 - self.model.evaluate(self.x_test, self.y_test)[1]))
    
def SGD_builder(lr=0.3):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(16, activation='tanh'),
        tf.keras.layers.Dense(16, activation='tanh'),
        tf.keras.layers.Dense(10, activation='tanh')
    ])
    model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.3),
                loss=tf.keras.losses.MSE,
                metrics=['accuracy'])
    return model

def RMS_builder():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(16, activation='tanh'),
        tf.keras.layers.Dense(16, activation='tanh'),
        tf.keras.layers.Dense(10, activation='tanh')
    ])
    model.compile(optimizer=tf.keras.optimizers.RMSprop(),
                loss=tf.keras.losses.MSE,
                metrics=['accuracy'])
    return model

def ADAM_builder():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(16, activation='tanh'),
        tf.keras.layers.Dense(16, activation='tanh'),
        tf.keras.layers.Dense(10, activation='tanh')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                loss=tf.keras.losses.MSE,
                metrics=['accuracy'])
    return model

def save_to_csv(filename, legend, results_list):
    with open(filename, 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(legend)
        csv_writer.writerows(results_list)

def multi_runs(epochs, n_runs, names, network_builders):
    for name, network_builder in zip(names, network_builders):
        results = []
        xs = []
        for i in range(n_runs):
            model = network_builder()
            logger = Tester(30000, x_test, to_categorical(y_test))
            model.fit(x_train, to_categorical(y_train), epochs=epochs,batch_size=20, callbacks=[logger])
            results.append(logger.errors)
        xs = [res[0] for res in results[0]]
        results = [[point[1] for point in result] for result in results]
        save_to_csv(name, xs, results)
