import csv
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test =  to_categorical(y_train), to_categorical(y_test)
x_train.shape = (-1,28,28,1)
x_test.shape = (-1,28,28,1)


def save_to_csv(filename, legend, results_list):
    with open(filename, 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(legend)
        csv_writer.writerows(results_list)

def multi_runs_mnist(epochs, n_runs, names, network_builders, batches_size=20):
    if type(network_builders) != list:
        network_builders = [network_builders for _ in names]
    if type(batches_size) == int:
        batches_size = [batches_size for _ in zip(names, network_builders)]
    for name, network_builder, batch_size in zip(names, network_builders, batches_size):
        results = []
        xs = []
        for i in range(n_runs):
            model = network_builder()
            logger = Tester(30000, x_test, y_test)
            model.fit(x_train, y_train, epochs=epochs,batch_size=batch_size, callbacks=[logger])
            results.append(logger.logs)
        xs = [res[0] for res in results[0]]
        results = [[point[1] for point in result] for result in results]
        save_to_csv(name, xs, results)

class Tester(Callback):
    def __init__(self, elements_per_log, x_test, y_test, filepath=""):
        self.seen = 0
        self.display = elements_per_log
        self.last_logged = 0
        self.logs = []
        self.x_test = x_test
        self.y_test = y_test

    def on_train_begin(self, logs={}):
        self.logs.append((0, 1 - self.model.evaluate(self.x_test, self.y_test)[1]))

    def on_batch_end(self, batch, logs={}):
        self.seen += logs.get('size', 0)
        if self.seen // self.display != self.last_logged:
            # you can access loss, accuracy in self.params['metrics']
            self.last_logged = self.seen // self.display
            self.logs.append((self.seen, 1 - self.model.evaluate(self.x_test, self.y_test)[1]))