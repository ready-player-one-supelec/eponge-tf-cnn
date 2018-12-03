import csv
from time import time
from contextlib import contextmanager
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test =  to_categorical(y_train), to_categorical(y_test)
x_train.shape = (-1,28,28,1)
x_test.shape = (-1,28,28,1)


def save_to_csv(filename, legend, results_list):
    with open(filename + '.csv', 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(legend)
        csv_writer.writerows(results_list)

def multi_runs_mnist(epochs, n_runs, names, network_builders, batches_size=32, metrics=[('sample', 'error_rate')]):
    if type(names) != list:
        names = [names]
    if type(network_builders) != list:
        network_builders = [network_builders for _ in names]
    if type(batches_size) == int:
        batches_size = [batches_size for _ in zip(names, network_builders)]
    for name, network_builder, batch_size in zip(names, network_builders, batches_size):
        metrics = {metric : [] for metric in metrics}
        xs = []
        print(name)
        for i in range(n_runs):
            model = network_builder()
            logger = Tester(30000, x_test, y_test)
            model.fit(x_train, y_train, epochs=epochs,batch_size=batch_size, callbacks=[logger])
            for metric, results in metrics.items():
                results.append(list(zip(logger.logs[metric[0]], logger.logs[metric[1]])))
        print(metrics)
        for (x,y), results in metrics.items():
            xs = [res[0] for res in results[0]]
            results = [[point[1] for point in result] for result in results]
            save_to_csv(name + f"{y}:f({x})", xs, results)

class Timer:
    def __init__(self):
        self.elapsed = 0
        self.current = time()
    
    def start(self):
        self.elapsed = 0
        self.current = time()
    
    def pause(self):
        self.elapsed += time() - self.current
    
    def unpause(self):
        self.current = time()

    @contextmanager
    def stop_for(self):
        try:
            self.pause()
            yield self
        finally:
            self.unpause()


class Tester(Callback):
    def __init__(self, elements_per_log, x_test, y_test, filepath=""):
        self.seen = 0
        self.display = elements_per_log
        self.last_logged = 0
        self.timer = Timer()
        self.logs = {"sample" : [], "error_rate" : [], "time" : []}
        self.x_test = x_test
        self.y_test = y_test

    def on_train_begin(self, logs={}):
        self.logs["sample"].append(0)
        self.logs["error_rate"].append(1 - self.model.evaluate(self.x_test, self.y_test)[1])
        self.logs["time"].append(0.0)
        self.timer.start()

    def on_batch_end(self, batch, logs={}):
        with self.timer.stop_for():
            self.seen += logs.get('size', 0)
            if self.seen // self.display != self.last_logged:
                # you can access loss, accuracy in self.params['metrics']
                self.logs["sample"].append(self.seen)
                self.logs["error_rate"].append(1 - self.model.evaluate(self.x_test, self.y_test)[1])
                self.logs["time"].append(self.timer.elapsed)

                self.last_logged = self.seen // self.display
                # self.logs.append((self.seen, 1 - self.model.evaluate(self.x_test, self.y_test)[1]))