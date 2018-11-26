from utils import Tester
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import categorical_crossentropy

def CNN_factory():
    model = Sequential([
        Conv2D(40, kernel_size=5, input_shape=(28, 28, 1)),
        LeakyReLU(),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(100, kernel_size=3),
        LeakyReLU(),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(200, kernel_size=3),
        LeakyReLU(),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Flatten(),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer=SGD(lr=0.01),
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    return model