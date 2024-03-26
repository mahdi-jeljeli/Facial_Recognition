from keras.src.layers import Add
from keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import SGD
from PrinipalCNNModel import load_dataset, prepare_pixels, evaluate_model, accuracy_summary, loss_summary

from keras.layers import Layer
class ResidualLayer(Layer):
    def __init__(self, filters, **kwargs):
        super(ResidualLayer, self).__init__(**kwargs)
        self.filters = filters

    def build(self, input_shape):
        self.conv1 = Conv2D(self.filters, (3, 3), activation='relu', padding='same')
        self.conv2 = Conv2D(self.filters, (3, 3), activation='relu', padding='same')
        super(ResidualLayer, self).build(input_shape)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = Add()([inputs, x])
        return x

def cnn_model():
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(ResidualLayer(64))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(ResidualLayer(64))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(40, activation='softmax'))  # 40 classes for ORL Database

    # Compiler le modèle
    opt = SGD(learning_rate=0.01, momentum=0.9)  # Utiliser learning_rate au lieu de lr
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def final():
    # Load dataset
    trainX, trainY, testX, testY = load_dataset(folder_path)

    # Prepare pixels
    trainX, testX = prepare_pixels(trainX, testX)

    # Evaluate model
    scores, histories = evaluate_model(trainX, trainY , 10)

    # Visualize accuracy and loss summaries
    accuracy_summary(histories)
    loss_summary(histories)


folder_path = "E:\\data"
# Call final() function to execute the entire process
final()
