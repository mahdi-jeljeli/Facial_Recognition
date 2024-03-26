import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input

def load_dataset(folder_path):
    """
        Loads the dataset from the specified folder path.

        Parameters:
        - folder_path (str): Path to the folder containing the dataset.

        Returns:
        - train_images (numpy.ndarray): Array of training images.
        - trainY (numpy.ndarray): Array of training labels.
        - test_images (numpy.ndarray): Array of testing images.
        - testY (numpy.ndarray): Array of testing labels.
    """
    images = []
    labels = []
    for subdir, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(subdir, file)
            try:
                with Image.open(file_path) as img:
                    img = img.convert('L')  # Convert to grayscale
                    img = img.resize((28, 28))  # Resize to 28x28
                    img_array = np.array(img)  # Convert to numpy array
                    images.append(img_array)
                    labels.append(subdir.split(os.path.sep)[-1])  # Assuming label is the name of the folder
            except Exception as e:
                print(f"Error loading image '{file_path}': {e}")

    # Convert labels to integers
    encoded_labels = np.array([int(label) - 1 for label in labels])  # Assuming labels start from 1

    # Convert images to numpy arrays
    images = np.array(images)

    # Determine the number of classes
    num_classes = len(set(labels))

    # Split the data into training and testing sets (80% train, 20% test)
    train_images, test_images, train_labels, test_labels = train_test_split(
        images, encoded_labels, test_size=0.2, random_state=42)

    # Perform one-hot encoding on the labels
    trainY = to_categorical(train_labels, num_classes=num_classes)
    testY = to_categorical(test_labels, num_classes=num_classes)

    return train_images, trainY, test_images, testY





def prepare_pixels(train, test):
    """
       Preprocesses the pixel values of the images.

       Parameters:
       - train (numpy.ndarray): Array of training images.
       - test (numpy.ndarray): Array of testing images.

       Returns:
       - train_norm (numpy.ndarray): Preprocessed training images.
       - test_norm (numpy.ndarray): Preprocessed testing images.
    """

    # Convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')

    # Normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0

    # Return normalized images
    return train_norm, test_norm

def cnn_model():
    """
       Defines the CNN model architecture.

       Returns:
       - model (Sequential): CNN model.
    """
    model = Sequential()
    model.add(Input(shape=(28, 28, 1)))  # Ajouter une couche Input(shape)
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(40, activation='softmax'))  # 40 classes pour la base de données ORL

    # Compiler le modèle
    opt = SGD(learning_rate=0.01, momentum=0.9)  # Utiliser learning_rate au lieu de lr
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def evaluate_model(dataX, dataY, n_folds):
    """
       Evaluates the CNN model using k-fold cross-validation.

       Parameters:
       - dataX (numpy.ndarray): Input images.
       - dataY (numpy.ndarray): Labels.
       - n_folds (int): Number of folds for cross-validation.

       Returns:
       - scores (list): List of accuracy scores for each fold.
       - histories (list): List of training histories for each fold.
    """

    scores, histories = list(), list()

    # Prepare cross-validation
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=1)

    # Enumerate splits
    for train_ix, test_ix in kfold.split(dataX):
        model = cnn_model()
        trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]

        # Fit the model
        history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)

        # Evaluate the model
        _, acc = model.evaluate(testX, testY, verbose=0)
        print('> %.3f' % (acc * 100.0))

        # Save model
        #model.save('final_model.h5')  # Save the model for future use
        model.save('my_model.keras')  # Sauvegarde dans le format natif Keras

        scores.append(acc)
        histories.append(history)

    return scores, histories



def accuracy_summary(histories):
    """
       Plots the classification accuracy (la précision) for each fold.

       Parameters:
       - histories (list): List of training histories.
    """
    num_plots = min(len(histories), 4)  # Limiter le nombre de sous-graphiques à 4
    plt.figure(figsize=(10, 6))
    for i in range(num_plots):
        plt.subplot(2, 2, i+1)
        plt.title('Classification Accuracy')
        plt.plot(histories[i].history['accuracy'], color='blue', label='train')
        plt.plot(histories[i].history['val_accuracy'], color='orange', label='validation')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
    plt.tight_layout()
    plt.show()

def loss_summary(histories):
    """
       Plots the loss (la perte) for each fold.

       Parameters:
       - histories (list): List of training histories.
    """
    num_plots = min(len(histories), 4)
    plt.figure(figsize=(10, 6))
    for i in range(num_plots):
        plt.subplot(2, 2, i+1)
        plt.title('Loss')
        plt.plot(histories[i].history['loss'], color='blue', label='train')
        plt.plot(histories[i].history['val_loss'], color='orange', label='validation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
    plt.tight_layout()
    plt.show()



"""
def final():
    # Load dataset
    trainX, trainY, testX, testY = load_dataset(folder_path)

    # Prepare pixels
    trainX, testX = prepare_pixels(trainX, testX)

    # Evaluate model
    scores, histories = evaluate_model(trainX, trainY)

    # Visualize accuracy and loss summaries
    accuracy_summary(histories)
    loss_summary(histories)
"""
#folder_path = "E:\\data"
# Call final() function to execute the entire process
#final()
