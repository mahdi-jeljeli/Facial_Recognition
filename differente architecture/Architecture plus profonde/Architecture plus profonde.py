from keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from PrinipalCNNModel import load_dataset, prepare_pixels, evaluate_model, accuracy_summary, loss_summary
def cnn_model():
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(40, activation='softmax'))  # 40 classes for ORL Database

        # Compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model
def final():
    # Load dataset
    trainX, trainY, testX, testY = load_dataset(folder_path)

    # Prepare pixels
    trainX, testX = prepare_pixels(trainX, testX)

    # Evaluate model
    scores, histories = evaluate_model(trainX, trainY,10)

    # Visualize accuracy and loss summaries
    accuracy_summary(histories)
    loss_summary(histories)


folder_path = "E:\\data"
# Call final() function to execute the entire process
final()
