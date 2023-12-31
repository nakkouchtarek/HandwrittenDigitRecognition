import os, cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


class HandwrittenDigitRecognizer:
    def __init__(self, model):
        self.model = tf.keras.models.load_model(model)

    def train_model(dataset, savefile):
        (x_train, y_train), (x_test, y_test) = dataset.load_data()

        # transform to 1d array
        x_train = tf.keras.utils.normalize(x_train, axis=1)
        x_test = tf.keras.utils.normalize(x_test, axis=1)

        model = tf.keras.models.Sequential()

        # flatter input to 1d array from 28x28 and is also our input layer
        model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

        # 3 layers for treatement with each having 128 neurons
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dense(128, activation='relu'))

        # output layer
        model.add(tf.keras.layers.Dense(10, activation='softmax'))

        model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

        model.fit(x_train, y_train, epochs=3)

        #loss, accuracy = model.evaluate(x_test, y_test)
        #print(loss, accuracy)

        model.save(savefile)

    def predict(self, name):
        try:
            img = cv2.imread(name)[:,:,0]
            img = np.invert(np.array([img]))
            prediction = self.model.predict(img)
            print(f"Digit is probably : {np.argmax(prediction)}")
            plt.imshow(img[0], cmap=plt.cm.binary)
            plt.show()
            
        except:
            print("Error")

if __name__ == "__main__":
    #HandwrittenDigitRecognizer.train_model(tf.keras.datasets.mnist, 'handwritten.model')
    d = HandwrittenDigitRecognizer('handwritten.model')
    d.predict("/home/user/Stuff/HandwrittenDigitRecognition/Numbers/four.png")