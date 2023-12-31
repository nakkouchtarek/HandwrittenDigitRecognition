import os, cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist
# ( input -> output )
(x_train, y_train), (x_test, y_test) = mnist.load_data()

def train_model():
    global x_train, x_test

    # transform to 1d array
    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=3)

    loss, accuracy = model.evaluate(x_test, y_test)

    print(loss, accuracy)

    model.save('handwritten.model')

def test_stats():
    model = tf.keras.models.load_model('handwritten.model')

    loss, accuracy = model.evaluate(x_test, y_test)

    print(loss, accuracy)

def predict(name):
    model = tf.keras.models.load_model('handwritten.model')

    try:
        img = cv2.imread(f"./Numbers/{name}.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"Digit is probably : {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
        
    except:
        print("Error")

if __name__ == "__main__":
    #train_model()
    predict("five")
    #test_stats()