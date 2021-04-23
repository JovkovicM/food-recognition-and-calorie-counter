import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

from sklearn.preprocessing import LabelBinarizer

label_binarizer = LabelBinarizer()

def apply_cnn(x_train, x_test, cnn_y_train, cnn_y_test):
    batch_size = 5
    num_classes = len(np.unique(cnn_y_train))
    epochs = 25
    
    y_train = label_binarizer.fit_transform(np.array(cnn_y_train))
    y_test = label_binarizer.fit_transform(np.array(cnn_y_test))
    
    x_train = np.concatenate([arr[np.newaxis] for arr in x_train])
    x_test = np.concatenate([arr[np.newaxis] for arr in x_test])
    
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                    input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # initiate RMSprop optimizer
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255


    model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_test, y_test),
            shuffle=True)

    scores = model.evaluate(x_test, y_test, verbose=1)
    y_pred = model.predict(x_test)
    print(scores)
    print(np.around(y_pred, decimals=3))
    print(y_test)

    return y_pred