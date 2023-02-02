import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
import pandas as pd
import numpy as np
import os
from skimage import io


def trim_images(ids, train_dir):
    trimmed_imgs = np.ndarray(shape=(len(ids), 32, 32, 3), dtype=float)
    for ix, id in enumerate(ids):
        image = io.imread(os.path.abspath(os.path.join(train_dir, id + '.tif')))
        new_img = image[32:64,32:64,:]
        new_img = new_img/255 
        trimmed_imgs[ix] = new_img
    return trimmed_imgs

root_dir = os.path.abspath(os.path.dirname(__file__))
train_dir = os.path.abspath(os.path.join(root_dir, 'input', 'train'))
test_dir = os.path.abspath(os.path.join(root_dir, 'input', 'test'))

train_df = pd.read_csv(os.path.join(root_dir, 'input', 'train_labels.csv'))
test_df = pd.read_csv(os.path.join(root_dir, 'input', 'sample_submission.csv'))

x_test = trim_images(ids=test_df['id'].to_list(), train_dir=test_dir)

x_train = trim_images(ids=train_df['id'].to_list(), train_dir=train_dir)
y_train = train_df['label'].astype(float).to_numpy()

model = keras.Sequential()
model.add(Conv2D(32, (3, 3), activation="relu", input_shape = (32, 32, 3), name="conv0"))
model.add(MaxPooling2D((2, 2), name="maxpool0"))
model.add(Conv2D(32, (3, 3), activation="relu", name="conv1"))
model.add(MaxPooling2D((2, 2), name="maxpool1"))
model.add(Conv2D(32, (3, 3), activation="relu", name="conv2"))
model.add(MaxPooling2D((2, 2), name="maxpool2"))
model.add(Flatten())
model.add(Dense(64, activation="relu", name="dense0"))
model.add(Dropout(0.3))
model.add(Dense(16, activation="relu", name="dense1"))
model.add(Dense(8, activation="relu", name="dense2"))
model.add(Dense(1, activation="sigmoid", name="dense3"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=20, batch_size=1024)
model.save('cancer_model')
y_test = model.predict(x_test)

y_test = np.where(y_test >= 0.5, 1, 0)
test_df['label'] = y_test
test_df.to_csv('y_test.csv', index=False)

