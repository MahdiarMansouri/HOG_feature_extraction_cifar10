from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPool2D, Flatten, Dropout, AvgPool2D
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.activations import relu, softmax
from keras.datasets import cifar10
from skimage.exposure import exposure
from skimage.feature import hog
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Downloading Data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# HOG feature extraction
X_train_hog = np.array([])
for idx, img in enumerate(X_train):
    img_hog = np.array([])
    for channel_num in range(img.shape[2]):
        _, hog_channel = hog(img[:, :, channel_num], pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                             block_norm='L2-Hys', visualize=True)
        hog_channel = exposure.rescale_intensity(hog_channel, in_range=(0, 10))
        img_hog = np.append(img_hog, hog_channel, axis=0)
    print(idx)
    print()

for idx, img in enumerate(X_test):
    for channel_num in range(img.shape[2]):
        _, hog_channel = hog(img[:, :, channel_num], pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                             block_norm='L2-Hys', visualize=True)
        hog_channel = exposure.rescale_intensity(hog_channel, in_range=(0, 10))
        img[:, :, channel_num] = hog_channel
    print(idx)

print('HOG feature extraction Done!..')

# Building Model
model = Sequential()

model.add(Conv2D(64, input_shape=(32, 32, 3), kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print('Model built!..')

# Train model
history = model.fit(
    X_train,
    y_train,
    validation_split=0.1,
    epochs=30,
)

# Evaluate model
print(model.evaluate(X_test, y_test))

# Plot the results
plt.subplot(1, 2, 1)
plt.title('Accuracy')
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.subplot(1, 2, 2)
plt.title('Loss')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.show()