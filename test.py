
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

from lib import data_normalizer as dn


# import tensorflow as tf
from tensorflow.keras import datasets, layers, models

from sklearn.model_selection import train_test_split


# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

data = pd.read_csv("normalized_data/unique_plates.csv")
print(data.head())
#data.head(10)


n = 50
sample = data.sample(n)
X = np.ndarray(shape=(n, 480, 480, 3), dtype=np.uint8)
Y = np.array([sample['tag'], sample['xmin'], sample['ymin'], sample['xmax'], sample['ymax']]).T
# Y = np.array(sample[1:])



for i, path in enumerate(sample['img_path']):
    X[i,:,:,:] = cv2.imread(path, cv2.IMREAD_COLOR)

print(sample,Y)
print(Y.shape, X.shape)

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)

# model = models.Sequential()

# # Convolution layers
# model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(480, 480, 3)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(32, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(16, (3, 3), activation='relu'))

# # Normal layers
# model.add(layers.Flatten())
# model.add(layers.Dense(64,activation='relu'))
# model.add(layers.Dropout(0.5))  # Evita la conexión total entre capaz para evitar overfitting
# model.add(layers.Dense(32,activation='relu'))
# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(32,activation='relu'))
# model.add(layers.Dense(5,activation='relu'))   # 5 neuronas de salida, desempeño + fronteras
# # model.summary()

# model.compile(loss="categorical_crossentropy", #MeanSquaredError",#
#               optimizer="Adadelta", 
#               metrics=['accuracy','mean_squared_error'])

# history = model.fit(X_train, Y_train, epochs=5, validation_data=[X_test,Y_test]).history

# model.save('models/model.h5')

# Load model from tf
model = models.load_model('models/model.h5')
print(model.summary())

# print input layer
print(model.layers[0].input)


# plt.figure(figsize=(12,5))
# plt.plot(history['loss'])
# plt.plot(history['val_loss'])
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.legend(['train_data','test_data'])
# plt.title('loss analysis')


plt.figure(figsize=(12,5))
s = data.sample(1)
img = cv2.imread(s['img_path'].values[0], cv2.IMREAD_COLOR)
print(img.shape)

plt.imshow(img)

plt.show()

box = model.predict(img)
print(box)

# dn.draw_image_with_boxes(img, [box,])


# Comentarios

# print(history)

# 1. muestreo %validación %entrenamiento
# 2. Cargar imágenes
# 3. Modelo
# 4. Entrenamiento
# 5. Validación
# 6. gráficas entrenamiento validación
# 7. Función de aplicación 