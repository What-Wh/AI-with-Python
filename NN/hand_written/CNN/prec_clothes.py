import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Load data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Normalize pixel values to [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape input to 28x28x1 for CNN
x_train_cnn = x_train.reshape(-1, 28, 28, 1)
x_test_cnn = x_test.reshape(-1, 28, 28, 1)

# One-hot encode labels
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(16, 5, activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(3), # reduce spatial dimensions x
    Dropout(0.25), # disable 25% of the neurons to prevent overfitting  
    Conv2D(32, 5, activation='relu'),
    MaxPooling2D(3), # reduce spatial dimensions
    Dropout(0.25), # disable 25% of the neurons to prevent overfitting
    Flatten(), # 2D to 1D
    Dense(128, activation='relu'),
    Dropout(0.5), # disable 50% of the neurons to prevent overfitting
    Dense(10, activation='softmax')
])
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(x_train_cnn, y_train_cat, epochs=5, validation_split=0.2, callbacks=[early_stop])

model.save('num_cnn_prec_clothes.h5')

test_loss, test_acc = model.evaluate(x_test_cnn, y_test_cat)
print("Test accuracy:", test_acc)

import numpy as np

# Predict the first 5 test samples
predictions = model.predict(x_test_cnn[:5])
for i in range(5):
    plt.imshow(x_test_cnn[i].reshape(28, 28), cmap="gray")
    plt.title(f"Predicted: {np.argmax(predictions[i])} - True: {y_test[i]}")
    plt.axis('off')
    plt.show()

Y1 = history.history['accuracy']
Y1_v = history.history['val_accuracy']
Y2 = history.history['loss']
Y2_v = history.history['val_loss']
X = [1, 2, 3, 4, 5]

plt.figure(figsize=(10,6))
plt.plot(X, Y1, label='Accuracy', color='blue')
plt.plot(X, Y1_v, label='Val_accuracy', color='red', linestyle='--')
plt.title("Epoch / Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10,6))
plt.plot(X, Y2, label='Loss', color='blue')
plt.plot(X, Y2_v, label='Val_loss', color='red', linestyle='--')
plt.title("Epoch / Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()