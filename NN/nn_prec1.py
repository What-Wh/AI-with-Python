from sklearn.datasets import load_wine
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

wine = load_wine()
X = wine.data
Y = wine.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

model = tf.keras.Sequential(
    [
        layers.Dense(16, activation="relu", input_shape=(13,)),
        layers.Dense(8, activation="relu"),
        layers.Dense(3)
    ]
)

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model.fit(X_train, Y_train, epochs=20, batch_size=8)

model.evaluate(X_test, Y_test)

sample = np.array([[13.2, 1.78, 2.14, 11.2, 100, 2.65, 2.76, 0.26, 1.28, 4.38, 1.05, 3.40, 1050]]) 
sample = scaler.transform(sample) 
pred_logits = model.predict(sample)

print("Logits:", pred_logits)

pred_class = np.argmax(pred_logits, axis=1)
print("Predicted class:", pred_class, "->", wine.target_names[pred_class][0])