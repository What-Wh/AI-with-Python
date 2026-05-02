from sklearn.datasets import load_wine
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import pandas as pd


df = pd.read_csv('./assets/cars.csv')

X = df[["speed_kmh", "fuel_consumption_l_per_100km", "travel_time_h"]].values
Y = df["engine_type"].map({"petrol": 0, "diesel": 1}).values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

model = tf.keras.Sequential(
    [
        layers.Dense(16, activation="relu", input_shape=(3,)),
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

sample = np.array([[55,6.7,1.82]]) 
sample = scaler.transform(sample) 
pred_logits = model.predict(sample)

print("Logits:", pred_logits)

pred_class = np.argmax(pred_logits, axis=1)
if pred_class == 1:
    p = "Petrol"
else: 
    p = "Diesel"
print("Predicted class:", pred_class, "->", p)