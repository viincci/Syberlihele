import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf

# load the intents file
with open("data/json/main.json") as file:
    intents = json.load(file)

# extract the questions and labels from the intents data
questions = [i["questions"] for i in intents["intents"]]
labels = [i["tag"] for i in intents["intents"]]

# vectorize the questions using CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(questions)

# convert the labels to integers
labels = [intents["intents"].index(i) for i in intents["intents"] if i["tag"] in labels]

# one-hot encode the labels
y = tf.keras.utils.to_categorical(labels)

# split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# create the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(32, input_shape=(X.shape[1],), activation='relu'))
model.add(tf.keras.layers.Dense(y.shape[1], activation='softmax'))

# compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# train the model
model.fit(X_train, y_train, epochs=100)

# evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print("Accuracy:", accuracy)

# save the model
model.save("model.h5")
