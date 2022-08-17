import tensorflow as tf
from scipy import io as sio
import numpy as np
import matplotlib.pyplot as plt
import PIL
import PIL.ImageOps

mat = sio.loadmat('emnist-byclass.mat')
data = mat['dataset']

X_train = data['train'][0,0]['images'][0,0]
y_train = data['train'][0,0]['labels'][0,0]
X_test = data['test'][0,0]['images'][0,0]
y_test = data['test'][0,0]['labels'][0,0]

X_train, X_test = X_train / 255, X_test / 255

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(62, activation="softmax"))

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

model.fit(X_train, y_train, epochs = 5, batch_size = 750)

val_loss, val_acc = model.evaluate(X_test, y_test)

print(val_loss,val_acc)

model.save("letter_digit_recognizer.model")


'''
model = tf.keras.models.load_model('letter_digit_recognizer.model')
idx = np.where(y_test == 28)
idx = idx[0][2]
test_image = np.reshape(X_test[idx], (28,28))
test_image = PIL.Image.fromarray(test_image * 255)
test_image = PIL.ImageOps.mirror(test_image)
test_image = test_image.rotate(90)
plt.imshow(test_image)
plt.show()
prediction = model.predict(np.expand_dims(X_test[idx], axis=0))
print("\n\n")
print("predicted",np.argmax(prediction))
print("actual",y_test[idx])
print("\n\n")'''