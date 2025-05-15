from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np

# Load the Fashion MNIST dataset
(trainx, trainy), (testx, testy) = fashion_mnist.load_data()

# Display the shape of training and test data
print('Train: X =', trainx.shape)
print('Test: X =', testx.shape)

# Display the first 8 images in the training set
for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.imshow(trainx[i], cmap='gray')
plt.show()

# Expand dimensions to add a channel for CNN input
trainx = np.expand_dims(trainx, -1)
testx = np.expand_dims(testx, -1)

print("Train shape after expansion:", trainx.shape)

# Define the CNN architecture
def model_arch():
    model = Sequential()
    model.add(Conv2D(64, (5, 5), padding="same", activation="relu", input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (5, 5), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, (5, 5), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dense(10, activation="softmax"))
    return model

model = model_arch()

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

# Show model architecture
model.summary()

# Train the model
history = model.fit(
    trainx.astype(np.float32), trainy.astype(np.float32),
    epochs=10,
    steps_per_epoch=100,
    validation_split=0.33
)

# Save model weights
model.save_weights('./model.h5', overwrite=True)

# Plot training and validation accuracy
plt.plot(history.history['sparse_categorical_accuracy'])
plt.plot(history.history['val_sparse_categorical_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Make predictions and display the result
labels = ['t_shirt', 'trouser', 'pullover', 'dress', 'coat',
          'sandal', 'shirt', 'sneaker', 'bag', 'ankle_boot']

predictions = model.predict(testx[:1])
label = labels[np.argmax(predictions)]
print("Predicted label:", label)

# Display the image
plt.imshow(testx[0], cmap='gray')
plt.title(f'Predicted: {label}')
plt.show()
