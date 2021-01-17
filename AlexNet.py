import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


import matplotlib.pyplot as plt
import seaborn as sns
import keras

from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report, confusion_matrix
from keras.callbacks import ReduceLROnPlateau
import cv2
import os

labels = ['PNEUMONIA', 'NORMAL']
img_size = 299


def get_training_data(data_dir):
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                resized_arr = cv2.resize(img_arr, (img_size, img_size))
                data.append([resized_arr, class_num])
            except Exception as exep:
                print(exep)
    return np.array(data)


train = get_training_data('C:/Users/Ahmet/Desktop/CS464Project1/train')
test = get_training_data('C:/Users/Ahmet/Desktop/CS464Project1/test')
val = get_training_data('C:/Users/Ahmet/Desktop/CS464Project1/val')

l = []
for i in train:
    if (i[1] == 0):
        l.append("Pneumonia")
    else:
        l.append("Normal")
sns.set_style('darkgrid')
sns.countplot(l)

plt.figure(figsize=(5, 5))
plt.imshow(train[0][0], cmap='gray')
plt.title(labels[train[0][1]])

plt.figure(figsize=(5, 5))
plt.imshow(train[-1][0], cmap='gray')
plt.title(labels[train[-1][1]])

x_train = []
y_train = []

x_val = []
y_val = []

x_test = []
y_test = []

def xy_split(data):
    x_list = []
    y_list = []
    for x,y in data:
        x_list.append(x)
        y_list.append(y)
    return x_list,y_list

x_train,y_train = xy_split(train)
x_test,y_test = xy_split(test)
x_val,y_val = xy_split(val)

x_train = np.array(x_train) / 255
x_val = np.array(x_val) / 255
x_test = np.array(x_test) / 255

x_train = x_train.reshape(-1, img_size, img_size, 1)
y_train = np.array(y_train)

x_val = x_val.reshape(-1, img_size, img_size, 1)
y_val = np.array(y_val)

x_test = x_test.reshape(-1, img_size, img_size, 1)
y_test = np.array(y_test)

datagen = ImageDataGenerator(

    rotation_range=10,
    zoom_range=0.2,
    vertical_flip=True,
    horizontal_flip=False,
    width_shift_range=0.2,
    height_shift_range=0.2
    )

datagen.fit(x_train)
epoch_size= 12
model = keras.models.Sequential([
    keras.layers.Conv2D(16,kernel_size=(11, 11), activation='relu', input_shape=(299, 299, 1)),
    keras.layers.MaxPool2D(2),
    keras.layers.Conv2D(32,kernel_size= (5,5), activation='relu'),
    keras.layers.MaxPool2D(2),
    keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
    keras.layers.MaxPool2D(2),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(units=1, activation='sigmoid')
])
model.compile(optimizer="rmsprop", loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.3, min_lr=0.000001)
history = model.fit(datagen.flow(x_train, y_train, batch_size=32), epochs=epoch_size,
                    validation_data=datagen.flow(x_val, y_val), callbacks=[learning_rate_reduction])

print("Model Loss-  ", model.evaluate(x_test, y_test)[0])
print("Model Accuracy - ", model.evaluate(x_test, y_test)[1] * 100, "%")

epochs = [i for i in range(epoch_size)]
fig, axis = plt.subplots(1, 2)
train_acc = history.history['accuracy']
train_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']
fig.set_size_inches(20, 10)

axis[0].plot(epochs, train_acc, label='Training Accuracy')
axis[0].plot(epochs, val_acc, label='Validation Accuracy')
axis[0].legend()
axis[0].set_xlabel("Epochs")
axis[0].set_ylabel("Accuracy")
axis[0].set_title('Training & Validation Accuracy')


axis[1].plot(epochs, train_loss, label='Training Loss')
axis[1].plot(epochs, val_loss, label='Validation Loss')
axis[1].legend()
axis[1].set_xlabel("Epochs")
axis[1].set_ylabel("Training & Validation Loss")
axis[1].set_title('Testing Accuracy & Loss')

plt.show()

pred = model.predict_classes(x_test)
pred = pred.reshape(1, -1)[0]
print(classification_report(y_test, pred, target_names=['Pneumonia', 'Normal']))

cm = confusion_matrix(y_test, pred)
cm

cm = pd.DataFrame(cm, index=['0', '1'], columns=['0', '1'])

plt.figure(figsize=(10, 10))
sns.heatmap(cm, cmap="Blues", linewidth=1, annot=True, fmt='', xticklabels=labels,
            yticklabels=labels)

