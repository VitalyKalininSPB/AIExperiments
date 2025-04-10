import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import cv2
from PIL import Image

#setting up the main folder
# Downloaded from: https://www.kaggle.com/datasets/tolgadincer/labeled-chest-xray-images
main_folder = "data/chest_xray/"

train_folder = os.path.join(main_folder, "train")
test_folder = os.path.join(main_folder, "test")

#pneumonia and normal image paths

pneumonia_train_images = glob.glob(train_folder+"/PNEUMONIA/*.jpeg")
normal_train_images = glob.glob(train_folder+"/NORMAL/*.jpeg")

pneumonia_test_images = glob.glob(test_folder+"/PNEUMONIA/*.jpeg")
normal_test_images = glob.glob(test_folder+"/NORMAL/*.jpeg")

#creating training and test dataframes

# Train dataset
train_list = [x for x in normal_train_images]
train_list.extend([x for x in pneumonia_train_images])

df_train = pd.DataFrame(np.concatenate([["Normal"]*len(normal_train_images),
                                       ["Pneumonia"]*len(pneumonia_train_images)]), columns=["class"])

df_train["image"] = [x for x in train_list]
print(df_train.head())
print(df_train.shape)

# Test dataset
test_list = [x for x in normal_test_images]
test_list.extend([x for x in pneumonia_test_images])

df_test = pd.DataFrame(np.concatenate([["Normal"]*len(normal_test_images),
                                      ["Pneumonia"]*len(pneumonia_test_images)]), columns=["class"])
df_test["image"] = [x for x in test_list]


plt.figure(figsize=(10,8))
plt.title("Number of cases", fontsize=12)
sns.countplot(data=df_train, x = df_train['class'])

#plt.show()

#normal images

fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(15,10))

for i, ax in enumerate(axes.flat):
    img = cv2.imread(normal_train_images[i])
    img = cv2.resize(img, (512,512))
    ax.imshow(img)
    ax.set_title("Normal")
    ax.set_xticks([])
    ax.set_yticks([])
fig.tight_layout()
#plt.show()

#pneumonia images

fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(15,10))

for i, ax in enumerate(axes.flat):
    img = cv2.imread(pneumonia_train_images[i])
    img = cv2.resize(img, (512,512))
    ax.imshow(img)
    ax.set_title("Pneumonia")
    ax.set_xticks([])
    ax.set_yticks([])
fig.tight_layout()
#plt.show()

from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
                rescale=1/255)

val_datagen = ImageDataGenerator(
                rescale=1/255)

train_generator = train_datagen.flow_from_dataframe(
                    df_train,
                    x_col="image",
                    y_col="class",
                    target_size=(150,150),
                    batch_size=32,
                    class_mode="binary",
                    seed=7)
# FIXME: compare prepared images with original

from sklearn.model_selection import train_test_split
df_train, df_val = train_test_split(df_train, test_size=0.2, random_state=13, stratify=df_train["class"])

val_generator = val_datagen.flow_from_dataframe(
                    df_val,
                    x_col="image",
                    y_col="class",
                    target_size=(150,150),
                    batch_size=32,
                    class_mode="binary",
                    seed=7)

test_generator = val_datagen.flow_from_dataframe(
                    df_test,
                    x_col="image",
                    y_col="class",
                    target_size=(150,150),
                    batch_size=32,
                    class_mode="binary",
                    shuffle=False,
                    seed=7)

# Construct the model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras.models import Sequential
model = Sequential()

#convolution
model.add(Conv2D(filters=16, kernel_size=(3,3), activation="relu", input_shape=(150,150,3)))

#pooling
model.add(MaxPooling2D(pool_size=(2,2)))

#2nd Conv
model.add(Conv2D(filters=32, kernel_size=(3,3), activation="relu"))

#2nd pooling
model.add(MaxPooling2D(pool_size=(2,2)))

#3rd conv
model.add(Conv2D(filters=64, kernel_size=(3,3), activation="relu"))

#3rd pooling
model.add(MaxPooling2D(pool_size=(2,2)))

#flatten
model.add(Flatten())

#fully connected layer
model.add(Dense(64, activation="relu"))

model.add(Dense(1, activation="sigmoid"))

#compiling
model.compile(loss="binary_crossentropy",
             optimizer="adam",
             metrics=["accuracy"])
# FIXME: Which else metrics exists?
"""
model_1 = model.fit(
                    train_generator,
                    epochs=4,
                    validation_data=val_generator)

plt.figure(figsize=(12,8))

plt.subplot(2,2,1)
plt.plot(model_1.history["loss"], label="Training Loss")
plt.plot(model_1.history["val_loss"], label="Validation Loss")
plt.legend()
plt.title("Loss Evolution")

plt.subplot(2,2,2)
plt.plot(model_1.history["accuracy"], label="Training Accuracy")
plt.plot(model_1.history["val_accuracy"], label="Validation Accuracy")
plt.legend()
plt.title("Accuracy Evolution")

#plt.show()
evaluation = model.evaluate(test_generator)
print(f"Test Accuracy: {evaluation[1] * 100:.2f}%")

evaluation = model.evaluate(train_generator)
print(f"Train Accuracy: {evaluation[1] * 100:.2f}%")

y_true = test_generator.classes
y_pred = (model.predict(test_generator) > 0.5).astype("int32")

from sklearn.metrics import confusion_matrix

plt.figure(figsize=(12,5))

confusion_matrix = confusion_matrix(y_true, y_pred)
sns.heatmap(confusion_matrix, annot=True, fmt="d")

plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)

plt.show()

# Precision, Recall and F1-Score of the model

tn, fp, fn, tp = confusion_matrix.ravel()

precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1_score = (2*precision*recall/(precision+recall))

print("Recall of the model is {:.2f}".format(recall))
print("Precision of the model is {:.2f}".format(precision))
print("F1-Score: {}".format(f1_score))
"""

# Measure and improvement!

# Image Augmentation
train_datagen_2 = ImageDataGenerator(
                rescale = 1./255,
                shear_range = 0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                rotation_range=10,
                fill_mode="nearest")

val_datagen = ImageDataGenerator(
              rescale = 1./255)

train_generator_2 = train_datagen_2.flow_from_dataframe(
                df_train,
                x_col="image",
                y_col="class",
                target_size=(150,150),
                batch_size=32,
                class_mode="binary",
                seed=7)

val_generator = val_datagen.flow_from_dataframe(
                df_val,
                x_col="image",
                y_col="class",
                target_size=(150,150),
                batch_size=32,
                class_mode="binary",
                seed=7)

test_generator = val_datagen.flow_from_dataframe(
                df_test,
                x_col="image",
                y_col="class",
                target_size=(150,150),
                batch_size=32,
                class_mode="binary",
                seed=7,
                shuffle=False
                )

# Hyperparameter tuning
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
lr_reduce = ReduceLROnPlateau(monitor="val_accuracy", factor=0.1, min_delta=0.0001, patience=1, verbose=1)

filepath="weights.keras"
checkpoint = ModelCheckpoint(filepath, monitor="val_accuracy", verbose=1, save_best_only=True, mode="max")

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3,3), activation="relu", padding="same",
                 input_shape=(150,150,3)))

model.add(Conv2D(filters=32, kernel_size=(3,3), activation="relu", padding="same"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64, kernel_size=(3,3), activation="relu", padding="same"))
model.add(Conv2D(filters=64, kernel_size=(3,3), activation="relu", padding="same"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(rate=0.6))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy",
              optimizer=Adam(learning_rate=0.001),
              metrics=["accuracy"])

print(model.summary())
print('Optimized model')

# training the model

'''model_htuning = model.fit(
            train_generator_2,
            epochs=10,
            validation_data=val_generator,
            callbacks=[lr_reduce,checkpoint])

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(model_htuning.history['loss'], label='Training Loss')
plt.plot(model_htuning.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss Evolution')

plt.subplot(2, 2, 2)
plt.plot(model_htuning.history['accuracy'], label='Training Accuracy')
plt.plot(model_htuning.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy Evolution')

plt.show()

evaluation = model.evaluate(test_generator)
print(f"Test Accuracy: {evaluation[1] * 100:.2f}%")

evaluation = model.evaluate(train_generator_2)
print(f"Train Accuracy: {evaluation[1] * 100:.2f}%")

y_true = test_generator.classes
y_pred = (model.predict(test_generator) > 0.5).astype("int32")

from sklearn.metrics import confusion_matrix

plt.figure(figsize=(12,5))

confusion_matrix = confusion_matrix(y_true, y_pred)
sns.heatmap(confusion_matrix, annot=True, fmt="d")

plt.xlabel("Predicted Label", fontsize= 12)
plt.ylabel("True Label", fontsize= 12)

plt.show()
'''


# Transfer Learning
from tensorflow.keras.applications import ResNet152V2
resnet_base_model = ResNet152V2(input_shape=(150, 150, 3), include_top=False, weights='imagenet')

resnet_base_model.summary()
# transfer learning network

model_tl = Sequential()
model_tl.add(resnet_base_model)
model_tl.add(Flatten())

model_tl.add(Dense(1024,activation="relu"))
model_tl.add(BatchNormalization())
model_tl.add(Dropout(rate=0.5))

model_tl.add(Dense(128,activation="relu"))
model_tl.add(BatchNormalization())
model_tl.add(Dropout(rate=0.4))

model_tl.add(Dense(1, activation="sigmoid"))

model_tl.summary()

#freeze initial layers of the network
model_tl.compile(loss="binary_crossentropy",
              optimizer=Adam(learning_rate=0.001),
              metrics=["accuracy"])

resnet_base_model.trainable = False

model_tl_final = model_tl.fit(
          train_generator_2,
          epochs=10,
          validation_data=val_generator,
          callbacks=[lr_reduce,checkpoint])

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(model_tl_final.history['loss'], label='Training Loss')
plt.plot(model_tl_final.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss Evolution')

plt.subplot(2, 2, 2)
plt.plot(model_tl_final.history['accuracy'], label='Training Accuracy')
plt.plot(model_tl_final.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy Evolution')

plt.show()

evaluation = model_tl.evaluate(test_generator)
print(f"Test Accuracy: {evaluation[1] * 100:.2f}%")

evaluation = model_tl.evaluate(train_generator_2)
print(f"Train Accuracy: {evaluation[1] * 100:.2f}%")

y_true = test_generator.classes
y_pred = (model_tl.predict(test_generator) > 0.5).astype("int32")

from sklearn.metrics import confusion_matrix

plt.figure(figsize=(12,5))

confusion_matrix = confusion_matrix(y_true, y_pred)
sns.heatmap(confusion_matrix, annot=True, fmt="d")

plt.xlabel("Predicted Label", fontsize= 12)
plt.ylabel("True Label", fontsize= 12)

plt.show()

# Precision, Recall and F1-Score of the model

tn, fp, fn, tp = confusion_matrix.ravel()

precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1_score = (2*precision*recall/(precision+recall))

print("Recall of the model is {:.2f}".format(recall))
print("Precision of the model is {:.2f}".format(precision))
print('F1-score: {}'.format(f1_score))
