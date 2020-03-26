import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image

from keras import layers
from keras import models
from keras import callbacks
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.utils import plot_model

base_path = os.getcwd()
if os.path.isdir(base_path):
    os.chdir(base_path)
print(base_path)


epochs = 32
batches = 128

verbose = 2
dropout = 0.05

im_size = 64
title = "bengali_cnn"


bengali_index = ["grapheme", "image_id", "grapheme_root", "vowel_diacritic", "consonant_diacritic"]

train_csv = pd.read_csv("data/train.csv")
test_csv = pd.read_csv("data/test.csv")
class_map_csv = pd.read_csv("data/class_map.csv")
sample_sub_csv = pd.read_csv("data/sample_submission.csv")


def read_parquet_image(index):
    start_time = time.time()
    train_ = pd.read_parquet(f"./data/train_image_data_r1000_{index}.parquet")
    train_ = pd.merge(train_, train_csv, on="image_id")
    print(f"train_image_data_{index} read in {round(time.time() - start_time, 1)} sec.")

    grapheme_ = train_["grapheme_root"].to_numpy()
    vowel_ = train_["vowel_diacritic"].to_numpy()
    conso_ = train_["consonant_diacritic"].to_numpy()

    train_df = train_.drop(bengali_index, axis=1)

    x_train_ = []
    for i in range(train_df.shape[0]):
        image = train_df.iloc[i].to_numpy()
        image = image.reshape((137, 236))
        image = Image.fromarray(image)
        image = image.resize((im_size, im_size))
        image_resize = np.array(image).reshape((im_size, im_size, 1))
        image_resize = image_resize.astype("float32") / 255
        x_train_.append(image_resize)
    x_train_ = np.array(x_train_)

    data_ = [grapheme_, vowel_, conso_, x_train_]

    return data_


class CustomDataGenerator(ImageDataGenerator):
    def flow(
        self,
        x,
        y=None,
        batch_size=batches,
        shuffle=True,
        sample_weight=None,
        seed=None,
        save_to_dir=None,
        save_prefix="",
        save_format="png",
        subset=None,
    ):

        # all the labels array will be concatenated in this single array
        keys = {}
        labels = None
        ordered = []

        for key, value in y.items():
            if labels is None:
                labels = value
            else:
                labels = np.concatenate((labels, value), axis=1)
            keys[key] = value.shape[1]
            ordered.append(key)

        for x, y in super().flow(x, labels, batch_size=batches):
            label_dict = {}
            i = 0
            for label in ordered:
                target_len = keys[label]
                label_dict[label] = y[:, i : i + target_len]
                i += target_len

            yield x, label_dict


input_shape = (im_size, im_size, 1)
inputs = layers.Input(shape=input_shape)
x = layers.Conv2D(filters=32, kernel_size=3, padding="SAME", activation="relu")(inputs)
x = layers.Conv2D(filters=32, kernel_size=3, padding="SAME")(x)
x = layers.LeakyReLU(alpha=0.15)(x)
x = layers.Conv2D(filters=32, kernel_size=3, padding="SAME")(x)
x = layers.LeakyReLU(alpha=0.15)(x)
x = layers.BatchNormalization(momentum=0.15)(x)
x = layers.MaxPool2D(pool_size=(2, 2))(x)
x = layers.Conv2D(filters=32, kernel_size=3, padding="SAME", activation="relu")(x)
x = layers.BatchNormalization(momentum=0.15)(x)
x = layers.Dropout(rate=dropout)(x)

x = layers.Conv2D(filters=32, kernel_size=3, padding="SAME")(x)
x = layers.LeakyReLU(alpha=0.3)(x)
x = layers.BatchNormalization(momentum=0.15)(x)
x = layers.MaxPool2D(pool_size=(2, 2))(x)
x = layers.Conv2D(filters=32, kernel_size=3, padding="SAME", activation="relu")(x)
x = layers.Conv2D(filters=32, kernel_size=3, padding="SAME", activation="relu")(x)
x = layers.BatchNormalization(momentum=0.15)(x)
x = layers.MaxPool2D(pool_size=(2, 2))(x)
x = layers.Dropout(rate=dropout)(x)

x = layers.Conv2D(filters=64, kernel_size=3, padding="SAME", activation="relu")(x)
x = layers.Conv2D(filters=64, kernel_size=3, padding="SAME", activation="relu")(x)
x = layers.BatchNormalization(momentum=0.15)(x)
x = layers.MaxPool2D(pool_size=(2, 2))(x)
x = layers.Dropout(rate=dropout)(x)

x = layers.Conv2D(filters=128, kernel_size=3, padding="SAME", activation="relu")(x)
x = layers.Conv2D(filters=128, kernel_size=3, padding="SAME", activation="relu")(x)
x = layers.BatchNormalization(momentum=0.15)(x)
x = layers.MaxPool2D(pool_size=(2, 2))(x)
x = layers.Dropout(rate=dropout)(x)

x = layers.Flatten()(x)
x = layers.Dense(1024)(x)
x = layers.LeakyReLU(alpha=0.3)(x)
x = layers.Dropout(rate=dropout)(x)
x = layers.Dense(512, activation="relu")(x)

grapheme = layers.Dense(168, activation="softmax", name="grapheme")(x)
vowel = layers.Dense(11, activation="softmax", name="vowel")(x)
conso = layers.Dense(7, activation="softmax", name="consonant")(x)

model = models.Model(inputs=inputs, outputs=[grapheme, vowel, conso])
model.summary()

adam = optimizers.Adam(learning_rate=0.005, beta_1=0.9, beta_2=0.999, amsgrad=False)
loss = ["categorical_crossentropy", "categorical_crossentropy", "categorical_crossentropy"]
loss_weights = [1.0, 1.0, 1.0]

model.compile(optimizer=adam, loss=loss, loss_weights=loss_weights, metrics=["acc"])

callback_list = [
    callbacks.EarlyStopping(monitor="val_grapheme_acc", patience=20),
    callbacks.ModelCheckpoint(
        monitor="val_grapheme_loss", filepath="results/best_modelcheck.h5", save_best_only=True
    ),
    callbacks.ReduceLROnPlateau(monitor="loss", factor=0.2, patience=2, min_lr=0.00001, verbose=1),
    callbacks.TensorBoard(log_dir=f"./logs/{time.time()}"),
]

hist_ = []
for i in range(4):
    data = read_parquet_image(i)

    grapheme_root = to_categorical(data[0])
    vowel_diacritic = to_categorical(data[1])
    conso_diacritic = to_categorical(data[2])

    (
        x_train,
        x_test,
        y_train_grapheme,
        y_test_grapheme,
        y_train_vowel,
        y_test_vowel,
        y_train_conso,
        y_test_conso,
    ) = train_test_split(
        data[3], grapheme_root, vowel_diacritic, conso_diacritic, test_size=0.1, random_state=10
    )

    datagen = CustomDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=20,
        zoom_range=0.1,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=False,
        vertical_flip=False,
    )

    datagen.fit(x_train)

    history = model.fit_generator(
        datagen.flow(
            x_train,
            {"grapheme": y_train_grapheme, "vowel": y_train_vowel, "consonant": y_train_conso},
        ),
        epochs=epochs,
        verbose=verbose,
        callbacks=callback_list,
        steps_per_epoch=x_train.shape[0] // batches,
        validation_data=(x_test, [y_test_grapheme, y_test_vowel, y_test_conso]),
    )

    hist_.append(history)


plt.style.use("ggplot")
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))

hist_index = [
    "grapheme_acc",
    "vowel_acc",
    "consonant_acc",
    "val_grapheme_acc",
    "val_vowel_acc",
    "val_consonant_acc",
    "grapheme_loss",
    "vowel_loss",
    "consonant_loss",
    "val_grapheme_loss",
    "val_vowel_loss",
    "val_consonant_loss",
]

hist_data = [[] for _ in range(len(hist_index))]
for i in range(4):
    for index, value in enumerate(hist_index):
        hist_data[index] += hist_[i].history[value]

ax1.set_xlabel("epoch")
ax2.set_xlabel("epoch")
ax1.set_ylabel("accuracy")
ax2.set_ylabel("loss")

x = np.arange(len(hist_data[0]))

color_index = ["r-", "g-", "b-", "m-", "y-", "k-", "r--", "g--", "b--", "m--", "y--", "k--"]

for i in range(len(hist_index)):
    if i < 6:
        ax1.plot(x, hist_data[i], color_index[i], label=hist_index[i])
    else:
        ax2.plot(x, hist_data[i], color_index[i], label=hist_index[i])

ax1.legend(loc="upper left")
ax2.legend(loc="upper left")


def read_test_parquet(index):
    start_time = time.time()
    test_ = pd.read_parquet(f"./data/test_image_data_{index}.parquet")
    print(f"test_image_data_{index} read in {round(time.time() - start_time, 1)} sec.")

    image_id_ = test_["image_id"]
    test_ = test_.drop(["image_id"], axis=1)

    x_test_ = []
    for i in range(test_.shape[0]):
        image = test_.iloc[i].to_numpy()
        image = image.reshape((137, 236))
        image = Image.fromarray(image)
        image = image.resize((im_size, im_size))
        image_resize = np.array(image).reshape((im_size, im_size, 1))
        image_resize = image_resize.astype("float32") / 255
        x_test_.append(image_resize)
    x_test_ = np.array(x_test_)

    return x_test_, image_id_


x_test = []
image_id = []
for i in range(4):
    data, id = read_test_parquet(i)
    x_test.append(data)
    image_id.append(id)
x_test = np.array(x_test).reshape(-1, im_size, im_size, 1)
image_id = np.array(image_id).reshape(-1)

preds = model.predict(x_test)

comp = ["grapheme_root", "vowel_diacritic", "consonant_diacritic"]

preds_value = []
for i in range(len(comp)):
    preds_value.append(np.argmax(preds[i], axis=1))

row_id = []
target = []
for i in range(x_test.shape[0]):
    for j in range(len(comp)):
        id = f"{image_id[i]}_{comp[j]}"
        row_id.append(id)
        target.append(preds_value[j][i])

sub_df = pd.DataFrame()
sub_df["row_id"] = row_id
sub_df["target"] = target

sub_df.to_csv("results/submission.csv", index=False)

plot_model(model, to_file=f"results/{title}_model.png")
model.save(f"results/{title}_model.h5")
model.save_weights(f"results/{title}_model_weight.h5")
plt.savefig(f"results/{title}.png")
plt.show()
