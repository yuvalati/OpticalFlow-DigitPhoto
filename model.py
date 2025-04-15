# model.py

import os
import numpy as np
import torch
from torch.utils.data import DataLoader

import tensorflow as tf
from tensorflow import keras
from keras import layers

from dataset import FlowDataset


##############################################################################
# 1) Load dataset => X is [N,H,W,6], Y is [N,H/4,W/4,1], from your final dataset code
##############################################################################

def load_dataset_into_numpy(dataset_path):
    ds = FlowDataset(root_dir=dataset_path, transform=None)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    X_list, Y_list = [], []
    for sample in loader:
        im0 = sample['im0'][0].numpy()      # [3,H,W]
        im1 = sample['im1'][0].numpy()      # [3,H,W]
        disp0 = sample['disp0'][0].numpy()  # [1,H/4,W/4]

        # (H,W,6)
        stacked = np.concatenate([im0, im1], axis=0)  # [6,H,W]
        stacked = np.transpose(stacked, (1,2,0))       # => [H,W,6]

        # (H/4,W/4,1)
        disp0 = np.transpose(disp0, (1,2,0))

        X_list.append(stacked.astype(np.float32))
        Y_list.append(disp0.astype(np.float32))

    X = np.array(X_list, dtype=np.float32)  # [N,H,W,6]
    Y = np.array(Y_list, dtype=np.float32)  # [N,H/4,W/4,1]
    print("X.shape:", X.shape, "Y.shape:", Y.shape)
    return X, Y


##############################################################################
# 2) A small layer that: (a) slices the upsampled feature to match the skip,
#    (b) concatenates them along the channels axis.  Fixes shape mismatches.
##############################################################################

def crop_and_concat(up, skip):
    """
    Receives 2 Keras symbolic tensors: up, skip
      - up might be (None, H_up, W_up, C_up)
      - skip might be (None, H_skip, W_skip, C_skip)
    Slices 'up' so [H_up, W_up] -> [H_skip, W_skip] if H_up>H_skip, W_up>W_skip
    Returns concatenated along axis=-1
    """
    def layer_func(tensors):
        up, skip = tensors
        sh = tf.shape(skip)[1]
        sw = tf.shape(skip)[2]
        # slice up
        up = up[:, :sh, :sw, :]
        return tf.concat([up, skip], axis=-1)

    return layers.Lambda(layer_func)([up, skip])


##############################################################################
# 3) FlowNetSimple with 6 downsamples, 4 up.  Each skip uses crop_and_concat.
##############################################################################

def build_flownet_simple(input_shape=(384, 512, 6), output_channels=1):
    inputs = keras.Input(shape=input_shape)

    # -- encoder
    conv1 = layers.Conv2D(64, 7, strides=2, padding='same', activation='relu')(inputs)
    conv2 = layers.Conv2D(128,5, strides=2, padding='same', activation='relu')(conv1)
    conv3 = layers.Conv2D(256,5, strides=2, padding='same', activation='relu')(conv2)
    conv3_1 = layers.Conv2D(256,3, strides=1, padding='same', activation='relu')(conv3)
    conv4 = layers.Conv2D(512,3, strides=2, padding='same', activation='relu')(conv3_1)
    conv4_1 = layers.Conv2D(512,3, strides=1, padding='same', activation='relu')(conv4)
    conv5 = layers.Conv2D(512,3, strides=2, padding='same', activation='relu')(conv4_1)
    conv5_1 = layers.Conv2D(512,3, strides=1, padding='same', activation='relu')(conv5)
    conv6 = layers.Conv2D(1024,3,strides=2, padding='same', activation='relu')(conv5_1)

    # -- partial decoder
    up5 = layers.Conv2DTranspose(512,4,strides=2,padding='same',activation='relu')(conv6)
    skip5 = crop_and_concat(up5, conv5_1)

    up4 = layers.Conv2DTranspose(256,4,strides=2,padding='same',activation='relu')(skip5)
    skip4 = crop_and_concat(up4, conv4_1)

    up3 = layers.Conv2DTranspose(128,4,strides=2,padding='same',activation='relu')(skip4)
    skip3 = crop_and_concat(up3, conv3_1)

    up2 = layers.Conv2DTranspose(64,4,strides=2,padding='same',activation='relu')(skip3)
    skip2 = crop_and_concat(up2, conv2)

    # final => shape ~ [batch, H/4, W/4, 1]
    prediction = layers.Conv2D(output_channels,3,padding='same',activation='linear')(skip2)

    model = keras.Model(inputs, prediction, name='FlowNetSimple')
    return model


##############################################################################
# 4) Train & Evaluate
##############################################################################

def main():
    dataset_path = "dataset"
    X, Y = load_dataset_into_numpy(dataset_path)  # X: [N,H,W,6], Y: [N,H/4,W/4,1]

    _, H, W, C = X.shape
    model = build_flownet_simple(
        input_shape=(H, W, C),
        output_channels=1
    )
    model.summary()

    # For single-channel, you can do "mae" or a custom EPE
    def endpoint_error(y_true, y_pred):
        return tf.reduce_mean(tf.abs(y_true - y_pred))

    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss="mae",
        metrics=[endpoint_error]
    )

    model.fit(
        X, Y,
        batch_size=2,
        epochs=5,
    )

    results = model.evaluate(X, Y)
    print("Evaluation:", results)

if __name__ == "__main__":
    main()
