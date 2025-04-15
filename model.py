import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np

# Import the pure-TF dataset builder
from dataset import build_dataset


##############################################################################
# 1) A small helper function for skip connections that forcibly crops one
#    feature map to match the other’s spatial dims (avoiding off-by-1 errors).
##############################################################################

def crop_and_concat(up, skip):
    """
    Symbolic layer:
      - up: (None,H_up,W_up,C_up)
      - skip: (None,H_skip,W_skip,C_skip)
    We'll slice 'up' to [0..H_skip, 0..W_skip], then concat along channels.
    """
    def layer_func(tensors):
        up, skip = tensors
        sh = tf.shape(skip)[1]
        sw = tf.shape(skip)[2]
        up = up[:, :sh, :sw, :]
        return tf.concat([up, skip], axis=-1)

    return layers.Lambda(layer_func)([up, skip])


##############################################################################
# 2) FlowNetSimple architecture: 6 downsampling steps, then 4 up-convs
#    -> final resolution is 1/4 of the input. We do partial decoding.
##############################################################################

def build_flownet_simple(input_shape=(384, 512, 6), output_channels=1):
    """
    FlowNetSimple style:
      - 6 strided conv layers (conv1..conv6)
      - partial decoder with upconv5..upconv2
      - 'crop_and_concat' merges skip connections
      - final output: 1/4 scale of input
    """
    inputs = keras.Input(shape=input_shape)

    # ---- Downsampling / Encoder ----
    conv1 = layers.Conv2D(64, 7, strides=2, padding='same', activation='relu')(inputs)
    conv2 = layers.Conv2D(128, 5, strides=2, padding='same', activation='relu')(conv1)
    conv3 = layers.Conv2D(256, 5, strides=2, padding='same', activation='relu')(conv2)
    conv3_1 = layers.Conv2D(256, 3, strides=1, padding='same', activation='relu')(conv3)
    conv4 = layers.Conv2D(512, 3, strides=2, padding='same', activation='relu')(conv3_1)
    conv4_1 = layers.Conv2D(512, 3, strides=1, padding='same', activation='relu')(conv4)
    conv5 = layers.Conv2D(512, 3, strides=2, padding='same', activation='relu')(conv4_1)
    conv5_1 = layers.Conv2D(512, 3, strides=1, padding='same', activation='relu')(conv5)
    conv6 = layers.Conv2D(1024, 3, strides=2, padding='same', activation='relu')(conv5_1)

    # ---- Partial Decoder (Upsampling) ----
    up5 = layers.Conv2DTranspose(512, kernel_size=4, strides=2, padding='same', activation='relu')(conv6)
    skip5 = crop_and_concat(up5, conv5_1)

    up4 = layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding='same', activation='relu')(skip5)
    skip4 = crop_and_concat(up4, conv4_1)

    up3 = layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same', activation='relu')(skip4)
    skip3 = crop_and_concat(up3, conv3_1)

    up2 = layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding='same', activation='relu')(skip3)
    skip2 = crop_and_concat(up2, conv2)

    # final prediction => shape ~ [N, H/4, W/4, output_channels]
    prediction = layers.Conv2D(output_channels, 3, padding='same', activation='linear')(skip2)

    model = keras.Model(inputs=inputs, outputs=prediction, name='FlowNetSimple')
    return model


##############################################################################
# 3) Extra metrics: MSE, fraction of pixels with error < 3
##############################################################################

def mse_metric(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_pred - y_true))

def acc3_metric(y_true, y_pred):
    """
    Fraction of pixels whose absolute error is < 3.0
    => returns a float in [0..1]
    """
    err = tf.abs(y_pred - y_true)
    within3 = tf.less(err, 3.0)
    return tf.reduce_mean(tf.cast(within3, tf.float32))

def endpoint_error(y_true, y_pred):
    """
    Standard L1 difference (for single-channel). Could also do full EPE for 2-channel flow.
    """
    return tf.reduce_mean(tf.abs(y_true - y_pred))


##############################################################################
# 4) Main script:
#    - build the dataset via build_dataset(...)
#    - build the FlowNetSimple
#    - compile with extra metrics
#    - train & evaluate
##############################################################################

def main():
    dataset_path = 'dataset'
    # This calls our pure TF-based dataset function from dataset_keras.py
    X, Y = build_dataset(dataset_path)  # X shape: (N,H_sub,W_sub,6), Y shape: (N,H_sub/4,W_sub/4,1)
    print("Loaded data shapes:", X.shape, Y.shape)

    # We build the FlowNet
    _, H_sub, W_sub, _ = X.shape
    model = build_flownet_simple(
        input_shape=(H_sub, W_sub, 6),  # e.g. (924,1148,6) from your dataset
        output_channels=1
    )
    model.summary()

    # Compile with multiple metrics
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="mae",  # or endpoint_error
        metrics=[endpoint_error, mse_metric, acc3_metric]
    )

    # Train
    model.fit(
        X, Y,
        batch_size=2,
        epochs=5,
    )

    # Evaluate
    results = model.evaluate(X, Y)
    print("Evaluation:", results)
    # Typically => [loss, endpoint_error, mse_metric, acc3_metric]

if __name__ == '__main__':
    main()
