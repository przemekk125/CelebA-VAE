import os
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np

def make_image_dataset(df,data_path="../data/img_align_celeba/img_align_celeba/",img_size = (64,64),
    batch_size = 64,shuffle= False,buffer_size = 10000):
    """
    Returns a tf.data.Dataset yielding batches of normalized images.
    """
    files = df['image_id'].tolist()
    bboxes = df[['y', 'x', 'h', 'w']].values.astype(np.int32)
    ds = tf.data.Dataset.from_tensor_slices((files, bboxes))
    if shuffle:
        ds = ds.shuffle(buffer_size)

    def _load_and_preprocess(path, bbox):
        path = tf.strings.join([data_path, path])
        # Read raw bytes
        raw = tf.io.read_file(path)
        # Decode JPEG, force RGB
        img = tf.image.decode_jpeg(raw, channels=3)
        # Normalize to [0,1]
        img = tf.cast(img, tf.float32) / 255.0
        # Crop and resize to target size
        img = tf.image.crop_to_bounding_box(img, bbox[0], bbox[1], bbox[2], bbox[3])
        img = tf.image.resize(img, img_size)
        return img
    # Map and prefetch
    ds = ds.map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size,drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def plotHistory(history,validation=False):
    fig,axes = plt.subplots(1,2,figsize=(12,5))
    axes[0].plot(history.history['loss'],label='training loss')
    if validation: axes[0].plot(history.history['val_loss'],label='validation loss')
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history.history['loss'],label='training loss')
    if validation: axes[1].plot(history.history['val_loss'],label='validation loss')
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_yscale("log")
    axes[1].legend()