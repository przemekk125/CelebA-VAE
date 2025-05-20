import os
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np

def load_partition_csv(csv_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path)
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding='latin1')
    # Expecting two columns without header: filename, partition (0=train,1=val,2=test)
    df.columns = ['filename', 'partition']
    return df

def build_file_lists(df: pd.DataFrame, image_dir: str) -> tuple[list, list, list]:
    # Map partition codes to masks
    train_mask = df['partition'] == 0
    val_mask   = df['partition'] == 1
    test_mask  = df['partition'] == 2

    # Prepend directory to filenames
    df['filepath'] = df['filename'].apply(lambda fn: os.path.join(image_dir, fn))

    train_files = df.loc[train_mask, 'filepath'].tolist()
    val_files   = df.loc[val_mask,   'filepath'].tolist()
    test_files  = df.loc[test_mask,  'filepath'].tolist()

    return train_files, val_files, test_files

def make_image_dataset(filepaths, img_size=(64, 64), batch_size=64, shuffle=False, buffer_size=10000):
    ds = tf.data.Dataset.from_tensor_slices(filepaths)
    if shuffle:
        ds = ds.shuffle(buffer_size)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")    
    def detect_and_crop_face(filepath, scale=1.2, img_size=img_size):
        img_bgr = tf.io.read_file(filepath)
        img_bgr = tf.io.decode_jpeg(img_bgr, channels=3)
        def _detect_and_crop(img, scale):
            faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
            if len(faces) == 0:
                # Return a dummy image and a flag (0 = no face)
                return np.zeros((*img_size, 3), dtype=np.uint8), np.int64(0)
            x, y, w, h = faces[0]
            cx, cy = x + w // 2, y + h // 2
            nw, nh = int(w * scale), int(h * scale)
            nx, ny = max(cx - nw // 2, 0), max(cy - nh // 2, 0)
            nx2, ny2 = min(nx + nw, img.shape[1]), min(ny + nh, img.shape[0])
            cropped = img[ny:ny2, nx:nx2]
            cropped_resized = cv2.resize(cropped, img_size)
            return cropped_resized.astype(np.uint8), np.int64(1)
        cropped, found = tf.numpy_function(_detect_and_crop, [img_bgr, scale], [tf.uint8, tf.int64])
        cropped.set_shape([img_size[0], img_size[1], 3])
        found.set_shape([])
        cropped = tf.cast(cropped, tf.float32) / 255.0
        return cropped, found

    ds = ds.map(detect_and_crop_face, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.filter(lambda img, found: found > 0)
    ds = ds.map(lambda img, found: img)
    ds = ds.batch(batch_size)
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