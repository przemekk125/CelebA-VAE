import os
import pandas as pd
import tensorflow as tf

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

def make_image_dataset(filepaths: list,img_size: tuple[int, int] = (128,128),
    batch_size: int = 64,shuffle: bool = False,buffer_size: int = 10000) -> tf.data.Dataset:
    """
    Returns a tf.data.Dataset yielding batches of normalized images.
    """
    ds = tf.data.Dataset.from_tensor_slices(filepaths)
    if shuffle:
        ds = ds.shuffle(buffer_size)

    def _load_and_preprocess(path,img_size=img_size):
        # Read raw bytes
        raw = tf.io.read_file(path)
        # Decode JPEG, force RGB
        img = tf.image.decode_jpeg(raw, channels=3)
        # Resize with bilinear interpolation
        img = tf.image.resize(img, img_size)
        # Normalize to [0,1]
        img = tf.cast(img, tf.float32) / 255.0
        return img

    # Parallelize map and prefetch
    ds = ds.map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size,drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds