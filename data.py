import pathlib

import numpy as np
import tensorflow as tf
from scipy.io import loadmat


def under_sample(img, mask):
    k = tf.signal.fft2d(img)
    k_u = mask * k
    img_u = tf.signal.ifft2d(k_u)

    return img_u, k_u


def cut_edge(img, edge=(0, 0)):
    return img[
           ...,
           edge[0]:None if edge[0] == 0 else -edge[0],
           edge[1]:None if edge[1] == 0 else -edge[1]]


def parse_path(data_dir, split_dir):
    base_dir = (pathlib.Path(__file__).parent.absolute() / pathlib.Path(split_dir)).resolve()
    mat_file_path = sorted([pathlib.Path(data_dir) / (x.name + '.mat') for x in base_dir.iterdir()])

    return mat_file_path


def cut_to_chunks(array, nt_network, overlap, nt_wait):
    """array: [t, x, y]"""
    chunks = list()

    if nt_wait > 0:  # the preparation stage
        assert nt_wait < nt_network, f'nt_wait({nt_wait}) must be smaller than nt_network({nt_network})'
        for i in range(nt_wait, nt_network):
            chunks.append(array[:i])

    total_t = array.shape[-3]

    if not overlap:  # slice the data along time dim according to nt_network with no overlapping
        complete_slice = total_t // nt_network
        for i in range(complete_slice):
            chunks.append(array[i * nt_network:(i + 1) * nt_network])
        if total_t % nt_network > 0:
            chunks.append(array[total_t - nt_network:total_t])
    else:  # ... with overlapping
        for i in range(total_t - (nt_network - 1)):
            chunks.append(array[i * nt_network:(i + 1) * nt_network])

    return chunks


def read_img_to_chunks(img_file_path,
                       nt_network=1,
                       overlap=False,
                       nt_wait=0):
    img_chunks = list()
    for idx, img_file in enumerate(img_file_path):
        mat_data = loadmat(img_file)
        img = mat_data['mFFE_img_real'] + 1j * mat_data['mFFE_img_imag']  # [x, y, time, echo]
        img = img[..., 1]  # -> [x, y, time]
        img = np.transpose(img, (2, 0, 1))  # -> [time, x, y]
        img_chunks += cut_to_chunks(img, nt_network, overlap, nt_wait)

    return img_chunks


def read_mask_to_chunks(mask_file_path,
                        nt_network=1,
                        overlap=False,
                        nt_wait=0):
    mask_chunks = list()
    for idx, mask_file in enumerate(mask_file_path):
        mat_data = loadmat(mask_file)
        mask = mat_data['mask']  # [x, y, time]
        mask = np.transpose(mask, (2, 0, 1))  # -> [time, x, y]
        mask_chunks += cut_to_chunks(mask, nt_network, overlap, nt_wait)

    return mask_chunks


def img_ds_from_file(data_dir,
                     split_dir,
                     nt_network=1,
                     overlap=False,
                     nt_wait=0):
    mat_file_path = parse_path(data_dir, split_dir)
    img_chunks = read_img_to_chunks(mat_file_path, nt_network, overlap, nt_wait)
    ds = tf.data.Dataset.from_tensor_slices(img_chunks)

    return ds


def mask_ds_from_file(data_dir,
                      split_dir,
                      nt_network=1,
                      overlap=False,
                      nt_wait=0):
    """for testing"""
    mat_file_path = parse_path(data_dir, split_dir)
    mask_chunks = read_mask_to_chunks(mat_file_path, nt_network, overlap, nt_wait)
    mask_ds = tf.data.Dataset.from_tensor_slices(mask_chunks)

    return mask_ds


def mask_ds_from_gen(gen):
    """for training"""
    mask_ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=tf.TensorSpec(shape=(None, None, None), dtype=tf.float32))

    return mask_ds


def prepare(ds_img,
            ds_mask,
            batch_size=1,
            transform=None,
            shuffle=False):
    if shuffle:
        ds_img = ds_img.shuffle(len(ds_img), reshuffle_each_iteration=True)  # a large enough buffer size is required

    ds = tf.data.Dataset.zip((ds_img, ds_mask))

    if transform:
        ds = ds.map(lambda img, mask: transform(img, mask), num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size)

    return ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# def make_LITT_dataset(data_dir,
#                       split,
#                       nt_network=1,
#                       overlap=False,
#                       nt_wait=0,
#                       batch_size=1,
#                       shuffle=False,
#                       transform=None
#                       ):
#     # read original images and create a dataset
#     data_root = pathlib.Path(data_dir)
#     base_dir = (pathlib.Path(__file__).parent.absolute() / pathlib.Path(split)).resolve()
#     mat_file_path = sorted([data_root/(x.name + '.mat') for x in base_dir.iterdir()])
#     ds = LITT(mat_file_path, nt_network, overlap, nt_wait)
#
#     # preprocessing (data augmentation, under-sampling ...)
#     ds = prepare(ds, batch_size, transform, shuffle)
#
#     return ds


# def prepare(ds, batch_size=1, transform=None, shuffle=False):
#
#     if transform:
#         ds = ds.map(lambda img: transform(img), num_parallel_calls=tf.data.AUTOTUNE)
#
#     if shuffle:
#         ds = ds.shuffle(len(ds))  # a large enough buffer size is required
#
#     ds = ds.batch(batch_size)
#
#     return ds.prefetch(buffer_size=tf.data.AUTOTUNE)
