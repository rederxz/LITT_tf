import math
from toolz import curry

import tensorflow as tf

import numpy as np


def normal_pdf(length, sensitivity):
    return np.exp(-sensitivity * (np.arange(length) - length / 2)**2)


def normal_pdf_tf(length, sensitivity):
    return tf.exp(-sensitivity * (tf.range(length, dtype=tf.float32) - length / 2) ** 2)


@ curry
def cs_mask(shape, acc, sample_n=8, centred=False):
    """
    Sampling density estimated from implementation of kt FOCUSS

    shape: tuple - of form (..., nx, ny)
    acc: float - doesn't have to be integer 4, 8, etc..
    sample_n: preserve how many center lines (sample_n // 2 each side)
    """
    N, Nx, Ny = int(np.prod(shape[:-2])), shape[-2], shape[-1]
    pdf_x = normal_pdf(Nx, 0.5/(Nx/10.)**2)
    lmda = Nx/(2.*acc)
    n_lines = int(Nx / acc)

    # add uniform distribution
    pdf_x += lmda * 1./Nx

    if sample_n:
        pdf_x[Nx//2-sample_n//2:Nx//2+sample_n//2] = 0
        pdf_x /= np.sum(pdf_x)
        n_lines -= sample_n

    mask = np.zeros((N, Nx))
    for i in range(N):
        idx = np.random.choice(Nx, n_lines, False, pdf_x)
        mask[i, idx] = 1

    if sample_n:
        mask[:, Nx//2-sample_n//2:Nx//2+sample_n//2] = 1

    size = mask.itemsize
    mask = np.lib.stride_tricks.as_strided(mask, (N, Nx, Ny), (size * Nx, size, 0))

    mask = mask.reshape(shape)

    if not centred:
        mask = np.fft.ifftshift(mask, axes=(-1, -2))

    return mask


@ curry
def kt_blast_mask(shape, acc, sample_n=10, centred=False):
    N, Nt, Nx, Ny = int(np.prod(shape[:-3])), shape[-3], shape[-2], shape[-1]

    chunk_size = int(acc)  # one line sampled per chunk

    mask = np.zeros((N, Nt, Nx))

    if sample_n:
        mask[..., Nx // 2 - sample_n // 2: Nx // 2 + sample_n // 2] = 1

    step = math.ceil(acc / 3)

    for i in range(N):
        start = np.random.randint(0, chunk_size)
        for j in range(Nt):
            mask[i, j, (start + j * step) % chunk_size::chunk_size] = 1

    mask = np.repeat(mask[..., None], repeats=Ny, axis=-1)

    mask = mask.reshape(shape)

    if not centred:
        mask = np.fft.ifftshift(mask, axes=(-1, -2))

    return mask


@ curry
def cs_mask_tf(shape, acc, sample_n=8, centred=False):
    """
    Sampling density estimated from implementation of kt FOCUSS

    shape: tuple - of form (nt, nx, ny)
    acc: float - doesn't have to be integer 4, 8, etc..
    sample_n: preserve how many center lines (sample_n // 2 each side)
    """
    nt, pe, fe = shape

    # get pdf
    pdf = normal_pdf_tf(pe, 0.5/(pe/10.)**2)
    # add uniform distribution
    lmda = pe/(2.*acc)
    pdf += lmda * 1./pe
    # lines sampled
    n_lines = tf.cast(pe / acc, tf.int32)

    # handle center lines
    center_indices = tf.range(pe//2-sample_n//2, pe//2+sample_n//2, dtype=tf.int64)
    center_mask = tf.sparse.SparseTensor(indices=center_indices[..., None],
                                         values=tf.ones_like(center_indices, dtype=tf.float32),
                                         dense_shape=[pe])
    center_mask = tf.sparse.reorder(center_mask)
    center_mask = tf.sparse.to_dense(center_mask)
    pdf = pdf * (1 - center_mask)
    pdf /= tf.reduce_sum(pdf)  # re-normalize
    n_lines -= sample_n

    # sampling non-center lines
    logits = tf.math.log(pdf)
    logits = tf.repeat(logits[None, ...], axis=0, repeats=nt)
    z = -tf.math.log(-tf.math.log(tf.random.uniform(tf.shape(logits), 0, 1)))  # Gumble-max trick
    _, other_indices = tf.nn.top_k(logits + z, n_lines)
    other_indices = tf.cast(other_indices, dtype=tf.int64)

    # merge
    center_indices = tf.repeat(center_indices[None, ...], axis=0, repeats=nt)
    indices = tf.concat([other_indices, center_indices], axis=-1)
    masks = list()
    for i in range(nt):
        mask = tf.sparse.SparseTensor(indices=indices[i][..., None],
                                      values=tf.ones_like(indices[i]),
                                      dense_shape=[pe])
        mask = tf.sparse.reorder(mask)
        mask = tf.sparse.to_dense(mask)
        masks.append(mask)

    masks = tf.stack(masks, axis=0)
    masks = tf.repeat(masks[..., None], axis=-1, repeats=fe)

    if not centred:
        masks = tf.signal.ifftshift(masks, axes=(-1, -2))

    return masks


def cs_mask_gen(**kwargs):
    def gen():
        while True:
            yield cs_mask(**kwargs)
    return gen


def kt_blast_mask_gen(**kwargs):
    def gen():
        while True:
            yield kt_blast_mask(**kwargs)
    return gen
