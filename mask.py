import math
from toolz import curry

import numpy as np


def normal_pdf(length, sensitivity):
    return np.exp(-sensitivity * (np.arange(length) - length / 2)**2)


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
