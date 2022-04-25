from toolz import curry

import tensorflow as tf
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.optimizers import Adam

from utils import setup_tpu, complex_l2_loss, complex_input_psnr, complex_input_ssim, c2r
from data import make_LITT_dataset, cut_edge, under_sample
from model import CRNN


@curry
def transform(img, mask_func, training):
    if training:  # data manipulation
        img = cut_edge(img, (0, 32))
    mask = mask_func(img.shape) * (1 + 1j)
    mask = tf.cast(tf.constant(mask), tf.complex128)
    img_u, k_u = under_sample(img, mask)  # under_sample
    img, img_u, k_u, mask = c2r(img), c2r(img_u), c2r(k_u), c2r(mask)  # to real
    return img, img_u, k_u, mask


nt_network = 6
train_batch_size = 1
test_batch_size = 8
base_lr = 0.0001
epochs = 200

tpu = setup_tpu()

ds_train = make_LITT_dataset(data_dir='/root/LITT_data',
                             split='data/train',
                             nt_network=nt_network,
                             batch_size=train_batch_size,
                             shuffle=True,
                             transform=transform(mask_func=None, training=True))
ds_test = make_LITT_dataset(data_dir='/root/LITT_data',
                            split='data/test',
                            nt_network=nt_network,
                            batch_size=test_batch_size,
                            transform=transform(mask_func=None, training=False))

with tpu.scope():
    model = CRNN()
    model.build([(1, nt_network, 256, 256), (1, nt_network, 256, 256), (1, nt_network, 256, 256)])
    model.summary()

    model.compile(
        optimizer=Adam(CosineDecay(base_lr, epochs), beta_1=0.5, beta_2=0.999),
        loss=complex_l2_loss,
        metrics=[complex_input_psnr, complex_input_ssim]
    )

model.fit(
    ds_train,
    epochs=epochs,
    validation_data=ds_test,
)
