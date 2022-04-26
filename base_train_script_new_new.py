import sys

sys.path.append('LITT_tf')

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecay
from toolz import curry

from data import cut_edge, under_sample, get_litt_ds, prepare
from mask import cs_mask_tf
from model import CRNN
from utils import setup_tpu, l2_loss, psnr, ssim, c2r


@curry
def transform(data, training):
    if training:
        img = data[0]
        img = cut_edge(img, (0, 32))
        mask = cs_mask_tf(img.shape, acc=8.0, sample_n=8)
    else:
        img, mask = data
    mask = tf.cast(mask, tf.complex128) * (1 + 1j)  # mask to complex
    img_u, k_u = under_sample(img, mask)  # under_sample
    img, img_u, k_u, mask = c2r(img), c2r(img_u), c2r(k_u), c2r(mask)  # to real
    return (img_u, k_u, mask), img


nt_network = 6
train_batch_size = 1
test_batch_size = 8
base_lr = 0.0001
epochs = 200

tpu = setup_tpu()

ds_img_train = get_litt_ds('LITT_data', 'data_small/train', nt_network)
ds_train = prepare(ds_img_train, batch_size=train_batch_size, transform=transform(training=True), shuffle=True)
ds_test = get_litt_ds('LITT_data', 'data_small/test', nt_network, mask_dir='LITT_mask_8x_c8')
ds_test = prepare(ds_test, batch_size=8, transform=transform(training=False))

with tpu.scope():
    model = CRNN()
    model.build([(None, nt_network, 256, 256, 2),
                 (None, nt_network, 256, 256, 2),
                 (None, nt_network, 256, 256, 2)])
    model.summary()

    model.compile(
        optimizer=Adam(CosineDecay(base_lr, epochs), beta_1=0.5, beta_2=0.999),
        loss=l2_loss,
        metrics=[psnr, ssim]
    )

model.fit(
    ds_train,
    epochs=epochs,
    validation_data=ds_test,
)
