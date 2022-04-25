import sys
sys.path.append('LITT_tf')

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecay
from toolz import curry

from data import cut_edge, under_sample, img_ds_from_file, mask_ds_from_gen, mask_ds_from_file, prepare
from mask import cs_mask_gen
from model import CRNN
from utils import setup_tpu, l2_loss, psnr, ssim, c2r


@curry
def transform(img, mask, training):
    if training:  # data manipulation
        img = cut_edge(img, (0, 32))
        mask = cut_edge(mask, (0, 32))
    mask = tf.cast(mask, img.dtype) * (1 + 1j)
    img_u, k_u = under_sample(img, mask)  # under_sample
    img, img_u, k_u, mask = c2r(img), c2r(img_u), c2r(k_u), c2r(mask)  # to real
    return (img_u, k_u, mask), img


nt_network = 6
train_batch_size = 1
test_batch_size = 8
base_lr = 0.0001
epochs = 200

tpu = setup_tpu()

ds_img_train = img_ds_from_file(data_dir='LITT_data', split_dir='data_small/train', nt_network=nt_network)
ds_mask_train = mask_ds_from_gen(cs_mask_gen(shape=[nt_network, 256, 256], acc=8.0, sample_n=8))
ds_img_test = img_ds_from_file(data_dir='LITT_data', split_dir='data_small/test', nt_network=nt_network)
ds_mask_test = mask_ds_from_file(data_dir='LITT_mask_8x_c8', split_dir='data_small/test', nt_network=nt_network)

ds_train = prepare(ds_img_train, ds_mask_train, batch_size=train_batch_size, transform=transform(training=True),
                   shuffle=True)
ds_test = prepare(ds_img_test, ds_mask_test, batch_size=test_batch_size, transform=transform(training=False))

with tpu.scope():
    model = CRNN()
    model.build([(None, nt_network, 256, 256, 2), (None, nt_network, 256, 256, 2), (None, nt_network, 256, 256, 2)])
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
