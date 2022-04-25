import os

import tensorflow as tf
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras import Model


def setup_tpu():
    tpu_address = os.environ.get("COLAB_TPU_ADDR")
    if tpu_address:
        tpu_address = "grpc://" + tpu_address
        tf.keras.backend.clear_session()
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu_address)
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
        return strategy


def r2c(x):
    x_real, x_imag = x[..., 0], x[..., -1]
    return tf.complex(x_real, x_imag)


def c2r(x):
    return tf.stack([tf.math.real(x), tf.math.imag(x)], -1)


def l2_loss(img_true, img_pred, complex_input=False):
    if complex_input:
        img_true, img_pred = c2r(img_true), c2r(img_pred)
    return mean_squared_error(img_true, img_pred)


def psnr(img_ref, img, complex_input=False):  # FIXME: complex PSNR calculation?
    if not complex_input:
        img_ref, img = r2c(img_ref), r2c(img)
    mse = tf.reduce_mean(tf.math.abs(tf.math.abs(img_ref) - tf.math.abs(img)) ** 2)
    return 10 * tf.experimental.numpy.log10(tf.reduce_max(tf.abs(img_ref)) ** 2 / mse)


def ssim(img_1, img_2, complex_input=False):
    if not complex_input:
        img_1, img_2 = r2c(img_1), r2c(img_2)
    img_1 = tf.transpose(img_1, [1, 0, 2, 3])
    img_2 = tf.transpose(img_2, [1, 0, 2, 3])
    print(img_1.shape)
    mean_ssim = tf.reduce_mean([tf.image.ssim(tf.math.abs(img_1[i]), tf.math.abs(img_2[i]), max_val=1.0) for i in range(img_1.shape[0])])
    return mean_ssim


class Learner(Model):
    def train_step(self, data):
        # unpack data
        img, img_u, k_u, mask = data
        with tf.GradientTape() as tape:
            img_pred = self(img_u, k_u, mask, training=True)  # Forward pass
            # Compute the loss value
            loss = self.compiled_loss(img, img_pred, regularization_losses=self.losses)
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Only return the metric that tracks the loss
        return {m.name: m.result() for m in self.compiled_loss.metrics}

    def test_step(self, data):
        # unpack data
        img, img_u, k_u, mask = data
        # forward pass
        img_pred = self(img_u, k_u, mask, training=False)
        # Update metrics
        self.compiled_loss(img, img_pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(img, img_pred)
        return {m.name: m.result() for m in self.metrics}
