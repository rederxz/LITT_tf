import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Conv2D, ReLU

from utils import c2r, r2c


class DC(Layer):
    def call(self, inputs):
        """
        Args:
            inputs[0]: image, [nb, nt, nx, ny, 2]
            inputs[1]: sampled k-space, [nb, nt, nx, ny, 2]
            inputs[2]: sampling mask, [nb, nt, nx, ny, 2]

        Returns:
            output: [nb, nt, nx, ny, 2]
        """
        assert isinstance(inputs, list)
        img, k_0, m = inputs

        img = r2c(img)  # -> [nb, nt, nx, ny]
        k = tf.signal.fft2d(img)
        k = c2r(k)  # -> [nb, nt, nx, ny, 2]
        k = m * k_0 + (1 - m) * k
        k = r2c(k)  # -> [nb, nt, nx, ny]
        img = tf.signal.ifft2d(k)
        img = c2r(img)  # -> [nb, nt, nx, ny, 2]

        return img


class CRNN_t_i(Layer):
    def __init__(self,
                 out_channels=64,
                 kernel_size=3,
                 dilation_rate=(1, 1),
                 uni_direction=False
                 ):
        super(CRNN_t_i, self).__init__()
        self.out_channels = out_channels
        self.uni_direction = uni_direction
        self.input2hidden = Conv2D(out_channels, kernel_size, dilation_rate=dilation_rate, padding='same')
        self.hidden_t2hidden = Conv2D(out_channels, kernel_size, dilation_rate=dilation_rate, padding='same')
        self.hidden_i2hidden = Conv2D(out_channels, kernel_size, dilation_rate=dilation_rate, padding='same')
        self.act = ReLU()

    def call(self, inputs):
        """
        Args:
            inputs[0]: _input, [nb, nt, nx, ny, nc]
            inputs[1]: hidden_i, [nb, nt, nx, ny, self.out_channels]

        Returns:
            output: [nb, nt, nx, ny, self.out_channels]
        """
        assert isinstance(inputs, list)
        _input = tf.transpose(inputs[0], [1, 0, 2, 3, 4])
        hidden_i = tf.transpose(inputs[1], [1, 0, 2, 3, 4])

        nt, nb, nx, ny, nc = _input.shape

        # forward
        output = []
        hidden_t = tf.zeros_like(hidden_i[0])
        for i in range(nt):  # past time frame
            hidden_t = self.act(
                self.input2hidden(_input[i]) +
                self.hidden_i2hidden(hidden_i[i]) +
                self.hidden_t2hidden(hidden_t)
            )
            output.append(hidden_t)
        output = tf.stack(output, axis=0)

        if not self.uni_direction:
            # backward
            output_b = []
            hidden_t = tf.zeros_like(hidden_i[0])
            for i in range(nt):  # future time frame
                hidden_t = self.act(
                    self.input2hidden(_input[nt - i - 1]) +
                    self.hidden_i2hidden(hidden_i[nt - i - 1]) +
                    self.hidden_t2hidden(hidden_t)
                )
                output_b.append(hidden_t)
            output_b = tf.stack(output_b[::-1], axis=0)
            output = output + output_b

        output = tf.transpose(output, [1, 0, 2, 3, 4])

        return output


class CRNN_i(Layer):
    def __init__(self,
                 out_channels=64,
                 kernel_size=3,
                 dilation_rate=(1, 1)
                 ):
        super(CRNN_i, self).__init__()
        self.input2hidden = Conv2D(out_channels, kernel_size, dilation_rate=dilation_rate, padding='same')
        self.hidden_i2hidden = Conv2D(out_channels, kernel_size, dilation_rate=dilation_rate, padding='same')
        self.act = ReLU()

    def call(self, inputs):
        """
        Args:
            inputs[0]: _input, [nb, nt, nx, ny, nc]
            inputs[1]: hidden_i, [nb, nt, nx, ny, self.out_channels]

        Returns:
            output: [nb, nt, nx, ny, self.out_channels]
        """
        assert isinstance(inputs, list)
        _input, hidden_i = inputs

        output = self.act(
            self.input2hidden(_input) +
            self.hidden_i2hidden(hidden_i)
        )

        return output


class CRNN(Model):
    def __init__(self,
                 hidden_dim=64,
                 kernel_size=3,
                 dilation_rate=(1, 1),
                 iteration=5,
                 uni_direction=False
                 ):
        super(CRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.iteration = iteration

        self.crnn_t_i = CRNN_t_i(hidden_dim, kernel_size, 1, uni_direction)
        self.crnn_i_1 = CRNN_i(hidden_dim, kernel_size, dilation_rate)
        self.crnn_i_2 = CRNN_i(hidden_dim, kernel_size, dilation_rate)
        self.crnn_i_3 = CRNN_i(hidden_dim, kernel_size, dilation_rate)
        self.conv_reduce = Conv2D(2, kernel_size, padding='same')
        self.dcs = [DC() for _ in range(iteration)]

    def call(self, inputs):
        """
        Args:
            inputs[0]: input in image domain, [batch_size, t, width, height, 2]
            inputs[1]: initially sampled elements in k-space, [batch_size, t, width, height, 2]
            inputs[2]: mask corresponding nonzero location, [batch_size, t, width, height, 2]

        Returns:
            [batch_size, t, width, height, 2]
        """
        img, k, m = inputs

        out_crnn_t_i = tf.repeat(tf.zeros_like(img[..., :1]), axis=-1, repeats=self.hidden_dim)
        out_crnn_i_1 = tf.repeat(tf.zeros_like(img[..., :1]), axis=-1, repeats=self.hidden_dim)
        out_crnn_i_2 = tf.repeat(tf.zeros_like(img[..., :1]), axis=-1, repeats=self.hidden_dim)
        out_crnn_i_3 = tf.repeat(tf.zeros_like(img[..., :1]), axis=-1, repeats=self.hidden_dim)

        for i in range(self.iteration):
            out_crnn_t_i = self.crnn_t_i([img, out_crnn_t_i])
            out_crnn_i_1 = self.crnn_i_1([out_crnn_t_i, out_crnn_i_1])
            out_crnn_i_2 = self.crnn_i_2([out_crnn_i_1, out_crnn_i_2])
            out_crnn_i_3 = self.crnn_i_3([out_crnn_i_2, out_crnn_i_3])
            out = self.conv_reduce(out_crnn_i_3)
            img = out + img
            img = self.dcs[i]([img, k, m])

        return img
