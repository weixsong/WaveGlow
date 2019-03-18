import tensorflow as tf
import numpy as np
from params import hparams


def create_variable(name, shape):
    with tf.device("/cpu:0"):
        initializer = tf.contrib.layers.xavier_initializer_conv2d()
        variable = tf.get_variable(initializer=initializer(shape=shape), name=name)
        return variable


def create_bias_variable(name, shape):
    with tf.device("/cpu:0"):
        initializer = tf.constant_initializer(value=0.0, dtype=tf.float32)
        return tf.get_variable(initializer=initializer(shape=shape), name=name)


def create_variable_zeros(name, shape):
    with tf.device("/cpu:0"):
        initializer = tf.constant_initializer(0.)
        variable = tf.get_variable(initializer=initializer(shape=shape), name=name)
        return variable


def time_to_batch(value, dilation, name=None):
    with tf.name_scope('time_to_batch'):
        shape = tf.shape(value)
        pad_elements = dilation - 1 - (shape[1] + dilation - 1) % dilation
        padded = tf.pad(value, [[0, 0], [0, pad_elements], [0, 0]])
        reshaped = tf.reshape(padded, [-1, dilation, shape[2]])
        transposed = tf.transpose(reshaped, perm=[1, 0, 2])
        return tf.reshape(transposed, [shape[0] * dilation, -1, shape[2]])


def batch_to_time(value, dilation, name=None):
    with tf.name_scope('batch_to_time'):
        shape = tf.shape(value)
        prepared = tf.reshape(value, [dilation, -1, shape[2]])
        transposed = tf.transpose(prepared, perm=[1, 0, 2])
        return tf.reshape(transposed,
                          [tf.div(shape[0], dilation), -1, shape[2]])


def causal_conv(value, filter_, dilation, filter_width=3, name='causal_conv'):
    with tf.name_scope(name):
        # Pad beforehand to preserve causality.
        pad = int((filter_width - 1) * dilation / 2)
        padding = [[0, 0], [pad, pad], [0, 0]]
        padded = tf.pad(value, padding)
        if dilation > 1:
            transformed = time_to_batch(padded, dilation)
            conv = tf.nn.conv1d(transformed, filter_, stride=1, padding='VALID')
            restored = batch_to_time(conv, dilation)
        else:
            restored = tf.nn.conv1d(padded, filter_, stride=1, padding='VALID')
        # Remove excess elements at the end.
        result = tf.slice(restored,
                          [0, 0, 0],
                          [-1, tf.shape(value)[1], -1])
        return result


def compute_waveglow_loss(z, log_s_list, log_det_W_list, sigma=1.0):
    '''negative log-likelihood of the data x'''
    for i, log_s in enumerate(log_s_list):
        if i == 0:
            log_s_total = tf.reduce_sum(log_s)
            log_det_W_total = log_det_W_list[i]
        else:
            log_s_total = log_s_total + tf.reduce_sum(log_s)
            log_det_W_total += log_det_W_list[i]

        tf.summary.scalar('logdet_%d' % i, log_det_W_list[i])
        tf.summary.scalar('log_s_%d' % i, tf.reduce_sum(log_s))

    loss = tf.reduce_sum(z * z) / (2 * sigma * sigma) - log_s_total - log_det_W_total
    shape = tf.shape(z)
    total_size = tf.cast(shape[0] * shape[1] * shape[2], 'float32')
    loss = loss / total_size

    tf.summary.scalar('mean_log_det', -log_det_W_total / total_size)
    tf.summary.scalar('mean_log_scale', -log_s_total / total_size)
    tf.summary.scalar('prior_loss', tf.reduce_sum(z * z / (2 * sigma * sigma)) / total_size)
    tf.summary.scalar('total_loss', loss)
    return loss


def invertible1x1Conv(z, n_channels, forward=True, name='inv1x1conv'):
    with tf.variable_scope(name):
        shape = tf.shape(z)
        batch_size, length, channels = shape[0], shape[1], shape[2]

        # sample a random orthogonal matrix to initialize weight
        W_init = np.linalg.qr(np.random.randn(n_channels, n_channels))[0].astype('float32')
        W = tf.get_variable('W', initializer=W_init, dtype=tf.float32)

        # compute log determinant
        det = tf.log(tf.abs(tf.cast(tf.matrix_determinant(tf.cast(W, tf.float64)), tf.float32)))
        logdet = det * tf.cast(batch_size * length, 'float32')
        if forward:
            _W = tf.reshape(W, [1, n_channels, n_channels])
            z = tf.nn.conv1d(z, _W, stride=1, padding='SAME')
            return z, logdet
        else:
            _W = tf.matrix_inverse(W)
            _W = tf.reshape(_W, [1, n_channels, n_channels])
            z = tf.nn.conv1d(z, _W, stride=1, padding='SAME')
            return z


class WaveNet(object):
    def __init__(self, n_in_channels, n_lc_dim, n_layers,
                 residual_channels=512, skip_channels=256, kernel_size=3, name='wavenet'):
        self.n_in_channels = n_in_channels
        self.n_lc_dim = n_lc_dim  # 80 * 8
        self.n_layers = n_layers
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.kernel_size = kernel_size
        self.name = name

    def create_network(self, audio_batch, lc_batch):
        with tf.variable_scope(self.name):
            # channel convert
            w_s = create_variable('w_s', [1, self.n_in_channels, self.residual_channels])
            b_s = create_bias_variable('b_s', [self.residual_channels])
            g_s = create_variable('g_s', [self.residual_channels])
            # weight norm
            w_s = g_s * tf.nn.l2_normalize(w_s, axis=[0, 1])
            audio_batch = tf.nn.bias_add(tf.nn.conv1d(audio_batch, w_s, 1, 'SAME'), b_s)

            skip_outputs = []
            for i in range(self.n_layers):
                dilation = 2 ** i
                audio_batch, _skip_output = self.dilated_conv1d(audio_batch, lc_batch, dilation)
                skip_outputs.append(_skip_output)

            # post process
            skip_output = sum(skip_outputs)
            # learn scale and shift
            w_e = create_variable_zeros('w_e', [1, self.skip_channels, self.n_in_channels * 2])
            b_e = create_bias_variable('b_e', [self.n_in_channels * 2])
            audio_batch = tf.nn.bias_add(tf.nn.conv1d(skip_output, w_e, 1, 'SAME'), b_e)
            return audio_batch[:, :, :self.n_in_channels], audio_batch[:, :, self.n_in_channels:]

    def dilated_conv1d(self, audio_batch, lc_batch, dilation=1):
        input = audio_batch
        with tf.variable_scope('dilation_%d' % (dilation,)):
            # compute gate & filter
            w_g_f = create_variable('w_g_f', [self.kernel_size, self.residual_channels, 2 * self.residual_channels])
            b_g_f = create_bias_variable('b_g_f', [2 * self.residual_channels])
            g_g_f = create_variable('g_g_f', [2 * self.residual_channels])

            # weight norm
            w_g_f = g_g_f * tf.nn.l2_normalize(w_g_f, [0, 1])

            # dilated conv1d
            audio_batch = causal_conv(audio_batch, w_g_f, dilation, self.kernel_size)

            # process local condition
            w_lc = create_variable('w_lc', [1, self.n_lc_dim, 2 * self.residual_channels])
            b_lc = create_bias_variable('b_lc', [2 * self.residual_channels])
            g_lc = create_variable('g_lc', [2 * self.residual_channels])
            # weight norm
            w_lc = g_lc * tf.nn.l2_normalize(w_lc, [0, 1])

            lc_batch = tf.nn.bias_add(tf.nn.conv1d(lc_batch, w_lc, 1, 'SAME'), b_lc)

            # gated conv
            in_act = audio_batch + lc_batch  # add local condtion
            filter = tf.nn.tanh(in_act[:, :, :self.residual_channels])
            gate = tf.nn.sigmoid(in_act[:, :, self.residual_channels:])
            acts = gate * filter

            # skip
            w_skip = create_variable('w_skip', [1, self.residual_channels, self.skip_channels])
            b_skip = create_bias_variable('b_skip', [self.skip_channels])
            g_skip = create_variable('g_skip', [self.skip_channels])
            # weight norm
            w_skip = g_skip * tf.nn.l2_normalize(w_skip, [0, 1])
            skip_output = tf.nn.bias_add(tf.nn.conv1d(acts, w_skip, 1, 'SAME'), b_skip)

            # residual conv1d
            w_res = create_variable('w_res', [1, self.residual_channels, self.residual_channels])
            b_res = create_bias_variable('b_res', [self.residual_channels])
            # weight norm
            g_res = create_variable('g_res', [self.residual_channels])
            w_res = g_res * tf.nn.l2_normalize(w_res)

            res_output = tf.nn.bias_add(tf.nn.conv1d(acts, w_res, 1, 'SAME'), b_res)

            return res_output + input, skip_output


class WaveGlow(object):
    def __init__(self, lc_dim=80, n_flows=12, n_group=8, n_early_every=4, n_early_size=2):
        self.lc_dim = lc_dim
        self.n_flows = n_flows
        self.n_group = n_group
        self.n_early_every = n_early_every
        self.n_early_size = n_early_size
        self.n_remaining_channels = n_group

    def create_forward_network(self, audio_batch, lc_batch, name='Waveglow'):
        '''
        :param audio_batch: B*T*1
        :param lc_batch: B*T*80, upsampled by directly repeat or transposed conv
        :param name:
        :return:
        '''
        with tf.variable_scope(name):
            # TODO: make local condition interleveled in each dimension
            batch, length = tf.shape(audio_batch)[0], tf.shape(audio_batch)[1]

            # sequeeze
            audio_batch = tf.reshape(audio_batch, [batch, -1, self.n_group])  # B*T'*8
            lc_batch = tf.reshape(lc_batch, [batch, -1, self.lc_dim * self.n_group])  # B*T'*640

            output_audio = []
            log_s_list = []
            log_det_W_list = []

            for k in range(0, self.n_flows):
                if k % self.n_early_every == 0 and k > 0:
                    output_audio.append(audio_batch[:, :, :self.n_early_size])
                    audio_batch = audio_batch[:, :, self.n_early_size:]
                    self.n_remaining_channels -= self.n_early_size  # update remaining channels

                with tf.variable_scope('glow_%d' % (k,)):
                    # invertiable 1X1 conv
                    audio_batch, log_det_w = invertible1x1Conv(audio_batch, self.n_remaining_channels)
                    log_det_W_list.append(log_det_w)

                    # affine coupling layer
                    n_half = int(self.n_remaining_channels / 2)
                    audio_0, audio_1 = audio_batch[:, :, :n_half], audio_batch[:, :, n_half:]

                    wavenet = WaveNet(n_half, self.lc_dim * self.n_group, hparams.n_layers,
                                      hparams.residual_channels, hparams.skip_channels)
                    log_s, shift = wavenet.create_network(audio_0, lc_batch)
                    audio_1 = audio_1 * tf.exp(log_s) + shift
                    audio_batch = tf.concat([audio_0, audio_1], axis=-1)

                    log_s_list.append(log_s)

            output_audio.append(audio_batch)
            return tf.concat(output_audio, axis=-1), log_s_list, log_det_W_list

    def infer(self, lc_batch, sigma=1.0, name='Waveglow'):
        with tf.variable_scope(name):
            # compute the remaining channels
            remaining_channels = self.n_group
            for k in range(0, self.n_flows):
                if k % self.n_early_every == 0 and k > 0:
                    remaining_channels = remaining_channels - self.n_early_size

            batch = tf.shape(lc_batch)[0]
            # need to make sure that length of lc_batch be multiple times of n_group
            pad = self.n_group - 1 - (tf.shape(lc_batch)[1] + self.n_group - 1) % self.n_group
            lc_batch = tf.pad(lc_batch, [[0, 0], [0, pad], [0, 0]])
            lc_batch = tf.reshape(lc_batch, [batch, -1, self.lc_dim * self.n_group])

            shape = tf.shape(lc_batch)
            audio_batch = tf.random_normal([shape[0], tf.shape(lc_batch)[1], remaining_channels])
            audio_batch = audio_batch * sigma

            # backward inference
            for k in reversed(range(0, self.n_flows)):
                with tf.variable_scope('glow_%d' % (k,)):
                    # affine coupling layer
                    n_half = int(remaining_channels / 2)
                    audio_0, audio_1 = audio_batch[:, :, :n_half], audio_batch[:, :, n_half:]
                    wavenet = WaveNet(n_half, self.lc_dim * self.n_group, hparams.n_layers,
                                      hparams.residual_channels, hparams.skip_channels)
                    log_s, shift = wavenet.create_network(audio_0, lc_batch)
                    audio_1 = (audio_1 - shift) / tf.exp(log_s)
                    audio_batch = tf.concat([audio_0, audio_1], axis=-1)

                    # inverse 1X1 conv
                    audio_batch = invertible1x1Conv(audio_batch, remaining_channels, forward=False)

                # early output
                if k % self.n_early_every == 0 and k > 0:
                    z = tf.random_normal([shape[0], tf.shape(lc_batch)[1], self.n_early_size])
                    z = z * sigma
                    remaining_channels += self.n_early_size

                    audio_batch = tf.concat([z, audio_batch], axis=-1)

            # reshape audio back to B*T*1
            audio_batch = tf.reshape(audio_batch, [shape[0], -1, 1])
            return audio_batch
