import tensorflow as tf
import numpy as np
from params import hparams


def noam_scheme(init_lr, global_step, warmup_steps=4000.):
    '''Noam scheme learning rate decay
    init_lr: initial learning rate. scalar.
    global_step: scalar.
    warmup_steps: scalar. During warmup_steps, learning rate increases
        until it reaches init_lr.
    '''
    step = tf.cast(global_step + 1, dtype=tf.float32)
    return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)


def positional_encoding(inputs,
                        maxlen=1024,
                        masking=True,
                        scope="positional_encoding"):
    '''Sinusoidal Positional_Encoding. See 3.5
    inputs: 3d tensor. (N, T, E)
    maxlen: scalar. Must be >= T
    masking: Boolean. If True, padding positions are set to zeros.
    scope: Optional scope for `variable_scope`.
    returns
    3d tensor that has the same shape as inputs.
    '''

    E = inputs.get_shape().as_list()[-1]  # static
    N, T = tf.shape(inputs)[0], tf.shape(inputs)[1]  # dynamic
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # position indices
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])  # (N, T)

        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, (i - i % 2) / E) for i in range(E)]
            for pos in range(maxlen)])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
        position_enc = tf.convert_to_tensor(position_enc, tf.float32)  # (maxlen, E)

        # lookup
        outputs = tf.nn.embedding_lookup(position_enc, position_ind)

        # masks
        if masking:
            outputs = tf.where(tf.equal(inputs, 0), inputs, outputs)

        return tf.to_float(outputs)


def layer_norm(inputs, epsilon=1e-8, scope="ln"):
    '''Applies layer normalization. See https://arxiv.org/abs/1607.06450.
    inputs: A tensor with 2 or more dimensions, where the first dimension has `batch_size`.
    epsilon: A floating number. A very small number for preventing ZeroDivision Error.
    scope: Optional scope for `variable_scope`.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
        gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
        normalized = (inputs - mean) / ((variance + epsilon) ** .5)
        outputs = gamma * normalized + beta

    return outputs


def scaled_dot_product_attention(Q, K, V, is_training=True, mask=None, q_mask=None):
    """
    Args:
        Q (tf.tensor): of shape (h * batch, q_size, d_model)
        K (tf.tensor): of shape (h * batch, k_size, d_model)
        V (tf.tensor): of shape (h * batch, k_size, d_model)
        mask (tf.tensor): of shape (h * batch, q_size, k_size)
    """
    padding_num = -100000
    d = hparams.encoder_conv_channels // hparams.num_heads
    assert d == Q.shape[-1] == K.shape[-1] == V.shape[-1]

    out = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # [h*batch, q_size, k_size]
    out = out / d ** 0.5  # scaled by sqrt(d_k), [h*batch, q_size, k_size]

    if mask is not None:
        # masking out (0.0) => setting to -inf.
        # method 1
        # out = tf.multiply(out, mask) + (1.0 - mask) * (-100000)

        # method 2
        paddings = tf.ones_like(out) * padding_num
        out = tf.where(tf.equal(mask, 0), paddings, out)

    out = tf.nn.softmax(out)  # [h * batch, q_size, k_size]
    attention = tf.transpose(out, [0, 2, 1])
    tf.summary.image("attention", tf.expand_dims(attention[:1], -1))

    if q_mask is not None:
        out = out * q_mask

    out = tf.layers.dropout(out, rate=0.1, training=is_training)
    out = tf.matmul(out, V)  # [h * batch, q_size, d_model]

    return out


def multihead_self_attention(Q, K, V, input_mask, query_mask, is_training, num_heads=8):
    '''

    :param Q: B*T_q*d
    :param K: B*T_k*d
    :param V: B*T_k*d
    :param input_mask: B*T_q*T_k
    :param is_training:
    :param num_heads:
    :return:
    '''

    d_model = Q.get_shape().as_list()[-1]
    with tf.variable_scope('multihead_self_attention', reuse=tf.AUTO_REUSE):
        # Linear projections
        Q = tf.layers.dense(Q, d_model, use_bias=False)  # (N, T_q, d_model)
        K = tf.layers.dense(K, d_model, use_bias=False)  # (N, T_k, d_model)
        V = tf.layers.dense(V, d_model, use_bias=False)  # (N, T_k, d_model)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, d_model/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, d_model/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, d_model/h)
        mask_ = tf.tile(input_mask, [num_heads, 1, 1])  # (h*N, T_q, T_k)

        q_mask = tf.expand_dims(query_mask, axis=-1)  # N, T_q, 1
        q_mask = tf.tile(q_mask, [1, 1, tf.shape(K)[1]])  # N, T_q, T_k
        q_mask = tf.tile(q_mask, [num_heads, 1, 1])  # h*N, T_q, T_k

        # Attention
        outputs = scaled_dot_product_attention(Q_, K_, V_, is_training, mask_, q_mask)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, d_model)

        # concat
        outputs = tf.concat([outputs, Q], axis=-1)

        # Linear projection
        outputs = tf.layers.dense(outputs, d_model, use_bias=False)  # N, T_q, d_model

        # dropout
        outputs = tf.layers.dropout(outputs, rate=0.1, training=is_training)

        # Residual connection
        outputs += Q

        # layer norm
        outputs = layer_norm(outputs)

    return outputs


def ffn(input):
    # input: B*T*d
    with tf.variable_scope('pointwise_feedforward', reuse=tf.AUTO_REUSE):
        output = tf.layers.dense(input, units=1024)
        output = tf.nn.relu(output)
        output = tf.layers.dense(output, units=256)

        output += input  # residual connection
        output = layer_norm(output)

    return output


def conv1d(inputs, kernel_size, channels, activation, dropout_rate, is_training, scope):
    with tf.variable_scope(scope):
        conv1d_output = tf.layers.conv1d(
            inputs,
            kernel_size=kernel_size,
            filters=channels,
            padding='same')

        conv1d_output = tf.layers.batch_normalization(conv1d_output, training=is_training)
        if activation is not None:
            conv1d_output = activation(conv1d_output)

        return tf.layers.dropout(conv1d_output, rate=dropout_rate, training=is_training, name='dropout')


def transformer_encoder(inputs, input_lengths, is_training=True):
    '''

    :param inputs: B*T*d
    :param input_lengths: B
    :param is_training:
    :return:
    '''

    # build input mask
    seq_length = tf.shape(inputs)[1]

    # mask for self-attention
    mask = tf.sequence_mask(input_lengths, maxlen=seq_length)  # B*T
    mask = tf.cast(mask, tf.float32)
    mask = tf.expand_dims(mask, axis=1)  # B*1*T
    mask = tf.tile(mask, [1, seq_length, 1])  # B*T*T

    # query mask
    q_mask = tf.sequence_mask(input_lengths, maxlen=seq_length)  # B*T
    q_mask = tf.cast(q_mask, tf.float32)

    x = inputs
    with tf.variable_scope('transformer_encoder', reuse=tf.AUTO_REUSE):
        # alpha = tf.get_variable('')

        # conv layers
        for i in range(hparams.encoder_conv_layers):
            activation = tf.nn.relu
            x = conv1d(x, hparams.encoder_conv_width, hparams.encoder_conv_channels,
                       activation, dropout_rate=0.2, is_training=is_training, scope='conv_%d' % i)

        # linear projection
        x = tf.layers.dense(x, hparams.encoder_conv_channels)  # B*T*256

        # Position embedding
        x += positional_encoding(x, maxlen=hparams.pos_encoding_maxlen, masking=False)
        x = tf.layers.dropout(x, rate=0.1, training=is_training)  # B*T*d

        inputs = x

        for i in range(hparams.transformer_encoder_layers):
            with tf.variable_scope('layer_{}'.format(i), reuse=tf.AUTO_REUSE):
                # self attention
                x = multihead_self_attention(Q=inputs,
                                             K=inputs,
                                             V=inputs,
                                             input_mask=mask,
                                             query_mask=q_mask,
                                             is_training=is_training,
                                             num_heads=hparams.num_heads)

                # FF
                x = ffn(x)

                inputs = x

    return inputs


def create_variable(name, shape):
    with tf.device("/cpu:0"):
        initializer = tf.contrib.layers.xavier_initializer_conv2d()
        variable = tf.get_variable(initializer=initializer(shape=shape), name=name)
        return variable


def create_variable_init(name, initializer):
    with tf.device("/cpu:0"):
        variable = tf.get_variable(initializer=initializer, name=name, dtype=tf.float32)
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
        W = create_variable_init('W', initializer=W_init)

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
            in_act = audio_batch + lc_batch  # add local condition
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
        self.mel_dim = hparams.num_mels
        self.lc_dim = lc_dim
        self.n_flows = n_flows
        self.n_group = n_group
        self.n_early_every = n_early_every
        self.n_early_size = n_early_size
        self.n_remaining_channels = n_group

        if hparams.lc_encode:
            self.lc_dim = hparams.lc_encode_size * 2

        if hparams.transposed_upsampling:
            self.lc_dim = hparams.transposed_conv_channels

    def create_lc_blstm_network(self, local_condition_batch):
        lstm_size = hparams.lc_encode_size
        lstm_layers = hparams.lc_encode_layers

        with tf.variable_scope("lc_blstm_embedding"):
            for layer_index in range(lstm_layers):
                with tf.variable_scope('layer_{}'.format(layer_index)):
                    fw_cell = tf.contrib.rnn.LSTMCell(lstm_size)
                    bw_cell = tf.contrib.rnn.LSTMCell(lstm_size)

                    outputs, states = tf.nn.bidirectional_dynamic_rnn(fw_cell,
                                                                      bw_cell,
                                                                      local_condition_batch,
                                                                      dtype=tf.float32)
                    local_condition_batch = tf.concat(outputs, axis=2)

        return local_condition_batch  # B*T*(lstm_channel*2)

    def create_transposed_conv1d(self, lc_batch, input_lc_dim=80):
        with tf.variable_scope('transpoed_conv'):
            # transposed conv layer 1
            lc_shape = tf.shape(lc_batch)
            batch_size, lc_length, lc_dim = lc_shape[0], lc_shape[1], lc_shape[2]
            filter1 = create_variable('layer1',
                                      [hparams.transposed_conv_layer1_filter_width, hparams.transposed_conv_channels,
                                       input_lc_dim])
            stride1 = hparams.transposed_conv_layer1_stride
            output_shape = [batch_size, lc_length * stride1, hparams.transposed_conv_channels]
            lc_batch = tf.contrib.nn.conv1d_transpose(lc_batch, filter1, output_shape, stride=stride1)
            # tf.nn.conv1d_transpose()
            lc_batch = tf.nn.relu(lc_batch)

            # transposed conv layer 2
            lc_shape = tf.shape(lc_batch)
            batch_size, lc_length, lc_dim = lc_shape[0], lc_shape[1], lc_shape[2]
            filter2 = create_variable('layer2',
                                      [hparams.transposed_conv_layer2_filter_width, hparams.transposed_conv_channels,
                                       hparams.transposed_conv_channels])
            stride2 = hparams.transposed_conv_layer2_stride
            output_shape = [batch_size, lc_length * stride2, hparams.transposed_conv_channels]
            lc_batch = tf.contrib.nn.conv1d_transpose(lc_batch, filter2, output_shape, stride=stride2)
            lc_batch = tf.nn.relu(lc_batch)

            return lc_batch

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

            if hparams.lc_encode:
                # local condition bi-directional encoding
                lc_batch = self.create_lc_blstm_network(lc_batch)

            if hparams.transposed_upsampling:
                # upsampling by transposed conv
                input_lc_dim = self.mel_dim
                if hparams.lc_encode:
                    input_lc_dim = hparams.lc_encode_size * 2

                lc_batch = self.create_transposed_conv1d(lc_batch, input_lc_dim)
            elif hparams.lc_encode and hparams.transposed_upsampling is False:
                # up-sampling in tf code by directly copy
                lc_batch = tf.tile(lc_batch, [1, 1, hparams.upsampling_rate])
                lc_batch = tf.reshape(lc_batch, [batch, -1, self.lc_dim])

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
            batch = tf.shape(lc_batch)[0]
            # compute the remaining channels
            remaining_channels = self.n_group
            for k in range(0, self.n_flows):
                if k % self.n_early_every == 0 and k > 0:
                    remaining_channels = remaining_channels - self.n_early_size

            if hparams.lc_encode:
                # local condition bi-directional encoding
                lc_batch = self.create_lc_blstm_network(lc_batch)

            if hparams.transposed_upsampling:
                # upsampling by transposed conv
                input_lc_dim = self.mel_dim
                if hparams.lc_encode:
                    input_lc_dim = hparams.lc_encode_size * 2

                lc_batch = self.create_transposed_conv1d(lc_batch, input_lc_dim)
            elif hparams.lc_encode and hparams.transposed_upsampling is False:
                # up-sampling in tf code by directly copy
                lc_batch = tf.tile(lc_batch, [1, 1, hparams.upsampling_rate])
                lc_batch = tf.reshape(lc_batch, [batch, -1, self.lc_dim])

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
