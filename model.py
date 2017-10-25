import keras
from keras import backend as K
from keras.models import Model, load_model, Sequential
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, \
    Lambda, BatchNormalization, Dropout, Activation, \
    UpSampling2D, GlobalMaxPooling1D, TimeDistributed, Dense, \
    Permute, RepeatVector, Dot, Multiply, Maximum, \
    CuDNNGRU, CuDNNLSTM, Bidirectional
from keras.layers.merge import Concatenate, Maximum
from keras.layers.advanced_activations import ELU
from keras.optimizers import RMSprop, Nadam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, \
    EarlyStopping, LearningRateScheduler, History
from keras.regularizers import l2
from keras.objectives import binary_crossentropy, categorical_crossentropy, \
    mean_squared_error

from keras.utils.vis_utils import model_to_dot


smooth = 1e-12

def dice_coef(y_true, y_pred, channels=None):
    intersection = K.sum(y_true * y_pred, axis=[0, 1, 2])
    union = K.sum(y_true + y_pred, axis=[0, 1, 2])
    dice = (intersection + smooth) / (union - intersection + smooth)
    if not channels is None:
        return K.mean(K.gather(dice, channels))
    else:
        return K.mean(dice)

    
def dice_coef_0(y_true, y_pred):
    return dice_coef(y_true, y_pred, channels=[0])


def dice_coef_1(y_true, y_pred):
    return dice_coef(y_true, y_pred, channels=[1])


def dice_coef_01(y_true, y_pred):
    return dice_coef(y_true, y_pred, channels=[0, 1])


def dice_coef_loss(y_true, y_pred, channels=None):
    return -dice_coef(y_true, y_pred, channels=channels)


def dice_ce_loss(y_true, y_pred):
    return -K.log(dice_coef(y_true, y_pred)) + categorical_crossentropy(y_true, y_pred)


def dice_ce_01_loss(y_true, y_pred):
    return categorical_crossentropy(y_true, y_pred) - 0.5 * K.log(dice_coef_01(y_true, y_pred))


def dice_01_loss(y_true, y_pred):
    return -dice_coef_01(y_true, y_pred)


def fake_loss(y_true, y_pred):
    return K.mean(y_true) - K.mean(y_pred)


def simple_attention(seq_size, input_size, name, features=1):
    model = Sequential()
    model.add(TimeDistributed(Dense(features, activation='sigmoid', name=name+'_dense'),
                              input_shape=(seq_size, input_size)))
    model.add(GlobalMaxPooling1D(name=name+'_pool'))
    return model


def colwise_rowwise_attention(src_shape, depth, in_tensor, name):
    rows_attention = TimeDistributed(simple_attention(src_shape[0],
                                                      depth,
                                                      name+'_rows_att'),
                                     name=name+'_rows_att_td_in')(in_tensor)
    rows_attention_matrix = TimeDistributed(RepeatVector(src_shape[1]),
                                            name=name+'_rows_att_td_out')(rows_attention)
    out_with_row_att = Multiply(name=name+'_rows_att_result')([in_tensor,
                                                               rows_attention_matrix])

    transpose = Permute((2, 1, 3),
                        name=name+'_cols_att_transp_in')(in_tensor)
    cols_attention = TimeDistributed(simple_attention(src_shape[1],
                                                      depth,
                                                      name+'_cols_att'),
                                     name=name+'_cols_att_td_in')(transpose)
    cols_attention_matrix_t = TimeDistributed(RepeatVector(src_shape[0]),
                                              name=name+'_cols_att_td_out')(cols_attention)
    cols_attention_matrix = Permute((2, 1, 3),
                                    name=name+'_cols_att_transp_out')(cols_attention_matrix_t)
    out_with_col_att = Multiply(name=name+'_cols_att_result')([in_tensor, cols_attention_matrix])
    
    out_with_full_att = Multiply(name=name+'_full_att')([out_with_col_att, out_with_row_att])
    return out_with_full_att


def conv_block(inp, kernels_n, name, dilations=[1], add_pool=True, upconv=False, add_upconv_input=None, out_att=False):
    if upconv or not add_upconv_input is None:
        up = UpSampling2D(name=name+'_up')(inp)
        if not add_upconv_input is None:
            inp = Concatenate(name=name+'_up_concat')([up, add_upconv_input])
        else:
            inp = up

    conv_outs = []
    for direction, k_shape in (('h', (3, 1)), ('v', (1, 3))):
        for dilation in dilations:
            conv_outs.append(Conv2D(kernels_n,
                                    k_shape,
                                    dilation_rate=dilation,
                                    padding='same',
                                    name=name+'_conv_{}_conv_dil{}'.format(direction, dilation))(inp))
    if len(conv_outs) > 1:
        conv = Concatenate(name=name+'_conv1_conv')(conv_outs)
    else:
        conv = conv_outs[0]
    conv = BatchNormalization(name=name+'_conv1_bn')(conv)
    conv = ELU(name=name+'_conv1_act')(conv)
#     conv = Activation('relu')(conv)

    
#    conv_outs = []
#    for dilation in dilations:
#        conv_outs.append(Conv2D(kernels_n,
#                                (1, 3),
#                                dilation_rate=dilation,
#                                padding='same',
#                                name=name+'_conv2_conv_dil{}'.format(dilation))(conv))
#    if len(conv_outs) > 1:
#        conv = Concatenate(name=name+'_conv2_conv')(conv_outs)
#    else:
#        conv = conv_outs[0]
#    conv = BatchNormalization(name=name+'_conv2_bn')(conv)
#    conv = ELU(name=name+'_conv2_act')(conv)
#     conv = Activation('relu')(conv)

    if out_att:
        conv = colwise_rowwise_attention(K.int_shape(conv)[1:-1], kernels_n, conv, name)

    if add_pool:
        pool = MaxPooling2D(pool_size=(2, 2), name=name+'_pool')(conv)
    else:
        pool = None
    return conv, pool


def get_unet(in_shape, out_channels):
    first_conv_n = 16
    inputs = Input(in_shape)

    conv1, pool1 = conv_block(inputs, first_conv_n, 'enc1', dilations=[1])
    conv2, pool2 = conv_block(pool1, first_conv_n * 2, 'enc2', dilations=[2])
#     conv3, pool3 = conv_block(pool2, first_conv_n * 4, 'enc1', dilations=[4])
#     conv4, pool4 = conv_block(pool3, first_conv_n * 8, 'enc1', dilations=[8])

    conv5, _ = conv_block(pool2, first_conv_n * 8, 'bottleneck', add_pool=False)

#     conv6, _ = conv_block(conv5, first_conv_n * 8, 'dec4', add_pool=False, add_upconv_input=conv4)
#    conv7, _ = conv_block(conv5, first_conv_n * 4, 'dec3', dilations=[1], add_pool=False, add_upconv_input=conv3)
    conv8, _ = conv_block(conv5, first_conv_n * 2, 'dec2', dilations=[2], add_pool=False, add_upconv_input=conv2)
    conv9, _ = conv_block(conv8, first_conv_n, 'dec1', dilations=[4], add_pool=False, add_upconv_input=conv1)

    conv10 = Conv2D(out_channels, (1, 1), padding='same', name='out_conv')(conv9)
    out = Activation('softmax', name='out_act')(conv10)

    model = Model(inputs=[inputs], outputs=[out])
    model.compile(optimizer=Nadam(), #RMSprop(1e-2),
                  loss=dice_ce_01_loss,
#                   loss=dice_coef_loss,
#                   loss=categorical_crossentropy,
                  metrics=[dice_coef_0, dice_coef_1, dice_coef_01, categorical_crossentropy])

    return model


def get_unet_row_col_info(in_shape, out_channels):
    first_conv_n = 12
    inputs = Input(in_shape)

    conv1, pool1 = conv_block(inputs, first_conv_n, 'enc1', dilations=[1, 2, 4])
    conv2, pool2 = conv_block(pool1, first_conv_n * 2, 'enc2', dilations=[1, 2, 4])
    conv3, pool3 = conv_block(pool2, first_conv_n * 4, 'enc3', dilations=[1, 2, 4])
    #conv4, pool4 = conv_block(pool3, first_conv_n * 8, 'enc4', dilations=[1, 2, 4])

    conv5, _ = conv_block(pool3, first_conv_n * 8, 'bottleneck', add_pool=False)

    #conv6, _ = conv_block(conv5, first_conv_n * 8, 'dec4', add_pool=False, add_upconv_input=conv4)
    conv7, _ = conv_block(conv5, first_conv_n * 4, 'dec3', dilations=[1, 2, 4], add_pool=False, add_upconv_input=conv3)
    conv8, _ = conv_block(conv7, first_conv_n * 2, 'dec2', dilations=[1, 2, 4], add_pool=False, add_upconv_input=conv2)
    conv9, _ = conv_block(conv8, first_conv_n, 'dec1', dilations=[1, 2, 4], add_pool=False, add_upconv_input=conv1)

    out_with_full_att = colwise_rowwise_attention(in_shape, K.int_shape(conv9)[-1], conv9, 'out_att')
    
    conv10 = Conv2D(out_channels, (1, 1), padding='same', name='out_conv')(out_with_full_att)
    out = Activation('softmax', name='out_act')(conv10)

    model = Model(inputs=[inputs], outputs=[out])
    model.compile(optimizer=Nadam(), #RMSprop(1e-2),
                  loss=dice_ce_01_loss,
#                   loss=dice_coef_loss,
#                   loss=categorical_crossentropy,
                  metrics=[dice_coef_0, dice_coef_1, dice_coef_01, categorical_crossentropy])

    return model


def get_unet_row_col_info_autoenc(in_shape, out_channels):
    first_conv_n = 12
    inputs = Input(in_shape)

    conv1, pool1 = conv_block(inputs, first_conv_n, 'enc1', dilations=[1, 2])
    conv2, pool2 = conv_block(pool1, first_conv_n * 2, 'enc2', dilations=[1, 2])
    conv3, pool3 = conv_block(pool2, first_conv_n * 4, 'enc3', dilations=[1, 2])
    #conv4, pool4 = conv_block(pool3, first_conv_n * 8, 'enc4', dilations=[1, 2, 4])

    conv5, _ = conv_block(pool3, first_conv_n * 8, 'bottleneck', add_pool=False)

    #conv6, _ = conv_block(conv5, first_conv_n * 8, 'dec4', add_pool=False, add_upconv_input=conv4)
    conv7, _ = conv_block(conv5, first_conv_n * 4, 'dec3', dilations=[1, 2],
                          add_pool=False, add_upconv_input=conv3)
    conv8, _ = conv_block(conv7, first_conv_n * 2, 'dec2', dilations=[1, 2],
                          add_pool=False, add_upconv_input=conv2)
    conv9, _ = conv_block(conv8, first_conv_n, 'dec1', dilations=[1, 2],
                          add_pool=False, add_upconv_input=conv1)

    aux_conv7, _ = conv_block(conv5, first_conv_n * 4, 'aux_dec3', dilations=[1, 2],
                              add_pool=False, upconv=True)
    #aux_loss1 = Lambda(lambda t: mean_squared_error(conv3, aux_conv7)
    aux_conv8, _ = conv_block(aux_conv7, first_conv_n * 2, 'aux_dec2', dilations=[1, 2],
                              add_pool=False, upconv=True)
    #aux_loss2 = mean_squared_error(conv2, aux_conv8)
    aux_conv9, _ = conv_block(aux_conv8, first_conv_n, 'aux_dec1', dilations=[1, 2],
                              add_pool=False, upconv=True)
    #aux_loss3 = mean_squared_error(conv1, aux_conv9)
    aux_conv10 = Conv2D(K.int_shape(inputs)[-1], (1, 1), padding='same', name='aux_out_conv')(aux_conv9)
    #aux_loss4 = mean_squared_error(inputs, aux_conv10)

    out_with_full_att = colwise_rowwise_attention(in_shape, K.int_shape(conv9)[-1], conv9, 'out_att')

    conv10 = Conv2D(out_channels, (1, 1), padding='same', name='out_conv')(out_with_full_att)
    out = Activation('softmax', name='out_act')(conv10)

    #aux_out = 0.1 * K.mean(aux_loss1) + 0.1 * K.mean(aux_loss2) + 0.1 * K.mean(aux_loss3) + 0.5 * K.mean(aux_loss4)
    
    model = Model(inputs=[inputs], outputs=[out, aux_conv10])
    model.compile(optimizer=Nadam(), #RMSprop(1e-2),
                  loss=[dice_ce_01_loss, mean_squared_error],
                  metrics=[dice_coef_0, dice_coef_1, dice_coef_01, categorical_crossentropy])

    return model


def get_unet_each_row_col_info(in_shape, out_channels):
    first_conv_n = 12
    inputs = Input(in_shape)

    conv1, pool1 = conv_block(inputs, first_conv_n, 'enc1', dilation=1, out_att=True)
    conv2, pool2 = conv_block(pool1, first_conv_n * 2, 'enc2', dilation=2, out_att=True)
    conv3, pool3 = conv_block(pool2, first_conv_n * 4, 'enc3', dilation=4, out_att=True)
    conv4, pool4 = conv_block(pool3, first_conv_n * 8, 'enc4', dilation=8, out_att=True)

    conv5, _ = conv_block(pool4, first_conv_n * 16, 'bottleneck', add_pool=False)

    conv6, _ = conv_block(conv5, first_conv_n * 8, 'dec4', add_pool=False,
                          add_upconv_input=conv4, out_att=True)
    conv7, _ = conv_block(conv6, first_conv_n * 4, 'dec3', dilation=1, add_pool=False,
                          add_upconv_input=conv3, out_att=True)
    conv8, _ = conv_block(conv7, first_conv_n * 2, 'dec2', dilation=2, add_pool=False,
                          add_upconv_input=conv2, out_att=True)
    conv9, _ = conv_block(conv8, first_conv_n, 'dec1', dilation=4, add_pool=False,
                          add_upconv_input=conv1, out_att=True)

    conv10 = Conv2D(out_channels, (1, 1), padding='same', name='out_conv')(conv9)
    out = Activation('softmax', name='out_act')(conv10)

    model = Model(inputs=[inputs], outputs=[out])
    model.compile(optimizer=Nadam(), #RMSprop(1e-2),
                  loss=dice_ce_01_loss,
#                   loss=dice_coef_loss,
#                   loss=categorical_crossentropy,
                  metrics=[dice_coef_0, dice_coef_1, dice_coef_01, categorical_crossentropy])

    return model


def get_unet_lstm(in_shape, out_channels):
    first_conv_n = 8
    inputs = Input(in_shape)

    conv1, pool1 = conv_block(inputs, first_conv_n, 'enc1', dilation=1)
    conv2, pool2 = conv_block(pool1, first_conv_n * 2, 'enc2', dilation=2)
    conv3, pool3 = conv_block(pool2, first_conv_n * 4, 'enc3', dilation=4)
    conv4, pool4 = conv_block(pool3, first_conv_n * 8, 'enc4', dilation=8)

    conv5, _ = conv_block(pool4, first_conv_n * 16, 'bottleneck', add_pool=False)

    conv6, _ = conv_block(conv5, first_conv_n * 8, 'dec4', dilation=1, add_pool=False, add_upconv_input=conv4)
    conv7, _ = conv_block(conv6, first_conv_n * 4, 'dec3', dilation=2, add_pool=False, add_upconv_input=conv3)
    conv8, _ = conv_block(conv7, first_conv_n * 2, 'dec2', dilation=3, add_pool=False, add_upconv_input=conv2)
    conv9, _ = conv_block(conv8, first_conv_n, 'dec1', dilation=4, add_pool=False, add_upconv_input=conv1)

    rows_repr_forward = TimeDistributed(CuDNNLSTM(64, return_sequences=True, name='rows_rnn_forward'),
                                        name='rows_td_forward')(conv9)
    rows_backward = Lambda(lambda t: K.reverse(t, 1), name='rows_reverse_in')(conv9)
    rows_repr_backward_rev = TimeDistributed(CuDNNLSTM(64, return_sequences=True, name='rows_rnn_backward'),
                                             name='rows_td_backward')(rows_backward)
    rows_repr_backward = Lambda(lambda t: K.reverse(t, 1), name='rows_reverse_out')(rows_repr_backward_rev)
    
    transpose = Permute((2, 1, 3), name='cols_transp_in')(conv9)
    cols_repr_forward_t = TimeDistributed(CuDNNLSTM(64, return_sequences=True, name='cols_rnn_forward'),
                                          name='cols_td_forward')(transpose)
    cols_repr_forward = Permute((2, 1, 3), name='cols_transp_out')(cols_repr_forward_t)

    cols_backward = Lambda(lambda t: K.reverse(t, 1), name='cols_reverse_in')(transpose)
    cols_repr_backward_rev_t = TimeDistributed(CuDNNLSTM(64, return_sequences=True, name='cols_rnn_backward'),
                                               name='cols_td_backward')(cols_backward)
    cols_repr_backward_t = Lambda(lambda t: K.reverse(t, 1), name='cols_reverse_out')(cols_repr_backward_rev_t)
    cols_repr_backward = Permute((2, 1, 3), name='cols_backward_transp_out')(cols_repr_backward_t)
    
    full_features = Concatenate(-1)([conv9,
                                     cols_repr_forward, cols_repr_backward,
                                     rows_repr_forward, rows_repr_backward])
    
    conv10 = Conv2D(out_channels, (1, 1), padding='same')(full_features)
    out = Activation('softmax')(conv10)

    model = Model(inputs=[inputs], outputs=[out])
    model.compile(optimizer=Nadam(), #RMSprop(1e-2),
                  loss=dice_ce_01_loss,
#                   loss=dice_coef_loss,
#                   loss=categorical_crossentropy,
                  metrics=[dice_coef_0, dice_coef_1, dice_coef_01, categorical_crossentropy])

    return model
