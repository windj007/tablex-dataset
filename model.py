import keras
from keras import backend as K
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, \
    Lambda, BatchNormalization, Dropout, Activation, \
    UpSampling2D
from keras.layers.merge import Concatenate, Maximum
from keras.layers.advanced_activations import ELU
from keras.optimizers import RMSprop, Nadam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, LearningRateScheduler, History
from keras.regularizers import l2
from keras.objectives import binary_crossentropy, categorical_crossentropy

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


def conv_block(inp, kernels_n, dilation=2, add_pool=True, add_upconv_input=None):
    if not add_upconv_input is None:
#         conv = ELU()(conv)
#         up = BatchNormalization()(inp)
#         up = Conv2DTranspose(kernels_n, (2, 2), strides=(2, 2), padding='same')(inp)
        up = UpSampling2D()(inp)
        inp = Concatenate()([up, add_upconv_input])

    conv = BatchNormalization()(inp)
    conv = Conv2D(kernels_n, (3, 3), dilation_rate=dilation, padding='same')(conv)
    conv = Dropout(0.1)(conv)
    conv = ELU()(conv)
#     conv = Activation('relu')(conv)

    conv = BatchNormalization()(conv)
    conv = Conv2D(kernels_n, (3, 3), dilation_rate=dilation, padding='same')(conv)
    conv = Dropout(0.1)(conv)
    conv = ELU()(conv)
#     conv = Activation('relu')(conv)

    if add_pool:
        pool = MaxPooling2D(pool_size=(2, 2))(conv)
    else:
        pool = None
    return conv, pool


def get_unet(in_shape, out_channels):
    first_conv_n = 16
    inputs = Input(in_shape)

    conv1, pool1 = conv_block(inputs, first_conv_n, dilation=1)
    conv2, pool2 = conv_block(pool1, first_conv_n * 2, dilation=2)
    conv3, pool3 = conv_block(pool2, first_conv_n * 4, dilation=4)
#     conv4, pool4 = conv_block(pool3, first_conv_n * 8)

    conv5, _ = conv_block(pool3, first_conv_n * 8, add_pool=False)

#     conv6, _ = conv_block(conv5, first_conv_n * 8, add_pool=False, add_upconv_input=conv4)
    conv7, _ = conv_block(conv5, first_conv_n * 4, dilation=1, add_pool=False, add_upconv_input=conv3)
    conv8, _ = conv_block(conv7, first_conv_n * 2, dilation=2, add_pool=False, add_upconv_input=conv2)
    conv9, _ = conv_block(conv8, first_conv_n, dilation=4, add_pool=False, add_upconv_input=conv1)

    conv10 = Conv2D(out_channels, (1, 1), padding='same')(conv9)
    out = Activation('softmax')(conv10)

    model = Model(inputs=[inputs], outputs=[out])
    model.compile(optimizer=Nadam(), #RMSprop(1e-2),
                  loss=dice_ce_01_loss,
#                   loss=dice_coef_loss,
#                   loss=categorical_crossentropy,
                  metrics=[dice_coef_0, dice_coef_1, dice_coef_01, categorical_crossentropy])

    return model