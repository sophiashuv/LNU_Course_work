from keras.models import Model
from keras.layers import Input, Conv2D, Activation, Add, Conv2DTranspose
from keras.applications.vgg16 import VGG16


def FCN(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS, classes=1, fcn8=False, fcn16=False):
    vgg = VGG16(include_top=False, weights='imagenet', input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))
    pool5 = vgg.get_layer('block5_pool').output
    pool4 = vgg.get_layer('block4_pool').output
    pool3 = vgg.get_layer('block3_pool').output

    conv_6 = Conv2D(1024, (7, 7), activation='relu', padding='same', name="conv_6")(pool5)
    conv_7 = Conv2D(1024, (1, 1), activation='relu', padding='same', name="conv_7")(conv_6)

    conv_8 = Conv2D(classes, (1, 1), activation='relu', padding='same', name="conv_8")(pool4)
    conv_9 = Conv2D(classes, (1, 1), activation='relu', padding='same', name="conv_9")(pool3)

    deconv_7 = Conv2DTranspose(classes, kernel_size=(2, 2), strides=(2, 2))(conv_7)
    add_1 = Add()([deconv_7, conv_8])
    deconv_8 = Conv2DTranspose(classes, kernel_size=(2, 2), strides=(2, 2))(add_1)
    add_2 = Add()([deconv_8, conv_9])
    deconv_9 = Conv2DTranspose(classes, kernel_size=(8, 8), strides=(8, 8))(add_2)

    if fcn8:
        output_layer = Activation('sigmoid')(deconv_9)
    elif fcn16:
        deconv_10 = Conv2DTranspose(classes, kernel_size=(16, 16), strides=(16, 16))(add_1)
        output_layer = Activation('sigmoid')(deconv_10)
    else:
        deconv_11 = Conv2DTranspose(classes, kernel_size=(32, 32), strides=(32, 32))(conv_7)
        output_layer = Activation('sigmoid')(deconv_11)

    model = Model(inputs=vgg.input, outputs=output_layer)
    return model