import torch
import torch.nn as nn
import torch.nn.functional as F

import keras
from keras import backend as K
from keras.models import Input, Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Conv2DTranspose, concatenate

K.set_image_data_format('channels_first')

# From https://github.com/jocicmarko/ultrasound-nerve-segmentation
def unet_keras(input_size=224):
    inputs = Input((1, input_size, input_size))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv_block1_32.conv')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv_block1_32.conv2')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_block32_64.conv')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_block32_64.conv2')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_block64_128.conv')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_block64_128.conv2')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv_block128_256.conv')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv_block128_256.conv2')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv_block256_512.conv')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv_block256_512.conv2')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='valid', name='up_block512_256.up')(conv5), conv4], axis=1)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same', name='up_block512_256.conv')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same', name='up_block512_256.conv2')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='valid', name='up_block256_128.up')(conv6), conv3], axis=1)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', name='up_block256_128.conv')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', name='up_block256_128.conv2')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='valid', name='up_block128_64.up')(conv7), conv2], axis=1)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same', name='up_block128_64.conv')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same', name='up_block128_64.conv2')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='valid', name='up_block64_32.up')(conv8), conv1], axis=1)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same', name='up_block64_32.conv')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same', name='up_block64_32.conv2')(conv9)

    conv10 = Conv2D(2, (1, 1), activation=None, name='last')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])
    model.compile(optimizer=keras.optimizers.SGD(), loss=keras.losses.categorical_crossentropy)

    return model


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu):
        super(UNetConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size, padding=1)
        self.activation = activation

    def forward(self, x):
        out = self.activation(self.conv(x))
        out = self.activation(self.conv2(out))
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu, space_dropout=False):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, 2, stride=2)
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size, padding=1)
        self.activation = activation

    def center_crop(self, layer, target_size):
        batch_size, n_channels, layer_width, layer_height = layer.size()
        xy1 = (layer_width - target_size) // 2
        return layer[:, :, xy1:(xy1 + target_size), xy1:(xy1 + target_size)]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.size()[2])
        out = torch.cat([up, crop1], 1)
        out = self.activation(self.conv(out))
        out = self.activation(self.conv2(out))

        return out


class UNetPytorch(nn.Module):
    def __init__(self):
        super(UNetPytorch, self).__init__()

        self.activation = F.relu
        
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)
        self.pool4 = nn.MaxPool2d(2)

        self.conv_block1_32 = UNetConvBlock(1, 32)
        self.conv_block32_64 = UNetConvBlock(32, 64)
        self.conv_block64_128 = UNetConvBlock(64, 128)
        self.conv_block128_256 = UNetConvBlock(128, 256)
        
        self.conv_block256_512 = UNetConvBlock(256, 512)
        self.up_block512_256 = UNetUpBlock(512, 256)
        
        self.up_block256_128 = UNetUpBlock(256, 128)
        self.up_block128_64 = UNetUpBlock(128, 64)
        self.up_block64_32 = UNetUpBlock(64, 32)

        self.last = nn.Conv2d(32, 2, 1)


    def forward(self, x):

        block1 = self.conv_block1_32(x)
        pool1 = self.pool1(block1)

        block2 = self.conv_block32_64(pool1)
        pool2 = self.pool2(block2)

        block3 = self.conv_block64_128(pool2)
        pool3 = self.pool3(block3)

        block4 = self.conv_block128_256(pool3)
        pool4 = self.pool4(block4)

        block5 = self.conv_block256_512(pool4)

        up1 = self.up_block512_256(block5, block4)
        up2 = self.up_block256_128(up1, block3)
        up3 = self.up_block128_64(up2, block2)
        up4 = self.up_block64_32(up3, block1)

        return self.last(up4)

if __name__ == "__main__":
    net = UNetPytorch()
