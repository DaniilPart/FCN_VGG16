import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Add
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

input_shape = (160, 576, 3)
num_classes = 2

inputs = Input(shape=input_shape, name='input_image')

block1_out = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_out')(inputs)
block2_out = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_out')(block1_out)

layer3_out = Conv2D(256, (3,3), activation='relu', padding='same', name='vgg_layer3_out')(block2_out)
layer4_out = Conv2D(512, (3,3), padding='same', activation='relu', strides=(2,2), name='vgg_layer4_out')(layer3_out)
layer7_out = Conv2D(4096, (7,7), padding='same', activation='relu', strides=(2,2), name='vgg_layer7_out')(layer4_out)

conv_1x1_7 = Conv2D(num_classes, (1,1), padding='same', name='conv_1x1_7')(layer7_out)
conv7_2x = Conv2DTranspose(num_classes, (4,4), strides=(2,2), padding='same', name='conv7_2x')(conv_1x1_7)
conv_1x1_4 = Conv2D(num_classes, (1,1), padding='same', name='conv_1x1_4')(layer4_out)
skip_4_to_7 = Add(name='skip_4_to_7')([conv7_2x, conv_1x1_4])
upsample2x_skip_4_to_7 = Conv2DTranspose(num_classes, (4,4), strides=(2,2), padding='same', name='upsample2x_skip_4_to_7')(skip_4_to_7)
conv_1x1_3 = Conv2D(num_classes, (1,1), padding='same', name='conv_1x1_3')(layer3_out)
skip_3 = Add(name='skip_3')([upsample2x_skip_4_to_7, conv_1x1_3])
output = Conv2DTranspose(num_classes, (16,16), strides=(8,8), padding='same', name='output')(skip_3)

model = Model(inputs=inputs, outputs=output, name='FCN_VGG16')

plot_model(model, to_file='extended_architecture.png', show_shapes=True, show_layer_names=True)
print("pic in extended_architecture.png")

print("\nAll layers in model:")
for layer in model.layers:
    print(f"Layer: {layer.name}, Shape input: {layer.input_shape}, Shape output: {layer.output_shape}")