
from keras.applications import VGG16


conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150,150,3))


# 冻结上层
conv_base.trainable = True

set_trainable = False

for layer in conv_base.layers:
    if layer.name == "block5_conv1":
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

# 微调模型

