from tensorflow.python.keras.layers import Input, Conv2D, MaxPool2D, Activation, concatenate, Dropout
from tensorflow.python.keras.layers import GlobalAvgPool2D, GlobalMaxPool2D
from tensorflow.python.keras.models import Model


def squeezenet_fire_module(input, input_channel_small=16, input_channel_large=64):

    channel_axis = 3

    input = Conv2D(input_channel_small, (1,1), padding="valid" )(input)
    input = Activation("relu")(input)

    input_branch_1 = Conv2D(input_channel_large, (1,1), padding="valid" )(input)
    input_branch_1 = Activation("relu")(input_branch_1)

    input_branch_2 = Conv2D(input_channel_large, (3, 3), padding="same")(input)
    input_branch_2 = Activation("relu")(input_branch_2)

    input = concatenate([input_branch_1, input_branch_2], axis=channel_axis)

    return input

def SqueezeNet(include_top = True, weights="imagenet", model_input=None, non_top_pooling=None,
               num_classes=1000, model_path = ""):

    if(weights == "imagenet" and num_classes != 1000):
        raise ValueError("You must parse in SqueezeNet model trained on the 1000 class ImageNet")




    image_input = model_input


    network = Conv2D(64, (3,3), strides=(2,2), padding="valid")(image_input)
    network = Activation("relu")(network)
    network = MaxPool2D( pool_size=(3,3) , strides=(2,2))(network)

    network = squeezenet_fire_module(input=network, input_channel_small=16, input_channel_large=64)
    network = squeezenet_fire_module(input=network, input_channel_small=16, input_channel_large=64)
    network = MaxPool2D(pool_size=(3,3), strides=(2,2))(network)

    network = squeezenet_fire_module(input=network, input_channel_small=32, input_channel_large=128)
    network = squeezenet_fire_module(input=network, input_channel_small=32, input_channel_large=128)
    network = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(network)

    network = squeezenet_fire_module(input=network, input_channel_small=48, input_channel_large=192)
    network = squeezenet_fire_module(input=network, input_channel_small=48, input_channel_large=192)
    network = squeezenet_fire_module(input=network, input_channel_small=64, input_channel_large=256)
    network = squeezenet_fire_module(input=network, input_channel_small=64, input_channel_large=256)

    if(include_top):
        network = Dropout(0.5)(network)

        network = Conv2D(num_classes, kernel_size=(1,1), padding="valid", name="last_conv")(network)
        network = Activation("relu")(network)

        network = GlobalAvgPool2D()(network)
        network = Activation("softmax")(network)

    else:
        if(non_top_pooling == "Average"):
            network = GlobalAvgPool2D()(network)
        elif(non_top_pooling == "Maximum"):
            network = GlobalMaxPool2D()(network)
        elif(non_top_pooling == None):
            pass

    input_image = image_input
    model = Model(inputs=input_image, outputs=network)

    if(weights =="imagenet"):
        weights_path = model_path
        model.load_weights(weights_path)
    elif(weights =="trained"):
        weights_path = model_path
        model.load_weights(weights_path)

    return model







