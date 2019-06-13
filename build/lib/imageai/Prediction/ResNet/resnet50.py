from tensorflow.python import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPool2D, AvgPool2D, GlobalMaxPool2D, GlobalAvgPool2D, BatchNormalization, add, Input
from tensorflow.python.keras.models import Model


def resnet_module(input, channel_depth, strided_pool=False ):
    residual_input = input
    stride = 1

    if(strided_pool):
        stride = 2
        residual_input = Conv2D(channel_depth, kernel_size=1, strides=stride, padding="same")(residual_input)
        residual_input = BatchNormalization()(residual_input)

    input = Conv2D(int(channel_depth/4), kernel_size=1, strides=stride, padding="same")(input)
    input = BatchNormalization()(input)
    input = Activation("relu")(input)

    input = Conv2D(int(channel_depth / 4), kernel_size=3, strides=1, padding="same")(input)
    input = BatchNormalization()(input)
    input = Activation("relu")(input)

    input = Conv2D(channel_depth, kernel_size=1, strides=1, padding="same")(input)
    input = BatchNormalization()(input)

    input = add([input, residual_input])
    input = Activation("relu")(input)

    return input



def resnet_first_block_first_module(input, channel_depth):
    residual_input = input
    stride = 1

    residual_input = Conv2D(channel_depth, kernel_size=1, strides=1, padding="same")(residual_input)
    residual_input = BatchNormalization()(residual_input)

    input = Conv2D(int(channel_depth/4), kernel_size=1, strides=stride, padding="same")(input)
    input = BatchNormalization()(input)
    input = Activation("relu")(input)

    input = Conv2D(int(channel_depth / 4), kernel_size=3, strides=stride, padding="same")(input)
    input = BatchNormalization()(input)
    input = Activation("relu")(input)

    input = Conv2D(channel_depth, kernel_size=1, strides=stride, padding="same")(input)
    input = BatchNormalization()(input)

    input = add([input, residual_input])
    input = Activation("relu")(input)

    return input


def resnet_block(input, channel_depth, num_layers, strided_pool_first = False ):
    for i in range(num_layers):
        pool = False
        if(i == 0 and strided_pool_first):
            pool = True
        input = resnet_module(input, channel_depth, strided_pool=pool)

    return input

def ResNet50(include_top=True, non_top_pooling=None, model_input=None, num_classes=1000, weights='imagenet', model_path="", initial_num_classes=None, transfer_with_full_training = True):
    layers = [3,4,6,3]
    channel_depths = [256, 512, 1024, 2048]

    input_object = model_input


    output = Conv2D(64, kernel_size=7, strides=2, padding="same")(input_object)
    output = BatchNormalization()(output)
    output = Activation("relu")(output)

    output = MaxPool2D(pool_size=(3,3), strides=(2,2))(output)
    output = resnet_first_block_first_module(output, channel_depths[0])


    for i in range(4):
        channel_depth = channel_depths[i]
        num_layers = layers[i]

        strided_pool_first = True
        if(i == 0):
            strided_pool_first = False
            num_layers = num_layers - 1
        output = resnet_block(output, channel_depth=channel_depth, num_layers=num_layers, strided_pool_first=strided_pool_first)


    if(include_top):
        output = GlobalAvgPool2D(name="global_avg_pooling")(output)


        if(initial_num_classes != None):
            output = Dense(initial_num_classes)(output)
        else:
            output = Dense(num_classes)(output)
        output = Activation("softmax")(output)
    else:
        if (non_top_pooling == "Average"):
            output = GlobalAvgPool2D()(output)
        elif (non_top_pooling == "Maximum"):
            output = GlobalMaxPool2D()(output)
        elif (non_top_pooling == None):
            pass

    model = Model(inputs=input_object, outputs=output)


    if(weights == "imagenet"):
        weights_path = model_path
        model.load_weights(weights_path)
        return model
    elif(weights == "trained"):
        weights_path = model_path
        model.load_weights(weights_path)
        return model
    elif(weights == "continued"):
        weights_path = model_path
        model.load_weights(weights_path)
        return model
    elif(weights == "transfer"):
        weights_path = model_path
        model.load_weights(weights_path)

        if(transfer_with_full_training == False):
            for eachlayer in model.layers:
                eachlayer.trainable = False
            print("Training with top layers of the Model")
        else:
            print("Training with all layers of the Model")


        output2 = model.layers[-3].output
        output2 = Dense(num_classes)(output2)
        output2 = Activation("softmax")(output2)

        new_model = Model(inputs=model.input, outputs = output2)

        return new_model
    elif(weights == "custom"):
        return model













