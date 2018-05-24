import os.path

import click
import keras
import keras.preprocessing.image
import numpy
import pkg_resources
import sklearn.model_selection
import tensorflow

import keras_resnet.classifiers

_benchmarks = {
    "CIFAR-10": keras.datasets.cifar10,
    "CIFAR-100": keras.datasets.cifar100,
    "MNIST": keras.datasets.mnist
}


_names = {
    "ResNet-18": keras_resnet.classifiers.ResNet18,
    "ResNet-34": keras_resnet.classifiers.ResNet34,
    "ResNet-50": keras_resnet.classifiers.ResNet50,
    "ResNet-101": keras_resnet.classifiers.ResNet101,
    "ResNet-152": keras_resnet.classifiers.ResNet152,
    "ResNet-200": keras_resnet.classifiers.ResNet200
}


@click.command()
@click.option(
    "--benchmark",
    default="CIFAR-10",
    type=click.Choice(
        [
            "CIFAR-10",
            "CIFAR-100",
            "ImageNet",
            "MNIST"
        ]
    )
)
@click.option("--device", default=0)
@click.option(
    "--name",
    default="ResNet-50",
    type=click.Choice(
        [
            "ResNet-18",
            "ResNet-34",
            "ResNet-50",
            "ResNet-101",
            "ResNet-152",
            "ResNet-200"
        ]
    )
)
def __main__(benchmark, device, name):
    configuration = tensorflow.ConfigProto()

    configuration.gpu_options.allow_growth = True

    configuration.gpu_options.visible_device_list = str(device)

    session = tensorflow.Session(config=configuration)

    keras.backend.set_session(session)

    (training_x, training_y), _ = _benchmarks[benchmark].load_data()

    training_x = training_x.astype(numpy.float16)

    training_y = keras.utils.np_utils.to_categorical(training_y)

    training_x, validation_x, training_y, validation_y = sklearn.model_selection.train_test_split(training_x, training_y)

    generator = keras.preprocessing.image.ImageDataGenerator(
        horizontal_flip=True
    )

    generator.fit(training_x)

    generator = generator.flow(
        x=training_x,
        y=training_y,
        batch_size=256
    )

    validation_data = keras.preprocessing.image.ImageDataGenerator()

    validation_data.fit(validation_x)

    validation_data = validation_data.flow(
        x=validation_x,
        y=validation_y,
        batch_size=256
    )

    shape, classes = training_x.shape[1:], training_y.shape[-1]

    x = keras.layers.Input(shape)

    model = _names[name](x, classes)

    model.compile("adam", "categorical_crossentropy", ["accuracy"])

    pathname = os.path.join("data", "checkpoints", benchmark, "{}.hdf5".format(name))

    pathname = pkg_resources.resource_filename("keras_resnet", pathname)

    model_checkpoint = keras.callbacks.ModelCheckpoint(pathname)

    pathname = os.path.join("data", "logs", benchmark, "{}.csv".format(name))

    pathname = pkg_resources.resource_filename("keras_resnet", pathname)

    csv_logger = keras.callbacks.CSVLogger(pathname)

    callbacks = [
        csv_logger,
        model_checkpoint
    ]

    model.fit_generator(
        callbacks=callbacks,
        epochs=200,
        generator=generator,
        steps_per_epoch=training_x.shape[0] // 256,
        validation_data=validation_data,
        validation_steps=validation_x.shape[0] // 256
    )

if __name__ == "__main__":
    __main__()
