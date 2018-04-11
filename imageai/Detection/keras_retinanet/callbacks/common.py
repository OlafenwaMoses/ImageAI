import keras.callbacks


class RedirectModel(keras.callbacks.Callback):
    """Callback which wraps another callback, but executed on a different model.
    # Arguments
        callback: callback to wrap.
        model: model to use when executing callbacks.
    # Example
        ```python
        model = keras.models.load_model('model.h5')
        model_checkpoint = ModelCheckpoint(filepath='snapshot.h5')
        parallel_model = multi_gpu_model(model, gpus=2)
        parallel_model.fit(X_train, Y_train, callbacks=[RedirectModel(model_checkpoint, model)])
        ```
    """

    def __init__(self,
                 callback,
                 model):
        super(RedirectModel, self).__init__()

        self.callback = callback
        self.redirect_model = model

    def on_epoch_begin(self, epoch, logs=None):
        self.callback.on_epoch_begin(epoch, logs=logs)

    def on_epoch_end(self, epoch, logs=None):
        self.callback.on_epoch_end(epoch, logs=logs)

    def on_batch_begin(self, batch, logs=None):
        self.callback.on_batch_begin(batch, logs=logs)

    def on_batch_end(self, batch, logs=None):
        self.callback.on_batch_end(batch, logs=logs)

    def on_train_begin(self, logs=None):
        # overwrite the model with our custom model
        self.callback.set_model(self.redirect_model)

        self.callback.on_train_begin(logs=logs)

    def on_train_end(self, logs=None):
        self.callback.on_train_end(logs=logs)
