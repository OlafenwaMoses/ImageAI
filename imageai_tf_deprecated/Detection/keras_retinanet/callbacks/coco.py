"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from tensorflow import keras
from ..utils.coco_eval import evaluate_coco


class CocoEval(keras.callbacks.Callback):
    """ Performs COCO evaluation on each epoch.
    """
    def __init__(self, generator, tensorboard=None, threshold=0.05):
        """ CocoEval callback intializer.

        Args
            generator   : The generator used for creating validation data.
            tensorboard : If given, the results will be written to tensorboard.
            threshold   : The score threshold to use.
        """
        self.generator = generator
        self.threshold = threshold
        self.tensorboard = tensorboard

        super(CocoEval, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        coco_tag = ['AP @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]',
                    'AP @[ IoU=0.50      | area=   all | maxDets=100 ]',
                    'AP @[ IoU=0.75      | area=   all | maxDets=100 ]',
                    'AP @[ IoU=0.50:0.95 | area= small | maxDets=100 ]',
                    'AP @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]',
                    'AP @[ IoU=0.50:0.95 | area= large | maxDets=100 ]',
                    'AR @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]',
                    'AR @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]',
                    'AR @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]',
                    'AR @[ IoU=0.50:0.95 | area= small | maxDets=100 ]',
                    'AR @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]',
                    'AR @[ IoU=0.50:0.95 | area= large | maxDets=100 ]']
        coco_eval_stats = evaluate_coco(self.generator, self.model, self.threshold)

        if coco_eval_stats is not None:
            for index, result in enumerate(coco_eval_stats):
                logs[coco_tag[index]] = result

            if self.tensorboard:
                import tensorflow as tf
                writer = tf.summary.create_file_writer(self.tensorboard.log_dir)
                with writer.as_default():
                    for index, result in enumerate(coco_eval_stats):
                        tf.summary.scalar('{}. {}'.format(index + 1, coco_tag[index]), result, step=epoch)
                    writer.flush()
