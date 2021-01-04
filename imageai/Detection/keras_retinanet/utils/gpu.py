"""
Copyright 2017-2019 Fizyr (https://fizyr.com)

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

import tensorflow as tf


def setup_gpu(gpu_id):
    try:
        visible_gpu_indices = [int(id) for id in gpu_id.split(',')]
        available_gpus = tf.config.list_physical_devices('GPU')
        visible_gpus = [gpu for idx, gpu in enumerate(available_gpus) if idx in visible_gpu_indices]

        if visible_gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs.
                for gpu in available_gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)

                # Use only the selcted gpu.
                tf.config.set_visible_devices(visible_gpus, 'GPU')
            except RuntimeError as e:
                # Visible devices must be set before GPUs have been initialized.
                print(e)

            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(available_gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        else:
            tf.config.set_visible_devices([], 'GPU')
    except ValueError:
        tf.config.set_visible_devices([], 'GPU')
