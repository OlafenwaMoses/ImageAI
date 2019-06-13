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

import tensorflow
import keras


def resize_images(*args, **kwargs):
    return tensorflow.image.resize_images(*args, **kwargs)


def non_max_suppression(*args, **kwargs):
    return tensorflow.image.non_max_suppression(*args, **kwargs)


def range(*args, **kwargs):
    return tensorflow.range(*args, **kwargs)


def scatter_nd(*args, **kwargs):
    return tensorflow.scatter_nd(*args, **kwargs)


def gather_nd(*args, **kwargs):
    return tensorflow.gather_nd(*args, **kwargs)


def meshgrid(*args, **kwargs):
    return tensorflow.meshgrid(*args, **kwargs)


def where(*args, **kwargs):
    return tensorflow.where(*args, **kwargs)
