import tensorflow as tf

py_all = all

def depth_to_space(input, scale, data_format=None):
    ''' Uses phase shift algorithm to convert channels/depth for spatial resolution '''
    data_format = 'NHWC'

    data_format = data_format.lower()
    out = tf.depth_to_space(input, scale, data_format=data_format)
    return out
