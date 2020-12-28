from tensorflow.keras.layers import Conv2D, MaxPool2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, LeakyReLU, Lambda
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import add, concatenate
from tensorflow.keras.layers import Layer
import tensorflow as tf




class YoloLayer(Layer):
    def __init__(self, anchors, max_grid, batch_size, warmup_batches, ignore_thresh, 
                    grid_scale, obj_scale, noobj_scale, xywh_scale, class_scale, 
                    **kwargs):
        # make the model settings persistent
        self.ignore_thresh  = ignore_thresh
        self.warmup_batches = warmup_batches
        self.anchors        = tf.constant(anchors, dtype='float', shape=[1,1,1,3,2])
        self.grid_scale     = grid_scale
        self.obj_scale      = obj_scale
        self.noobj_scale    = noobj_scale
        self.xywh_scale     = xywh_scale
        self.class_scale    = class_scale        

        # make a persistent mesh grid
        max_grid_h, max_grid_w = max_grid

        cell_x = tf.cast(tf.reshape(tf.tile(tf.range(max_grid_w), [max_grid_h]), (1, max_grid_h, max_grid_w, 1, 1)), dtype=tf.float32)
        cell_y = tf.transpose(cell_x, (0,2,1,3,4))
        self.cell_grid = tf.tile(tf.concat([cell_x,cell_y],-1), [batch_size, 1, 1, 3, 1])

        super(YoloLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(YoloLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        input_image, y_pred, y_true, true_boxes = x

        # adjust the shape of the y_predict [batch, grid_h, grid_w, 3, 4+1+nb_class]
        y_pred = tf.reshape(y_pred, tf.concat([tf.shape(y_pred)[:3], tf.constant([3, -1])], axis=0))
        
        # initialize the masks
        object_mask     = tf.expand_dims(y_true[..., 4], 4)

        # the variable to keep track of number of batches processed
        batch_seen = tf.Variable(0.)        

        # compute grid factor and net factor
        grid_h      = tf.shape(y_true)[1]
        grid_w      = tf.shape(y_true)[2]
        grid_factor = tf.reshape(tf.cast([grid_w, grid_h], tf.float32), [1,1,1,1,2])

        net_h       = tf.shape(input_image)[1]
        net_w       = tf.shape(input_image)[2]            
        net_factor  = tf.reshape(tf.cast([net_w, net_h], tf.float32), [1,1,1,1,2])
        
        """
        Adjust prediction
        """
        pred_box_xy    = (self.cell_grid[:,:grid_h,:grid_w,:,:] + tf.sigmoid(y_pred[..., :2]))  # sigma(t_xy) + c_xy
        pred_box_wh    = y_pred[..., 2:4]                                                       # t_wh
        pred_box_conf  = tf.expand_dims(tf.sigmoid(y_pred[..., 4]), 4)                          # adjust confidence
        pred_box_class = y_pred[..., 5:]                                                        # adjust class probabilities      

        """
        Adjust ground truth
        """
        true_box_xy    = y_true[..., 0:2] # (sigma(t_xy) + c_xy)
        true_box_wh    = y_true[..., 2:4] # t_wh
        true_box_conf  = tf.expand_dims(y_true[..., 4], 4)
        true_box_class = tf.argmax(y_true[..., 5:], -1)         

        """
        Compare each predicted box to all true boxes
        """        
        # initially, drag all objectness of all boxes to 0
        conf_delta  = pred_box_conf - 0 

        # then, ignore the boxes which have good overlap with some true box
        true_xy = true_boxes[..., 0:2] / grid_factor
        true_wh = true_boxes[..., 2:4] / net_factor
        
        true_wh_half = true_wh / 2.
        true_mins    = true_xy - true_wh_half
        true_maxes   = true_xy + true_wh_half
        
        pred_xy = tf.expand_dims(pred_box_xy / grid_factor, 4)
        pred_wh = tf.expand_dims(tf.exp(pred_box_wh) * self.anchors / net_factor, 4)
        
        pred_wh_half = pred_wh / 2.
        pred_mins    = pred_xy - pred_wh_half
        pred_maxes   = pred_xy + pred_wh_half    

        intersect_mins  = tf.maximum(pred_mins,  true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)

        intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
        
        true_areas = true_wh[..., 0] * true_wh[..., 1]
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores  = tf.truediv(intersect_areas, union_areas)

        best_ious   = tf.reduce_max(iou_scores, axis=4)        
        conf_delta *= tf.expand_dims(tf.cast((best_ious < self.ignore_thresh), dtype=tf.float32), 4)

        """
        Compute some online statistics
        """            
        true_xy = true_box_xy / grid_factor
        true_wh = tf.exp(true_box_wh) * self.anchors / net_factor

        true_wh_half = true_wh / 2.
        true_mins    = true_xy - true_wh_half
        true_maxes   = true_xy + true_wh_half

        pred_xy = pred_box_xy / grid_factor
        pred_wh = tf.exp(pred_box_wh) * self.anchors / net_factor 
        
        pred_wh_half = pred_wh / 2.
        pred_mins    = pred_xy - pred_wh_half
        pred_maxes   = pred_xy + pred_wh_half      

        intersect_mins  = tf.maximum(pred_mins,  true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
        
        true_areas = true_wh[..., 0] * true_wh[..., 1]
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores  = tf.truediv(intersect_areas, union_areas)
        iou_scores  = object_mask * tf.expand_dims(iou_scores, 4)
        
        count       = tf.reduce_sum(object_mask)
        count_noobj = tf.reduce_sum(1 - object_mask)
        detect_mask = tf.cast((pred_box_conf*object_mask >= 0.5), dtype=tf.float32)
        class_mask  = tf.expand_dims(tf.cast(tf.equal(tf.argmax(pred_box_class, -1), true_box_class), dtype=tf.float32), 4)
        recall50    = tf.reduce_sum(tf.cast((iou_scores >= 0.5), dtype=tf.float32) * detect_mask  * class_mask) / (count + 1e-3)
        recall75    = tf.reduce_sum(tf.cast((iou_scores >= 0.75), dtype=tf.float32) * detect_mask  * class_mask) / (count + 1e-3)    
        avg_iou     = tf.reduce_sum(iou_scores) / (count + 1e-3)
        avg_obj     = tf.reduce_sum(pred_box_conf  * object_mask)  / (count + 1e-3)
        avg_noobj   = tf.reduce_sum(pred_box_conf  * (1-object_mask))  / (count_noobj + 1e-3)
        avg_cat     = tf.reduce_sum(object_mask * class_mask) / (count + 1e-3) 

        """
        Warm-up training
        """
        batch_seen = tf.compat.v1.assign_add(batch_seen, 1.)
        
        true_box_xy, true_box_wh, xywh_mask = tf.cond(tf.less(batch_seen, self.warmup_batches+1), 
                              lambda: [true_box_xy + (0.5 + self.cell_grid[:,:grid_h,:grid_w,:,:]) * (1-object_mask), 
                                       true_box_wh + tf.zeros_like(true_box_wh) * (1-object_mask), 
                                       tf.ones_like(object_mask)],
                              lambda: [true_box_xy, 
                                       true_box_wh,
                                       object_mask])

        """
        Compare each true box to all anchor boxes
        """      
        wh_scale = tf.exp(true_box_wh) * self.anchors / net_factor
        wh_scale = tf.expand_dims(2 - wh_scale[..., 0] * wh_scale[..., 1], axis=4) # the smaller the box, the bigger the scale

        xy_delta    = xywh_mask   * (pred_box_xy-true_box_xy) * wh_scale * self.xywh_scale
        wh_delta    = xywh_mask   * (pred_box_wh-true_box_wh) * wh_scale * self.xywh_scale
        conf_delta  = object_mask * (pred_box_conf-true_box_conf) * self.obj_scale + (1-object_mask) * conf_delta * self.noobj_scale
        class_delta = object_mask * \
                      tf.expand_dims(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class), 4) * \
                      self.class_scale

        loss_xy    = tf.reduce_sum(tf.square(xy_delta),       list(range(1,5)))
        loss_wh    = tf.reduce_sum(tf.square(wh_delta),       list(range(1,5)))
        loss_conf  = tf.reduce_sum(tf.square(conf_delta),     list(range(1,5)))
        loss_class = tf.reduce_sum(class_delta,               list(range(1,5)))

        loss = loss_xy + loss_wh + loss_conf + loss_class


        return loss*self.grid_scale

    def compute_output_shape(self, input_shape):
        return [(None, 1)]



def dummy_loss(y_true, y_pred):
    return tf.sqrt(tf.reduce_sum(y_pred))

def NetworkConv2D_BN_Leaky(input, channels, kernel_size, kernel_regularizer = l2(5e-4), strides=(1,1), padding="same", use_bias=False):

    network = Conv2D( filters=channels, kernel_size=kernel_size, strides=strides, padding=padding, kernel_regularizer=kernel_regularizer, use_bias=use_bias)(input)
    network = BatchNormalization()(network)
    network = LeakyReLU(alpha=0.1)(network)
    return network

def residual_block(input, channels, num_blocks):
    network = ZeroPadding2D(((1,0), (1,0)))(input)
    network = NetworkConv2D_BN_Leaky(input=network,channels=channels, kernel_size=(3,3), strides=(2,2), padding="valid")

    for blocks in range(num_blocks):
        network_1 = NetworkConv2D_BN_Leaky(input=network, channels= channels // 2, kernel_size=(1,1))
        network_1 = NetworkConv2D_BN_Leaky(input=network_1,channels= channels, kernel_size=(3,3))

        network = Add()([network, network_1])
    return network

def darknet(input):
    network = NetworkConv2D_BN_Leaky(input=input, channels=32, kernel_size=(3,3))
    network = residual_block(input=network, channels=64, num_blocks=1)
    network = residual_block(input=network, channels=128, num_blocks=2)
    network = residual_block(input=network, channels=256, num_blocks=8)
    network = residual_block(input=network, channels=512, num_blocks=8)
    network = residual_block(input=network, channels=1024, num_blocks=4)


    return network

def last_layers(input, channels_in, channels_out, layer_name=""):



    network = NetworkConv2D_BN_Leaky( input=input, channels=channels_in, kernel_size=(1,1))
    network = NetworkConv2D_BN_Leaky(input=network, channels= (channels_in * 2) , kernel_size=(3, 3))
    network = NetworkConv2D_BN_Leaky(input=network, channels=channels_in, kernel_size=(1, 1))
    network = NetworkConv2D_BN_Leaky(input=network, channels=(channels_in * 2), kernel_size=(3, 3))
    network = NetworkConv2D_BN_Leaky(input=network, channels=channels_in, kernel_size=(1, 1))

    network_1 = NetworkConv2D_BN_Leaky(input=network, channels=(channels_in * 2), kernel_size=(3, 3))
    network_1 = Conv2D(filters=channels_out, kernel_size=(1,1), name=layer_name)(network_1)

    return  network, network_1

def yolov3_base(input, num_anchors, num_classes):
    
    darknet_network = Model(input, darknet(input))

    network, network_1 = last_layers(darknet_network.output, 512, num_anchors * (num_classes + 5), layer_name="last1")

    network = NetworkConv2D_BN_Leaky( input=network, channels=256, kernel_size=(1,1))
    network = UpSampling2D(2)(network)
    network = Concatenate()([network, darknet_network.layers[152].output])

    network, network_2 = last_layers(network,  256,  num_anchors * (num_classes + 5), layer_name="last2")

    network = NetworkConv2D_BN_Leaky(input=network, channels=128, kernel_size=(1, 1))
    network = UpSampling2D(2)(network)
    network = Concatenate()([network, darknet_network.layers[92].output])

    network, network_3 = last_layers(network, 128, num_anchors * (num_classes + 5), layer_name="last3")

    return input, network_1, network_2, network_3

def yolov3_main(input, num_anchors, num_classes):

    input, network_1, network_2, network_3 = yolov3_base(input, num_anchors, num_classes)

    return Model(input, [network_1, network_2, network_3])


def yolov3_train(num_classes,
                anchors,
                max_box_per_image, 
                max_grid, 
                batch_size, 
                warmup_batches,
                ignore_thresh,
                grid_scales,
                obj_scale,
                noobj_scale,
                xywh_scale,
                class_scale):

    input_image = Input(shape=(None, None, 3)) # net_h, net_w, 3
    true_boxes  = Input(shape=(1, 1, 1, max_box_per_image, 4))
    true_yolo_1 = Input(shape=(None, None, len(anchors)//6, 4+1+num_classes)) # grid_h, grid_w, nb_anchor, 5+nb_class
    true_yolo_2 = Input(shape=(None, None, len(anchors)//6, 4+1+num_classes)) # grid_h, grid_w, nb_anchor, 5+nb_class
    true_yolo_3 = Input(shape=(None, None, len(anchors)//6, 4+1+num_classes)) # grid_h, grid_w, nb_anchor, 5+nb_class
    
    
    
    _ , network_1, network_2, network_3 = yolov3_base(input_image, len(anchors)//6, num_classes)
    
    loss_yolo_1 = YoloLayer(anchors[12:], 
                            [1*num for num in max_grid], 
                            batch_size, 
                            warmup_batches, 
                            ignore_thresh, 
                            grid_scales[0],
                            obj_scale,
                            noobj_scale,
                            xywh_scale,
                            class_scale)([input_image, network_1, true_yolo_1, true_boxes])

    loss_yolo_2 = YoloLayer(anchors[6:12], 
                            [2*num for num in max_grid], 
                            batch_size, 
                            warmup_batches, 
                            ignore_thresh, 
                            grid_scales[1],
                            obj_scale,
                            noobj_scale,
                            xywh_scale,
                            class_scale)([input_image, network_2, true_yolo_2, true_boxes])

    loss_yolo_3 = YoloLayer(anchors[:6], 
                            [4*num for num in max_grid], 
                            batch_size, 
                            warmup_batches, 
                            ignore_thresh, 
                            grid_scales[2],
                            obj_scale,
                            noobj_scale,
                            xywh_scale,
                            class_scale)([input_image, network_3, true_yolo_3, true_boxes]) 

    train_model = Model([input_image, true_boxes, true_yolo_1, true_yolo_2, true_yolo_3], [loss_yolo_1, loss_yolo_2, loss_yolo_3])
    infer_model = Model(input_image, [network_1, network_2, network_3])

    return [train_model, infer_model]


def tiny_yolov3_main(input, num_anchors, num_classes):

    network_1 = NetworkConv2D_BN_Leaky(input=input, channels=16, kernel_size=(3,3) )
    network_1 = MaxPool2D(pool_size=(2,2), strides=(2,2), padding="same")(network_1)
    network_1 = NetworkConv2D_BN_Leaky(input=network_1, channels=32, kernel_size=(3, 3))
    network_1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")(network_1)
    network_1 = NetworkConv2D_BN_Leaky(input=network_1, channels=64, kernel_size=(3, 3))
    network_1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")(network_1)
    network_1 = NetworkConv2D_BN_Leaky(input=network_1, channels=128, kernel_size=(3, 3))
    network_1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")(network_1)
    network_1 = NetworkConv2D_BN_Leaky(input=network_1, channels=256, kernel_size=(3, 3))

    network_2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")(network_1)
    network_2 = NetworkConv2D_BN_Leaky(input=network_2, channels=512, kernel_size=(3, 3))
    network_2 = MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding="same")(network_2)
    network_2 = NetworkConv2D_BN_Leaky(input=network_2, channels=1024, kernel_size=(3, 3))
    network_2 = NetworkConv2D_BN_Leaky(input=network_2, channels=256, kernel_size=(1, 1))

    network_3 = NetworkConv2D_BN_Leaky(input=network_2, channels=512, kernel_size=(3, 3))
    network_3 = Conv2D(num_anchors * (num_classes + 5),  kernel_size=(1,1))(network_3)

    network_2 = NetworkConv2D_BN_Leaky(input=network_2, channels=128, kernel_size=(1, 1))
    network_2 = UpSampling2D(2)(network_2)

    network_4 = Concatenate()([network_2, network_1])
    network_4 = NetworkConv2D_BN_Leaky(input=network_4, channels=256, kernel_size=(3, 3))
    network_4 = Conv2D(num_anchors * (num_classes + 5), kernel_size=(1,1))(network_4)

    return Model(input, [network_3, network_4])

def dummy_loss(y_true, y_pred):
    return tf.sqrt(tf.reduce_sum(y_pred))