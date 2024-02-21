from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Lambda, Dense, Flatten,MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
import numpy.random as rng
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from tensorflow import keras

def W_init(shape,dtype=None):
    """Initialize weights as in paper"""
    values = rng.normal(loc=0,scale=1e-2,size=shape)
    return K.variable(values,dtype=dtype)


def b_init(shape,dtype = None):
    """Initialize bias as in paper"""
    values=rng.normal(loc=0.5,scale=1e-2,size=shape)
    return K.variable(values,dtype=dtype)

input_shape = (105, 105, 1)
inputLayer = keras.Input(input_shape)
convLayer1 = Conv2D(64,(10,10),activation='relu',input_shape=input_shape,kernel_initializer=W_init,kernel_regularizer=l2(2e-4))(inputLayer)
poolingLayer1 = MaxPooling2D()(convLayer1)
convLayer2 = Conv2D(128,(7,7),activation='relu',kernel_regularizer=l2(2e-4),kernel_initializer=W_init,bias_initializer=b_init)(poolingLayer1)
poolingLayer2 = MaxPooling2D()(convLayer2)
convLayer3 = Conv2D(128,(4,4),activation='relu',kernel_initializer=W_init,kernel_regularizer=l2(2e-4),bias_initializer=b_init)(poolingLayer2)
poolingLayer3 = MaxPooling2D()(convLayer3)
convLayer4 = Conv2D(256,(4,4),activation='relu',kernel_initializer=W_init,kernel_regularizer=l2(2e-4),bias_initializer=b_init)(poolingLayer3)
flatten = Flatten()(convLayer4)
output = Dense(4096,activation="sigmoid",kernel_regularizer=l2(1e-3),kernel_initializer=W_init,bias_initializer=b_init)(flatten)
convNet = keras.Model(inputs = inputLayer,outputs= output, name= 'siamese_model')


# siamese net
left_input = keras.Input(input_shape)
right_input = keras.Input(input_shape)
encoded_l = convNet(left_input)
encoded_r = convNet(right_input)
L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
L1_distance = L1_layer([encoded_l, encoded_r])
prediction = Dense(1,activation='sigmoid',bias_initializer=b_init)(L1_distance)
siameseNet = Model(inputs=[left_input,right_input],outputs=prediction)
optimizer = Adam(0.00006)
siameseNet.compile(loss="binary_crossentropy",optimizer=optimizer)
# siameseNet.count_params()





class Siamese_Loader:
    """For loading batches and testing tasks to a siamese net"""
    def __init__(self, path, data_subsets = ["train", "val"]):
        self.data = {}
        self.categories = {}
        self.info = {}
        
        for name in data_subsets:
            file_path = os.path.join(path, name + ".pickle")
            print("loading data from {}".format(file_path))
            with open(file_path,"rb") as f:
                (X,c) = pickle.load(f)
                self.data[name] = X
                self.categories[name] = c

    def get_batch(self,batch_size,s="train"):
        """Create batch of n pairs, half same class, half different class"""
        X=self.data[s]
        n_classes, n_examples, w, h = X.shape

        #randomly sample several classes to use in the batch
        categories = rng.choice(n_classes,size=(batch_size,),replace=False)
        #initialize 2 empty arrays for the input image batch
        pairs=[np.zeros((batch_size, h, w,1)) for i in range(2)]
        #initialize vector for the targets, and make one half of it '1's, so 2nd half of batch has same class
        targets=np.zeros((batch_size,))
        targets[batch_size//2:] = 1
        for i in range(batch_size):
            category = categories[i]
            idx_1 = rng.randint(0, n_examples)
            pairs[0][i,:,:,:] = X[category, idx_1].reshape(w, h, 1)
            idx_2 = rng.randint(0, n_examples)
            #pick images of same class for 1st half, different for 2nd
            if i >= batch_size // 2:
                category_2 = category  
            else: 
                #add a random number to the category modulo n classes to ensure 2nd image has
                # ..different category
                category_2 = (category + rng.randint(1,n_classes)) % n_classes
            pairs[1][i,:,:,:] = X[category_2,idx_2].reshape(w, h,1)
        return pairs, targets
    
    def generate(self, batch_size, s="train"):
        """a generator for batches, so model.fit_generator can be used. """
        while True:
            pairs, targets = self.get_batch(batch_size,s)
            yield (pairs, targets)    

    
    def test_oneshot(self,model,N,k,s="val",verbose=0):
        """Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks"""
        n_correct = 0
        if verbose:
            print("Evaluating model on {} random {} way one-shot learning tasks ...".format(k,N))
        for i in range(k):
            inputs, targets = self.make_oneshot_task(N,s)
            probs = model.predict(inputs)
            if np.argmax(probs) == np.argmax(targets):
                n_correct+=1
        percent_correct = (100.0*n_correct / k)
        if verbose:
            print("Got an average of {}% {} way one-shot learning accuracy".format(percent_correct,N))
        return percent_correct
    
    def train(self, model, epochs, verbosity):
        model.fit_generator(self.generate(batch_size),)
    
    