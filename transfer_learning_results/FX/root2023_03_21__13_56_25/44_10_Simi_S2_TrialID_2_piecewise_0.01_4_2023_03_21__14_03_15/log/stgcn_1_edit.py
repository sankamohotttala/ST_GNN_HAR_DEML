#embeddding is also given by this code for  the t-sne visualization
from graph.ntu_rgb_d_cwbg import Graph
import tensorflow as tf
import numpy as np

# REGULARIZER_ = tf.keras.regularizers.l2(l=0.001) #0.0001 is the default value
# # INITIALIZER = tf.keras.initializers.VarianceScaling(scale=2.,
# #                                                     mode="fan_in",
# #                                                     distribution="truncated_normal") #experiment with this
# INITIALIZER_ = tf.keras.initializers.VarianceScaling(scale=2.,
#                                                     mode="fan_out",
#                                                     distribution="truncated_normal") #default values
                            

class INITIALIZERS():
    def __init__(self, scale=2., mode="fan_out", distribution="truncated_normal",reg='l1',reg_value=0.001):
        self.scale = scale
        self.mode = mode
        self.distribution = distribution
        self.reg_val = reg_value
        if reg == 'l1':
            self.REGULARIZER = tf.keras.regularizers.l1(l=self.reg_val)
        elif reg == 'l2':
            self.REGULARIZER = tf.keras.regularizers.l2(l=self.reg_val)
        else:
            ValueError('Regularizer not supported')
        self.INITIALIZER = tf.keras.initializers.VarianceScaling(scale=self.scale,mode=self.mode,distribution=self.distribution)
        #later add for other distributions classes as initializers

class SGCN(tf.keras.Model):
    def __init__(self, initializers=None, filters=None, kernel_size=3):#probably this kernal size is related to the channel number may be change it to 2?
        super().__init__()
        self.initializer = initializers
        self.kernel_size = kernel_size
        self.conv = tf.keras.layers.Conv2D(filters*kernel_size,
                                           kernel_size=1,
                                           padding='same',
                                           kernel_initializer=self.initializer.INITIALIZER,
                                           data_format='channels_first',
                                           kernel_regularizer=self.initializer.REGULARIZER)

    # N, C, T, V
    def call(self, x, A, training):
        x = self.conv(x)

        N = tf.shape(x)[0]
        C = tf.shape(x)[1] #192 ?? where does it come from 64*3??
        T = tf.shape(x)[2]
        V = tf.shape(x)[3]

        x = tf.reshape(x, [N, self.kernel_size, C//self.kernel_size, T, V])
        x = tf.einsum('nkctv,kvw->nctw', x, A) # 16, 3, 64, 300, 25 --*--3, 25, 25
        return x, A
        

class STGCN(tf.keras.Model):
    def __init__(self, filters,initializers=None, kernel_size=[9, 3], stride=1, activation='relu',
                 residual=True, downsample=False):
        super().__init__()

        self.initializer = initializers
        self.sgcn = SGCN(initializers,filters, kernel_size=kernel_size[1])

        self.tgcn = tf.keras.Sequential()
        self.tgcn.add(tf.keras.layers.BatchNormalization(axis=1))
        self.tgcn.add(tf.keras.layers.Activation(activation))
        self.tgcn.add(tf.keras.layers.Conv2D(filters,
                                                kernel_size=[kernel_size[0], 1],
                                                strides=[stride, 1],
                                                padding='same',
                                                kernel_initializer=self.initializer.INITIALIZER,
                                                data_format='channels_first',
                                                kernel_regularizer=self.initializer.REGULARIZER))
        self.tgcn.add(tf.keras.layers.BatchNormalization(axis=1))#why use at the end of the model as well??

        self.act = tf.keras.layers.Activation(activation)

        if not residual:
            self.residual = lambda x, training=False: 0
        elif residual and stride == 1 and not downsample:
            self.residual = lambda x, training=False: x
        else:
            self.residual = tf.keras.Sequential()
            self.residual.add(tf.keras.layers.Conv2D(filters,
                                                        kernel_size=[1, 1],
                                                        strides=[stride, 1],
                                                        padding='same',
                                                        kernel_initializer=self.initializer.INITIALIZER,
                                                        data_format='channels_first',
                                                        kernel_regularizer=self.initializer.REGULARIZER))
            self.residual.add(tf.keras.layers.BatchNormalization(axis=1))

    def call(self, x, A, training):
        res = self.residual(x, training=training)#lambda function
        x, A = self.sgcn(x, A, training=training)
        x = self.tgcn(x, training=training)
        x += res
        x = self.act(x) #16,64,300,25
        return x, A


class Model(tf.keras.Model):
    def __init__(self, num_classes=60,layer_out=10,initializers=None):
    # def __init__(self, num_classes=60):
        super().__init__()

        self.initializers=initializers
        self.out_stgcn=layer_out

        graph = Graph()
        self.A = tf.Variable(graph.A,
                             dtype=tf.float32,
                             trainable=False,
                             name='adjacency_matrix')

        self.data_bn = tf.keras.layers.BatchNormalization(axis=1)

        self.STGCN_layers = []
        self.STGCN_layers.append(STGCN(64,initializers, residual=False ))
        self.STGCN_layers.append(STGCN(64,initializers))
        self.STGCN_layers.append(STGCN(64,initializers))
        self.STGCN_layers.append(STGCN(64,initializers))
        self.STGCN_layers.append(STGCN(128,initializers, stride=2, downsample=True))
        self.STGCN_layers.append(STGCN(128,initializers))
        self.STGCN_layers.append(STGCN(128,initializers))
        self.STGCN_layers.append(STGCN(256,initializers, stride=2, downsample=True))
        self.STGCN_layers.append(STGCN(256,initializers))
        self.STGCN_layers.append(STGCN(256,initializers))

        self.pool = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_first')

        self.logits = tf.keras.layers.Conv2D(num_classes,
                                             kernel_size=1,
                                             padding='same',
                                             kernel_initializer=self.initializers.INITIALIZER,
                                             data_format='channels_first',
                                             kernel_regularizer=self.initializers.REGULARIZER)

    def call(self, x, training):
        N = tf.shape(x)[0]
        C = tf.shape(x)[1]
        T = tf.shape(x)[2]
        V = tf.shape(x)[3]
        M = tf.shape(x)[4]

        x = tf.transpose(x, perm=[0, 4, 3, 1, 2])
        x = tf.reshape(x, [N * M, V * C, T])
        x = self.data_bn(x, training=training)#batch normalizarion
        x = tf.reshape(x, [N, M, V, C, T])
        x = tf.transpose(x, perm=[0, 1, 3, 4, 2])
        x = tf.reshape(x, [N * M, C, T, V])

        A = self.A
        indexLayer = 0
        for layer in self.STGCN_layers:
            indexLayer += 1
            x, A = layer(x, A, training=training) #check if A changes inside ie learns something
            if indexLayer == self.out_stgcn:  #get the output of GCN layer as a feature map ; later try to give 5 in model constructor; may not work with loading weights
                returnLayer = x
                
            
            # print(indexLayer) 
            
        # N*M,C,T,V
        x = self.pool(x)
        out=self.pool(returnLayer) #notice x and returnLayer are same

        b_ = tf.reshape(out, [N, M, -1, 1, 1])
        b__ = tf.reduce_mean(b_, axis=1) #this result in
        embed=b__ 

        x = tf.reshape(x, [N, M, -1, 1, 1])
        x = tf.reduce_mean(x, axis=1) #this result in 
        x = self.logits(x)

        #print(self.logits.weights[0])
        x = tf.reshape(x, [N, -1]) #shape = (batch size,number of classes)

        return x , embed , out


class Model2(tf.keras.Model):
    def __init__(self, num_classes=10,intializers=None):
        super().__init__()
        self.intializers=intializers

     
        self.dense1=tf.keras.layers.Dense(196, activation='relu',kernel_initializer=self.intializers.INITIALIZER,kernel_regularizer=self.intializers.REGULARIZER) #256
        self.dense2=tf.keras.layers.Dense(128, activation='relu',kernel_initializer=self.intializers.INITIALIZER,kernel_regularizer=self.intializers.REGULARIZER)
        self.dense3=tf.keras.layers.Dense(num_classes)#, activation='relu'


    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        #print(self.dense2.weights[0])
        x= self.dense3(x)
        return x


class Model2_original(tf.keras.Model):
    def __init__(self, num_classes=10,intializers=None):
        super().__init__()
        self.intializers=intializers
     
        # self.dense1=tf.keras.layers.Dense(196, activation='relu') #256
        self.dense2=tf.keras.layers.Dense(256, activation='relu',kernel_initializer=self.intializers.INITIALIZER,kernel_regularizer=self.intializers.REGULARIZER)
        self.dense3=tf.keras.layers.Dense(num_classes)#, activation='relu'
        
        
        


    def call(self, inputs):
        #x = self.dense1(inputs)
        x = self.dense2(inputs)
        #print(self.dense2.weights[0])
        x= self.dense3(x)
        return x
    
class Model2_new(tf.keras.Model):
    def __init__(self, num_classes=10):
        super().__init__()

     
        self.dense1=tf.keras.layers.Dense(256, activation='relu') # 512 originally
        self.drop1=tf.keras.layers.Dropout(.5) #.7
        self.dense2=tf.keras.layers.Dense(128, activation='relu')
        # self.dense_2=tf.keras.layers.Dense(64, activation='relu')
        self.dense3=tf.keras.layers.Dense(num_classes)#, activation='relu'

        


    def call(self, inputs,training=None):
        # x=tf.keras.activations.relu(inputs)
        x = self.dense1(inputs)
        if training:
            x=self.drop1(x)
        x = self.dense2(x)
        # x=self.dense_2(x)
        #print(self.dense2.weights[0])
        x= self.dense3(x)
        return x


# #here we try to improve the modelfor feature extraction approach
# from graph.ntu_rgb_d_cwbg import Graph
# import tensorflow as tf
# import numpy as np

# REGULARIZER = tf.keras.regularizers.l2(l=0.0001)
# INITIALIZER = tf.keras.initializers.VarianceScaling(scale=2.,
#                                                     mode="fan_out",
#                                                     distribution="truncated_normal")
                            


# class SGCN(tf.keras.Model):
#     def __init__(self, filters, kernel_size=3):#probably this kernal size is related to the channel number may be change it to 2?
#         super().__init__()
#         self.kernel_size = kernel_size
#         self.conv = tf.keras.layers.Conv2D(filters*kernel_size,
#                                            kernel_size=1,
#                                            padding='same',
#                                            kernel_initializer=INITIALIZER,
#                                            data_format='channels_first',
#                                            kernel_regularizer=REGULARIZER)

#     # N, C, T, V
#     def call(self, x, A, training):
#         x = self.conv(x)

#         N = tf.shape(x)[0]
#         C = tf.shape(x)[1] #192 ?? where does it come from 64*3??
#         T = tf.shape(x)[2]
#         V = tf.shape(x)[3]

#         x = tf.reshape(x, [N, self.kernel_size, C//self.kernel_size, T, V])
#         x = tf.einsum('nkctv,kvw->nctw', x, A)
#         return x, A



# class STGCN(tf.keras.Model):
#     def __init__(self, filters, kernel_size=[9, 3], stride=1, activation='relu',
#                  residual=True, downsample=False):
#         super().__init__()
#         self.sgcn = SGCN(filters, kernel_size=kernel_size[1])

#         self.tgcn = tf.keras.Sequential()
#         self.tgcn.add(tf.keras.layers.BatchNormalization(axis=1))
#         self.tgcn.add(tf.keras.layers.Activation(activation))
#         self.tgcn.add(tf.keras.layers.Conv2D(filters,
#                                                 kernel_size=[kernel_size[0], 1],
#                                                 strides=[stride, 1],
#                                                 padding='same',
#                                                 kernel_initializer=INITIALIZER,
#                                                 data_format='channels_first',
#                                                 kernel_regularizer=REGULARIZER))
#         self.tgcn.add(tf.keras.layers.BatchNormalization(axis=1))#why use at the end of the model as well??

#         self.act = tf.keras.layers.Activation(activation)

#         if not residual:
#             self.residual = lambda x, training=False: 0
#         elif residual and stride == 1 and not downsample:
#             self.residual = lambda x, training=False: x
#         else:
#             self.residual = tf.keras.Sequential()
#             self.residual.add(tf.keras.layers.Conv2D(filters,
#                                                         kernel_size=[1, 1],
#                                                         strides=[stride, 1],
#                                                         padding='same',
#                                                         kernel_initializer=INITIALIZER,
#                                                         data_format='channels_first',
#                                                         kernel_regularizer=REGULARIZER))
#             self.residual.add(tf.keras.layers.BatchNormalization(axis=1))

#     def call(self, x, A, training):
#         res = self.residual(x, training=training)#lambda function
#         x, A = self.sgcn(x, A, training=training)
#         x = self.tgcn(x, training=training)
#         x += res
#         x = self.act(x) #16,64,300,25
#         return x, A



# class Model(tf.keras.Model):
#     def __init__(self, num_classes=60):
#         super().__init__()

#         graph = Graph()
#         self.A = tf.Variable(graph.A,
#                              dtype=tf.float32,
#                              trainable=False,
#                              name='adjacency_matrix')

#         self.data_bn = tf.keras.layers.BatchNormalization(axis=1)

#         self.STGCN_layers = []
#         self.STGCN_layers.append(STGCN(64, residual=False))
#         self.STGCN_layers.append(STGCN(64))
#         self.STGCN_layers.append(STGCN(64))
#         self.STGCN_layers.append(STGCN(64))
#         self.STGCN_layers.append(STGCN(128, stride=2, downsample=True))
#         self.STGCN_layers.append(STGCN(128))
#         self.STGCN_layers.append(STGCN(128))
#         self.STGCN_layers.append(STGCN(256, stride=2, downsample=True))
#         self.STGCN_layers.append(STGCN(256))
#         self.STGCN_layers.append(STGCN(256))

#         self.pool = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_first')

#         self.logits = tf.keras.layers.Conv2D(num_classes,
#                                              kernel_size=1,
#                                              padding='same',
#                                              kernel_initializer=INITIALIZER,
#                                              data_format='channels_first',
#                                              kernel_regularizer=REGULARIZER)

#     def call(self, x, training):
#         N = tf.shape(x)[0]
#         C = tf.shape(x)[1]
#         T = tf.shape(x)[2]
#         V = tf.shape(x)[3]
#         M = tf.shape(x)[4]

#         x = tf.transpose(x, perm=[0, 4, 3, 1, 2])
#         x = tf.reshape(x, [N * M, V * C, T])
#         x = self.data_bn(x, training=training)#batch normalizarion
#         x = tf.reshape(x, [N, M, V, C, T])
#         x = tf.transpose(x, perm=[0, 1, 3, 4, 2])
#         x = tf.reshape(x, [N * M, C, T, V])

#         A = self.A
#         indexLayer=0
#         returnLayer=None
#         for layer in self.STGCN_layers:
#             x, A = layer(x, A, training=training)#check if A changes inside ie learns something
#             #there are 10 STGCN layers , layer==9 is not the final !!
#             if indexLayer==5: #get the output of GCN layer as a feature map
#                 returnLayer=x# not returned atm
#             indexLayer+=1
#             # print(indexLayer) 


#         # N*M,C,T,V
#         x = self.pool(x)
#         a_=x            #(M,256)
#         a_=self.pool(returnLayer)
#         x = tf.reshape(x, [N, M, -1, 1, 1])
#         x = tf.reduce_mean(x, axis=1)
#         x = self.logits(x)

#         #print(self.logits.weights[0])
#         x = tf.reshape(x, [N, -1]) #shape = (batch size,number of classes)

#         return a_, x


# class Model2(tf.keras.Model):
#     def __init__(self, num_classes=10):
#         super().__init__()

     
#         self.dense2=tf.keras.layers.Dense(512, activation='relu')
#         # self.dense2=tf.keras.layers.Dense(2, activation='relu')
#         self.dense3=tf.keras.layers.Dense(num_classes)#, activation='relu'
        
        
        


#     def call(self, inputs):
#         #x = self.dense1(inputs)
#         x = self.dense2(inputs)
#         #print(self.dense2.weights[0])
#         x= self.dense3(x)
#         return x