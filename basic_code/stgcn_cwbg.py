#embeddding is also given by this code for  the t-sne visualization
from ntu_rgb_d_cwbg import Graph
import tensorflow as tf
import numpy as np

REGULARIZER = tf.keras.regularizers.l2(l=0.01) 
INITIALIZER = tf.keras.initializers.VarianceScaling(scale=2.,
                                                    mode="fan_in",
                                                    distribution="truncated_normal") 
# INITIALIZER=tf.keras.initializers.glorot_uniform()


class SGCN(tf.keras.Model):
    def __init__(self, filters, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = tf.keras.layers.Conv2D(filters*kernel_size,
                                           kernel_size=1,
                                           padding='same',
                                           kernel_initializer=INITIALIZER,
                                           data_format='channels_first',
                                           kernel_regularizer=REGULARIZER)

    def call(self, x, A, training):
        x = self.conv(x)

        N = tf.shape(x)[0]
        C = tf.shape(x)[1] 
        T = tf.shape(x)[2]
        V = tf.shape(x)[3]

        x = tf.reshape(x, [N, self.kernel_size, C//self.kernel_size, T, V])
        x = tf.einsum('nkctv,kvw->nctw', x, A) # 16, 3, 64, 300, 25 --*--3, 25, 25
        return x, A
        

class STGCN(tf.keras.Model):
    def __init__(self, filters, kernel_size=[9, 3], stride=1, activation='relu',
                 residual=True, downsample=False):
        super().__init__()
        self.sgcn = SGCN(filters, kernel_size=kernel_size[1])

        self.tgcn = tf.keras.Sequential()
        self.tgcn.add(tf.keras.layers.BatchNormalization(axis=1))
        self.tgcn.add(tf.keras.layers.Activation(activation))
        self.tgcn.add(tf.keras.layers.Conv2D(filters,
                                                kernel_size=[kernel_size[0], 1],
                                                strides=[stride, 1],
                                                padding='same',
                                                kernel_initializer=INITIALIZER,
                                                data_format='channels_first',
                                                kernel_regularizer=REGULARIZER))
        self.tgcn.add(tf.keras.layers.BatchNormalization(axis=1))

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
                                                        kernel_initializer=INITIALIZER,
                                                        data_format='channels_first',
                                                        kernel_regularizer=REGULARIZER))
            self.residual.add(tf.keras.layers.BatchNormalization(axis=1))

    def call(self, x, A, training):
        res = self.residual(x, training=training)
        x, A = self.sgcn(x, A, training=training)
        x = self.tgcn(x, training=training)
        x += res
        x = self.act(x) #16,64,300,25
        return x, A


class Model(tf.keras.Model):
    def __init__(self, num_classes=60):
        super().__init__()

        graph = Graph()
        self.A = tf.Variable(graph.A,
                             dtype=tf.float32,
                             trainable=False,
                             name='adjacency_matrix')

        self.data_bn = tf.keras.layers.BatchNormalization(axis=1)

        self.STGCN_layers = []
        self.STGCN_layers.append(STGCN(64, residual=False))
        self.STGCN_layers.append(STGCN(64))
        self.STGCN_layers.append(STGCN(64))
        self.STGCN_layers.append(STGCN(64))
        self.STGCN_layers.append(STGCN(128, stride=2, downsample=True))
        self.STGCN_layers.append(STGCN(128))
        self.STGCN_layers.append(STGCN(128))
        self.STGCN_layers.append(STGCN(256, stride=2, downsample=True))
        self.STGCN_layers.append(STGCN(256))
        self.STGCN_layers.append(STGCN(256))

        self.pool = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_first')

        self.logits = tf.keras.layers.Conv2D(num_classes,
                                             kernel_size=1,
                                             padding='same',
                                             kernel_initializer=INITIALIZER,
                                             data_format='channels_first',
                                             kernel_regularizer=REGULARIZER)

    def call(self, x, training):
        N = tf.shape(x)[0]
        C = tf.shape(x)[1]
        T = tf.shape(x)[2]
        V = tf.shape(x)[3]
        M = tf.shape(x)[4]

        x = tf.transpose(x, perm=[0, 4, 3, 1, 2])
        x = tf.reshape(x, [N * M, V * C, T])
        x = self.data_bn(x, training=training)
        x = tf.reshape(x, [N, M, V, C, T])
        x = tf.transpose(x, perm=[0, 1, 3, 4, 2])
        x = tf.reshape(x, [N * M, C, T, V])

        A = self.A
        for layer in self.STGCN_layers:
            x, _ = layer(x, A, training=training)

        x = self.pool(x)
        x = tf.reshape(x, [N, M, -1, 1, 1])
        x = tf.reduce_mean(x, axis=1) 
        embed=x
        x = self.logits(x)

        #print(self.logits.weights[0])
        x = tf.reshape(x, [N, -1]) #shape = (batch size,number of classes)

        return x , embed


