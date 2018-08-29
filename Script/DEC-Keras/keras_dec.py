'''
Keras implementation of deep embedder to improve clustering, inspired by:
"Unsupervised Deep Embedding for Clustering Analysis" (Xie et al, ICML 2016)
Definition can accept somewhat custom neural networks. Defaults are from paper.
'''
import sys
import numpy as np
import keras.backend as K
from keras.initializers import RandomNormal
from keras.engine.topology import Layer, InputSpec
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Input
from keras.optimizers import SGD
from sklearn.preprocessing import normalize
from keras.callbacks import LearningRateScheduler
# 匈牙利算法 输入代价矩阵 返回寻找的成本最小化分配。
from sklearn.utils.linear_assignment_ import linear_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import time
import tensorflow as tf

start_time = time.time()
def consume_time():
    end_time = time.time()
    interval = end_time - start_time
    seconds = interval
    if seconds > 60:
        minutes = int(seconds/60)
        seconds = seconds % 60
        seconds = round(seconds, 2)
        print("目前耗时 {} 分 {} 秒".format(minutes, seconds))
    else:
        print("目前耗时 {} 秒".format(round(seconds, 2)))

if (sys.version[0] == 2):
    import cPickle as pickle
else:
    import pickle
import numpy as np

class ClusteringLayer(Layer):
    '''
    Clustering layer which converts latent space Z of input layer
    into a probability vector for each cluster defined by its centre in
    Z-space. Use Kullback-Leibler divergence as loss, with a probability
    target distribution.
    # Arguments
        output_dim: int > 0. Should be same as number of clusters.
        input_dim: dimensionality of the input (integer).
            This argument (or alternatively, the keyword argument `input_shape`)
            is required when using this layer as the first layer in a model.
        weights: list of Numpy arrays to set as initial weights.
            The list should have 2 elements, of shape `(input_dim, output_dim)`
            and (output_dim,) for weights and biases respectively.
        alpha: parameter in Student's t-distribution. Default is 1.0.
    # Input shape
        2D tensor with shape: `(nb_samples, input_dim)`.
    # Output shape
        2D tensor with shape: `(nb_samples, output_dim)`.
    '''
    def __init__(self, output_dim, input_dim=None, weights=None, alpha=1.0, **kwargs):
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.alpha = alpha
        # kmeans cluster centre locations
        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim=2)]

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(ClusteringLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     shape=(None, input_dim))]

        self.W = K.variable(self.initial_weights)
        self.trainable_weights = [self.W]

    def call(self, x, mask=None):
        q = 1.0/(1.0 + K.sqrt(K.sum(K.square(K.expand_dims(x, 1) - self.W), axis=2))**2 /self.alpha)
        q = q**((self.alpha+1.0)/2.0)
        q = K.transpose(K.transpose(q)/K.sum(q, axis=1))
        return q

    def get_output_shape_for(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return (input_shape[0], self.output_dim)

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'input_dim': self.input_dim}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DeepEmbeddingClustering(object):
    def __init__(self,
                 n_clusters,
                 input_dim,
                 encoded=None,
                 decoded=None,
                 alpha=1.0,
                 pretrained_weights=None,
                 cluster_centres=None,
                 batch_size=256,
                 **kwargs):

        super(DeepEmbeddingClustering, self).__init__()

        self.n_clusters = n_clusters
        self.input_dim = input_dim
        self.encoded = encoded
        self.decoded = decoded
        self.alpha = alpha
        self.pretrained_weights = pretrained_weights
        self.cluster_centres = cluster_centres
        self.batch_size = batch_size

        self.learning_rate = 0.1
        self.iters_lr_update = 20000
        self.lr_change_rate = 0.1

        # greedy layer-wise training before end-to-end training:

        print('\n\ninput_dim is ', self.input_dim)
        self.encoders_dims = [self.input_dim, 500, 500, 2000, 10]

        self.input_layer = Input(shape=(self.input_dim,), name='input')
        dropout_fraction = 0.2
        init_stddev = 0.01

        # 分层自动编码 收集每个autoencoding层  共有len(encoding_dims)层
        self.layer_wise_autoencoders = []
        self.encoders = []
        self.decoders = []
        for i  in range(1, len(self.encoders_dims)):

            encoder_activation = 'linear' if i == (len(self.encoders_dims) - 1) else 'relu'
            encoder = Dense(self.encoders_dims[i], activation=encoder_activation,
                            input_shape=(self.encoders_dims[i-1],),
                            kernel_initializer=RandomNormal(mean=0.0, stddev=init_stddev, seed=None),
                            bias_initializer='zeros', name='encoder_dense_%d'%i)
            self.encoders.append(encoder)

            decoder_index = len(self.encoders_dims) - i
            decoder_activation = 'linear' if i == 1 else 'relu'
            decoder = Dense(self.encoders_dims[i-1], activation=decoder_activation,
                            kernel_initializer=RandomNormal(mean=0.0, stddev=init_stddev, seed=None),
                            bias_initializer='zeros',
                            name='decoder_dense_%d'%decoder_index)
            self.decoders.append(decoder)

            autoencoder = Sequential([
                Dropout(dropout_fraction, input_shape=(self.encoders_dims[i-1],),
                        name='encoder_dropout_%d'%i),
                encoder,
                Dropout(dropout_fraction, name='decoder_dropout_%d'%decoder_index),
                decoder
            ])
            autoencoder.compile(loss='mse', optimizer=SGD(lr=self.learning_rate, decay=0, momentum=0.9))
            self.layer_wise_autoencoders.append(autoencoder)

        # build the end-to-end autoencoder for finetuning
        # Note that at this point dropout is discarded
        self.encoder = Sequential(self.encoders)
        self.encoder.compile(loss='mse', optimizer=SGD(lr=self.learning_rate, decay=0, momentum=0.9))
        self.decoders.reverse()
        self.autoencoder = Sequential(self.encoders + self.decoders)
        self.autoencoder.compile(loss='mse', optimizer=SGD(lr=self.learning_rate, decay=0, momentum=0.9))

        if cluster_centres is not None:
            assert cluster_centres.shape[0] == self.n_clusters
            assert cluster_centres.shape[1] == self.encoder.layers[-1].output_dim

        if self.pretrained_weights is not None:
            self.autoencoder.load_weights(self.pretrained_weights)

    def p_mat(self, q):
        weight = q**2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def initialize(self, X, save_autoencoder=False, layerwise_pretrain_iters=50000, finetune_iters=100000):
        if self.pretrained_weights is None:

            iters_per_epoch = int(len(X) / self.batch_size)
            layerwise_epochs = max(int(layerwise_pretrain_iters / iters_per_epoch), 1)

            # too long !!!----------------------------------------------------------1
            layerwise_epochs = 18

            finetune_epochs = max(int(finetune_iters / iters_per_epoch), 1)

            # too long !!!----------------------------------------------------------2
            finetune_epochs = 18

            print('前-训练 分层 layerwise pretrain')
            current_input = X
            print('\n current_input ', current_input)
            lr_epoch_update = max(1, self.iters_lr_update / float(iters_per_epoch))

            def step_decay(epoch):
                initial_rate = self.learning_rate
                factor = int(epoch / lr_epoch_update)
                lr = initial_rate / (10 ** factor)
                return lr
            # 作为回调函数的一员,LearningRateScheduler 可以按照epoch的次数自动调整学习率
            # returns a new learning rate as output (float)
            lr_schedule = LearningRateScheduler(step_decay)




            for i, autoencoder in enumerate(self.layer_wise_autoencoders):


                #--------------------------------------------------------------5
                for d in ['/device:GPU:2', '/device:GPU:3']:
                    with tf.device(d):

                        consume_time()
                        print('循环次数: ', i ,'/', len(self.layer_wise_autoencoders))
                        time.sleep(1)
                        if i > 0:
                            weights = self.encoders[i-1].get_weights()
                            dense_layer = Dense(self.encoders_dims[i], input_shape=(current_input.shape[1],),
                                                activation='relu', weights=weights,
                                                name='encoder_dense_copy_%d'%i)
                            encoder_model = Sequential([dense_layer])
                            encoder_model.compile(loss='mse', optimizer=SGD(lr=self.learning_rate, decay=0, momentum=0.9))
                            current_input = encoder_model.predict(current_input)

                        autoencoder.fit(current_input, current_input,
                                        batch_size=self.batch_size, epochs=layerwise_epochs, callbacks=[lr_schedule], verbose=2)
                        self.autoencoder.layers[i].set_weights(autoencoder.layers[1].get_weights())
                        self.autoencoder.layers[len(self.autoencoder.layers) - i - 1].set_weights(autoencoder.layers[-1].get_weights())



            consume_time()
            print('微调自动编码 Finetuning autoencoder')




            #update encoder and decoder weights: ------------------------------------
            print('update encoder and decoder weights')
            self.autoencoder.fit(X, X, batch_size=self.batch_size, epochs=finetune_epochs, callbacks=[lr_schedule], verbose=2)

            if save_autoencoder:
                print('\n\nautoencoder.save_weights(autoencoder.h5)\n\n')
                self.autoencoder.save_weights('autoencoder.h5')
        else:
            print('Loading pretrained weights for autoencoder.')
            self.autoencoder.load_weights(self.pretrained_weights)

        # update encoder, decoder
        # TODO: is this needed? Might be redundant...
        for i in range(len(self.encoder.layers)):
            self.encoder.layers[i].set_weights(self.autoencoder.layers[i].get_weights())

        # initialize cluster centres using k-means
        print('\n\nInitializing cluster centres with k-means.\n')
        print('有没有给cluster_center ?')
        if self.cluster_centres is None:
            print('No')
            kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
            self.y_pred = kmeans.fit_predict(self.encoder.predict(X))
            self.cluster_centres = kmeans.cluster_centers_
        else:
            print('Yes')

        print('分成的类数的中心位置： ', self.cluster_centres, '\n', self.cluster_centres.shape)
        # prepare DEC model
        #self.DEC = Model(inputs=self.input_layer,
        #                 outputs=ClusteringLayer(self.n_clusters,
        #                                        weights=self.cluster_centres,
        #                                        name='clustering')(self.encoder))
        self.DEC = Sequential([self.encoder,
                             ClusteringLayer(self.n_clusters,
                                                weights=self.cluster_centres,
                                                name='clustering')])
        self.DEC.compile(loss='kullback_leibler_divergence', optimizer='adadelta')
        return

    # 准确率的判定----------------------------
    def cluster_acc(self, y_true, y_pred):
        assert y_pred.size == y_true.size
        # why?
        D = max(y_pred.max(), y_true.max())+1
        # weight
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        ind = linear_assignment(w.max() - w)
        return sum([w[i, j] for i, j in ind])*1.0/y_pred.size, w


    def cluster(self, X, y=None,
                # 什么意思
                tol=0.01,
                update_interval=None,

                # 设置最大的迭代训练次数
                iter_max=1e6,
                save_interval=None,
                **kwargs):

        # 真不知道是什么
        if update_interval is None:
            # 1 epochs
            update_interval = X.shape[0]/self.batch_size
        print('Update interval', update_interval)

        if save_interval is None:
            # 50 epochs
            save_interval = X.shape[0]/self.batch_size*50
        print('Save interval', save_interval)

        assert save_interval >= update_interval

        train = True
        iteration, index = 0, 0
        self.accuracy = []

        print('开始训练')
        consume_time()
        while train:
            sys.stdout.write('\r')
            # cutoff iteration
            if iter_max < iteration:
                print('Reached maximum iteration limit. Stopping training.')
                return self.y_pred

            # update (or initialize) probability distributions and propagate weight changes
            # from DEC model to encoder.
            # 输出条件 ------------------------------------------------------ 4
            if iteration/3 % update_interval == 0:
                consume_time()

                self.q = self.DEC.predict(X, verbose=0)                  # =================== 预测！！
                self.p = self.p_mat(self.q)

                # 什么东西   =============== 预测数据 分10类
                y_pred = self.q.argmax(1)


                # --------------------------3
                print('y_pred and self.y_pred:\n', y_pred, '\n', self.y_pred)


                delta_label = ((y_pred == self.y_pred).sum().astype(np.float32) / y_pred.shape[0])



                # y用来计算准确率的=================================================================
                if y is not None:
                    acc = self.cluster_acc(y, y_pred)[0]
                    # import pdb; pdb.set_trace()
                    self.accuracy.append(acc)
                    print('Iteration '+str(iteration)+', Accuracy '+str(np.round(acc, 5)))
                else:
                    print(str(np.round(delta_label*100, 5))+'% change in label assignment')


                # 循环终止条件！
                print('循环终止条件  ==========     delta_label < tol')
                print('delta_label is     {} \ntol is                {}'.format(delta_label, tol))
                # -------------------------------------------------------------------------------------
                # import pdb;pdb.set_trace()
                if delta_label < tol:
                    print('达到容差阈值。 停止训练 Reached tolerance threshold. Stopping training.')
                    train = False
                    continue
                else:
                    self.y_pred = y_pred

                for i in range(len(self.encoder.layers)):
                    self.encoder.layers[i].set_weights(self.DEC.layers[0].layers[i].get_weights())
                self.cluster_centres = self.DEC.layers[-1].get_weights()[0]

            # train on batch
            sys.stdout.write('迭代次数 Iteration %d, ' % iteration)
            if (index+1)*self.batch_size > X.shape[0]:
                loss = self.DEC.train_on_batch(X[index*self.batch_size::], self.p[index*self.batch_size::])
                index = 0
                sys.stdout.write('Loss %f' % loss)
            else:
                loss = self.DEC.train_on_batch(X[index*self.batch_size:(index+1) * self.batch_size],
                                               self.p[index*self.batch_size:(index+1) * self.batch_size])
                sys.stdout.write('Loss %f' % loss)
                index += 1

            # save intermediate
            if iteration % save_interval == 0:
                print('iteration % save_interval == 0')
                z = self.encoder.predict(X)
                pca = PCA(n_components=2).fit(z)
                z_2d = pca.transform(z)
                clust_2d = pca.transform(self.cluster_centres)
                # save states for visualization========================
                pickle.dump({'z_2d': z_2d, 'clust_2d': clust_2d, 'q': self.q, 'p': self.p},
                            open('c'+str(iteration)+'.pkl', 'wb'))
                # save DEC model checkpoints======================
                self.DEC.save('DEC_model_'+str(iteration)+'_.h5')

            iteration += 1
            # 好设计---
            sys.stdout.flush()


        print('训练完成')
        consume_time()
        return

