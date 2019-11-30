import numpy as np




class ACModel:

    def __init__(self, eps, alpha, beta, ):
        self.eps = 1e-8
        self.alpha, self.beta, self.gamma = 1.0, 1.0, 1.0
        self.base_units = NF
        self.latent_units = 4
        self.n_layer = np.uint8(np.log2(self.base_units/self.latent_units))

         # kernel initializers
        KIS = {'glorot_uniform': glorot_uniform(SEED),
            'lecun_uniform': lecun_uniform(SEED),
            'he_uniform': he_uniform(SEED),
            'varscale': VarianceScaling(seed=SEED),
            'glorot_normal': glorot_normal(SEED),
            'lecun_normal': lecun_normal(SEED),
            'he_normal': he_normal(SEED)}

        # optmizers
        OPTS = {'sgd': SGD(lr=LR, momentum=0.0, decay=0.0, nesterov=True),
                'adadelta': Adadelta(lr=LR, rho=0.95, epsilon=None, decay=0.0),
                'adam': Adam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True),
                'adamax': Adamax(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0),
                'nadam': Nadam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)}

        # reconstruction losses
        RCLS = {'mse': lambda a, b: se(a, b),
                'mae': lambda a, b: ae(a, b),
                'rmse': lambda a, b: rse(a, b),
                'bc': lambda a, b: bc(a, b)}

        # regularizers
        REGS = {'kld': lambda a: kld(a),
                'mmd': lambda a, b: mmd(a, b),
                'sw': lambda a: sw(a),
                'tc': lambda a, b: tc(a, b)}

        enc_input = Input(shape=(NF,), name='encoder_input')
        self.encoder = self.build_encoder(enc_input)
        enc_output = self.encoder(enc_input)[2]

        self.decoder = self.build_decoder()
        dec_output = self.decoder(enc_output)

        self.vae = Model(enc_input, dec_output, name='vae')
        self.vae.summary()
        self.vae.add_loss(self.vae_loss(enc_input, dec_output))
        self.vae.compile(optimizer=Nadam(lr=1e-3))

    def gauss_sampling(beta, batch_size=None):
        ''' samples a point in a multivariate gaussian distribution '''
        if batch_size is None:
            batch_size = BS
        mu, logvar = beta
        epsilon = K.random_normal(shape=(batch_size, LD), seed=SEED)
        return mu+K.exp(0.5*logvar)*epsilon


    def gauss_log_prob(z, beta=None, batch_size=None):
        ''' logarithmic probability for multivariate gaussian distribution given samples z and parameters beta = (mu, log(var)) '''
        if batch_size is None:
            batch_size = BS
        if beta is None:
            # mu = 0, stdv = 1 => log(var) = 0
            mu, logvar = K.zeros((batch_size, LD)), K.zeros((batch_size, LD))
        else:
            mu, logvar = beta
        norm = K.log(2*np.pi)
        zsc = (z-mu)*K.exp(-0.5*logvar)
        return -0.5*(zsc**2+logvar+norm)


    def se(x, y, batch_size=None):
        ''' squared error between samples x and predictions y '''
        if batch_size is None:
            batch_size = BS
        return K.sum(K.reshape(K.square(x-y), (batch_size, -1)), 1)


    def ae(x, y, batch_size=None):
        ''' absolute error between samples x and predictions y '''
        if batch_size is None:
            batch_size = BS
        return K.sum(K.reshape(K.abs(x-y), (batch_size, -1)), 1)


    def rse(x, y, batch_size=None):
        if batch_size is None:
            batch_size = BS
        return K.sqrt(K.sum(K.reshape(K.square(x-y), (batch_size, -1)), 1))


    def bc(x, y, batch_size=None):
        ''' binary crossentropy between samples x and predictions y '''
        if batch_size is None:
            batch_size = BS
        return -K.sum(K.reshape(x*K.log(y+EPS)+(1-x)*K.log(1-y+EPS),
                                (batch_size, -1)), 1)


    def kld(beta):
        ''' kullback-leibler divergence for gaussian parameters beta = (mu, log(var)) '''
        mu, logvar = beta
        return 0.5*K.sum(K.exp(logvar)+K.square(mu)-logvar-1, axis=-1)


    def kernel_computation(x, y, batch_size=None):
        ''' kernel trick computation for maximum mean discrepancy ccalculation '''
        if batch_size is None:
            batch_size = BS
        tiled_x = K.tile(K.reshape(x, (batch_size, 1, LD)), (1, batch_size, 1))
        tiled_y = K.tile(K.reshape(y, (1, batch_size, LD)), (batch_size, 1, 1))
        return K.exp(-K.mean(K.square(tiled_x-tiled_y), axis=2)/K.cast(LD, 'float32'))


    def mmd(x, y, batch_size=None):
        ''' maximum mean discrepancy for samples x and predictions y '''
        if batch_size is None:
            batch_size = BS
        x_kernel = kernel_computation(x, x, batch_size)
        y_kernel = kernel_computation(y, y, batch_size)
        xy_kernel = kernel_computation(x, y, batch_size)
        return x_kernel+y_kernel-2*xy_kernel


    def sw(z, batch_size=None):
        ''' sliced wasserstein calculation for gaussian samples z'''
        if batch_size is None:
            batch_size = BS
        nrp = batch_size**2
        theta = K.random_normal(shape=(nrp, LD))
        theta = theta/K.sqrt(K.sum(K.square(theta), axis=1, keepdims=True))
        y = K.random_uniform(shape=(batch_size, LD), minval=-0.5, maxval=0.5, seed=SEED)
        px = K.dot(z, K.transpose(theta))
        py = K.dot(y, K.transpose(theta))
        w2 = (tf.nn.top_k(tf.transpose(px), k=batch_size).values-
            tf.nn.top_k(tf.transpose(py), k=batch_size).values)**2
        return w2


    def log_importance_weight(batch_size=None, dataset_size=None):
        ''' logarithmic importance weights for minibatch stratified sampling '''
        if batch_size is None:
            batch_size = BS
        if dataset_size is None:
            dataset_size = SNH*SNT*SNS
        n, m = dataset_size, batch_size-1
        strw = (n-m)/(n*m)
        w = K.concatenate((K.concatenate((1/n*K.ones((batch_size-2, 1)),
                                        strw*K.ones((1, 1)),
                                        1/n*K.ones((1, 1))), axis=0),
                        strw*K.ones((batch_size, 1)),
                        1/m*K.ones((batch_size, batch_size-2))), axis=1)
        return K.log(w)


    def log_sum_exp(x):
        ''' numerically stable logarithmic sum of exponentials '''
        m = K.max(x, axis=1, keepdims=True)
        u = x-m
        m = K.squeeze(m, 1)
        return m+K.log(K.sum(K.exp(u), axis=1, keepdims=False))


    def tc(z, beta, batch_size=None):
        ''' modified elbo objective using tc decomposition given gaussian samples z and parameters beta = (mu, log(var)) '''
        if batch_size is None:
            batch_size = BS
        mu, logvar = beta
        # log p(z)
        logpz = K.sum(K.reshape(gauss_log_prob(z), (batch_size, -1)), 1)
        # log q(z|x)
        logqz_x = K.sum(K.reshape(gauss_log_prob(z, (mu, logvar)), (batch_size, -1)), 1)
        # log q(z) ~ log (1/MN) sum_m q(z|x_m) = -log(MN)+log(sum_m(exp(q(z|x_m))))
        _logqz = gauss_log_prob(K.reshape(z, (batch_size, 1, LD)),
                                (K.reshape(mu, (1, batch_size, LD)),
                                K.reshape(logvar, (1, batch_size, LD))))
        if MSS:
            log_iw = log_importance_weight(batch_size=batch_size)
            logqz_prodmarginals = K.sum(log_sum_exp(K.reshape(log_iw, (batch_size, batch_size, 1))+_logqz), 1)
            logqz = log_sum_exp(log_iw+K.sum(_logqz, axis=2))
        else:
            logqz_prodmarginals = K.sum(log_sum_exp(_logqz)-K.log(K.cast(BS*SNH*SNT*SNS, 'float32')), 1)
            logqz = log_sum_exp(K.sum(_logqz, axis=2))-K.log(K.cast(BS*SNH*SNT*SNS, 'float32'))
        # alpha controls index-code mutual information
        # beta controls total correlation
        # lambda controls dimension-wise kld
        melbo = -ALPHA*(logqz_x-logqz)-BETA*(logqz-logqz_prodmarginals)-LMBDA*(logqz_prodmarginals-logpz)
        return -melbo

    def sampler(self, beta):
        mu, logvar = beta
        return self.mu+K.exp(0.5*logvar)*K.random_normal(shape=(K.shape(mu)[0], self.latent_units), seed=SEED)

    def kld(self):
        return 0.5*K.sum(K.exp(self.logvar)+K.square(self.mu)-self.logvar-1, axis=-1)

    def reconstruction(self, enc_input, dec_output):
        return K.sum(K.square(enc_input-dec_output), axis=-1)

    def vae_loss(self, enc_input, dec_output):
        return K.mean(self.alpha*self.reconstruction(enc_input, dec_output)+self.beta*self.kld())

    def build_encoder(self, enc_input):
        d = Dense(units=self.base_units, kernel_initializer='lecun_normal', activation='selu', name='encoder_dense_0')(enc_input)
        d = Dense(units=self.base_units//2, kernel_initializer='lecun_normal', activation='selu', name='encoder_dense_1')(d)
        d = Dense(units=self.base_units//4, kernel_initializer='lecun_normal', activation='selu', name='encoder_dense_2')(d)
        d = Dense(units=self.base_units//8, kernel_initializer='lecun_normal', activation='selu', name='encoder_dense_3')(d)
        self.mu = Dense(self.latent_units, kernel_initializer='lecun_normal', activation='linear', name='encoder_mu')(d)
        self.logvar = Dense(self.latent_units, kernel_initializer='lecun_normal', activation='linear', name='encoder_logvar')(d)
        z = Lambda(self.sampler, output_shape=(self.latent_units,), name='encoder_output')([self.mu, self.logvar])
        encoder = Model(enc_input, [self.mu, self.logvar, z], name='encoder')
        encoder.summary()
        return encoder

    def build_decoder(self):
        dec_input = Input(shape=(self.latent_units,), name='decoder_input')
        d = Dense(units=self.base_units//8, kernel_initializer='lecun_normal', activation='selu', name='decoder_dense_0')(dec_input)
        d = Dense(units=self.base_units//4, kernel_initializer='lecun_normal', activation='selu', name='decoder_dense_1')(d)
        d = Dense(units=self.base_units//2, kernel_initializer='lecun_normal', activation='selu', name='decoder_dense_2')(d)
        d = Dense(units=self.base_units, kernel_initializer='lecun_normal', activation='selu', name='decoder_dense_3')(d)
        dec_output = Dense(units=NF, kernel_initializer='lecun_normal', activation='sigmoid', name='decoder_output')(d)
        decoder = Model(dec_input, dec_output, name='decoder')
        decoder.summary()
        return decoder