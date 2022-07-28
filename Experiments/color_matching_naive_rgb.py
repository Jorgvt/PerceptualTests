import pickle
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import numpy as np
import tensorflow as tf

from perceptualtests.baselines import NaiveModel
from perceptualtests.matching import ColorMatchingExperiment, xyz2nrgb_tf
from perceptualtests.matching import ReduceLROnPlateau, EarlyStopping

def rmse(img1, img2):
    loss = (img1 - img2)**2
    loss = tf.reduce_sum(loss, axis=[1,2,3])
    loss = loss**(1/2)
    return loss

model = NaiveModel()

cme = ColorMatchingExperiment(wavelengths=np.linspace(380, 770, 50),
                              central_wavelengths=[400.0, 490.0, 550.0, 605.00],
                              lambdas=np.linspace(380,770,300, dtype=np.float32),
                              max_radiance=1.5e-3, #1.5e-3 #1.25e-4
                              background_radiance=0.5e-4, #0.1e-4 #0.5e-4
                              img_size=(256, 256),
                              space_transform_fn=lambda x: xyz2nrgb_tf(x, gamma=2.0, clip=False),
                              width_lambda=10,
                              width_central_lambdas=5,
                              initial_weights=[0.001, 0.001, 0.001, 0.001],
                              norm_grads=True)

optimizer = tf.optimizers.SGD(learning_rate=3e-4) #5e-6
cme.compile(loss=rmse,
            optimizer=optimizer)

def create_lr_cb(**kwargs):
    return ReduceLROnPlateau(**kwargs)

def create_earlystopping_cb(**kwargs):
    return EarlyStopping(**kwargs)

histories = cme.fit(model=NaiveModel(),
                    epochs=10000,
                    verbose=100,
                    use_tqdm=False,
                    lr_cb_fn=lambda : create_lr_cb(factor=0.5,
                                                   patience=5,
                                                   cooldown=2,
                                                   min_lr=1e-10,
                                                   delta=1,
                                                   optimizer=tf.optimizers.SGD(learning_rate=3e-4),
                                                   verbose=True),
                    es_cb_fn=lambda : create_earlystopping_cb(patience=100,
                                                              delta=1,
                                                              verbose=True))

with open('histories_color_matching_naive_rgb_normgrad_newlambdas_cbs.pkl', 'wb') as f:
    pickle.dump(histories, f)