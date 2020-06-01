#!/usr/bin/env python
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input, BatchNormalization, MaxPooling2D
from tensorflow.keras import Model
from tensorflow.keras.losses import Huber
import numpy as np

class DQN():
    def __init__(self):
        self.K = 4
        image_size=64
        self.X_img = Input(shape=[image_size,image_size,3])
        Z_image = self.X_img / 255.0
        #Z_image = Conv2D(32, 3,input_shape=(image_size, image_size, 3), activation='relu')(Z_image)
        #Z_image = MaxPooling2D(pool_size=(2, 2))(Z_image)
        #Z_image = BatchNormalization()(Z_image)
        #self.Z_image=Z_image
        Z_image = Conv2D(32, 3, activation='relu')(Z_image)
        Z_image = MaxPooling2D(pool_size=(2, 2))(Z_image)
        Z_image = BatchNormalization()(Z_image)
        #self.Z_image=Z_image
        #Z_image = Conv2D(64, 3, activation='relu')(Z_image)
        #Z_image = MaxPooling2D(pool_size=(2, 2))(Z_image)
        #Z_image = BatchNormalization()(Z_image)
        self.Z_image=Z_image
        self.model = Model(inputs=[self.X_img], outputs=self.Z_image)
        Z_image = Conv2D(64, 3, activation='relu')(Z_image)
        
        self.Z_image=Z_image
        #self.Z_image=Z_image
        #Z_image = Flatten()(Z_image)
        #self.model = Model(inputs=[self.X_img], outputs=self.Z_image)
        self.predict_op = Dense(self.K)(Z_image)

        
        
        
        
        
        #self.model = Model(inputs=[self.X_img], outputs=self.Z_image)
        #self.loss_object  = Huber()
        
        
        
        #self.train_op = tf.keras.optimizers.Adam(1e-6)
            
    def copy_from(self, other):
        self.model.set_weights(other.model.get_weights()) 
    
    def save(self):
        self.model.save_weights('model_weights.h5')
    
    def load(self):
        self.model.load_weights('model_weights.h5')
        
    def predict(self,states):
        return self.model.predict(states)
    
    def get_weights(self):
        return np.reshape(self.model.get_weights()[-2], (-1))
    
    @tf.function
    def train_step(self,states,states_lsr, actions, targets):
        with tf.GradientTape() as tape:
            predictions = self.model([states,states_lsr], training=True)
            selected_action_value = tf.reduce_sum(predictions * tf.one_hot(actions,self.K), axis=1)
            cost = tf.reduce_mean(self.loss_object(targets, selected_action_value))
        gradients = tape.gradient(cost, self.model.trainable_variables)
        self.train_op.apply_gradients(zip(gradients, self.model.trainable_variables))
        return cost
      
    def sample_action(self,x_img,x_lsr,eps):
        if np.random.random() < eps:
            return np.random.choice(self.K)
        else:
            inp1 = tf.expand_dims(x_img, 0)
            inp2 = tf.expand_dims(x_lsr, 0)
            return np.argmax(self.predict([inp1,inp2])[0])