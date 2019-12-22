import tensorflow as tf
import numpy as np
from keras import backend as K
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, GlobalMaxPooling2D, Input, Lambda, Multiply
from keras import regularizers
from keras.models import Model

def huber_loss(y_true, y_pred):
        return tf.losses.huber_loss(y_true,y_pred)




class DQN():
    def __init__(self, K, scope, image_size):
        self.K = K
        self.scope = scope
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            self.is_training = tf.placeholder_with_default(False, (), 'is_training')
            self.X = tf.placeholder(tf.float32, shape=(None, image_size,image_size, 4), name='X')
            self.G = tf.placeholder(tf.float32, shape=(None,), name='G')
            self.actions = tf.placeholder(tf.int32, shape=(None,), name='actions')
            Z = self.X / 255.0
            Z = tf.layers.conv2d(Z, 32, [8,8], activation=tf.nn.relu)
            Z = tf.layers.max_pooling2d(Z,[2,2],2)
            Z = tf.layers.conv2d(Z, 64, [4,4], activation=tf.nn.relu)
            Z = tf.layers.max_pooling2d(Z,[2,2],2)
            Z = tf.layers.conv2d(Z, 64, [3,3], activation=tf.nn.relu)
            Z = tf.layers.max_pooling2d(Z,[2,2],2)
            Z = tf.contrib.layers.flatten(Z)
            Z = tf.layers.dense(Z, 512, activation=tf.nn.relu)

            self.predict_op = tf.layers.dense(Z,K, activation=tf.nn.relu)
            selected_action_value = tf.reduce_sum(self.predict_op * tf.one_hot(self.actions,K), reduction_indices=[1])
            cost = tf.reduce_mean(tf.losses.huber_loss(self.G, selected_action_value))
            self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            self.train_op = tf.train.AdamOptimizer(1e-6).minimize(cost)
            self.cost = cost
            
            
    def copy_from(self, other):
        mine = [t for t in tf.trainable_variables() if t.name.startswith(self.scope)]
        mine = sorted(mine, key=lambda v: v.name)
        theirs = [t for t in tf.trainable_variables() if t.name.startswith(other.scope)]
        theirs = sorted(theirs, key=lambda v: v.name)
        
        ops = []
        for p,q in zip(mine, theirs):
            op = p.assign(q)
            ops.append(op)
        self.session.run(ops)
    
    def save(self):
        params = [t for t in tf.trainable_variables() if t.name.startswith(self.scope)]
        params = self.session.run(params)
        np.savez('tf_dqn_weights.npz', *params)
    
    def load(self):
        params = [t for t in tf.trainable_variables() if t.name.startswith(self.scope)]
        npz = np.load('tf_dqn_weights.npz')
        ops = []
        for p, (_, v) in zip(params, npz.iteritems()):
            ops.append(p.assign(v))
        self.session.run(ops)
        
    def set_session(self,session):
        self.session = session
    
    def predict(self, states):
        return self.session.run(self.predict_op, feed_dict = {self.X: states, self.is_training: False})
    
    def update(self, states, actions, targets):
        c = self.session.run(
                [self.cost, self.train_op, self.update_ops],
                feed_dict = {self.X: states, self.G: targets, self.actions: actions, self.is_training: True}
                )[0]
        return c
    
    def sample_action(self,x,eps):
        if np.random.random() < eps:
            return np.random.choice(self.K)
        else:
            return np.argmax(self.predict([x])[0])





class DQN_prey():
    def __init__(self, K, scope, image_size1,image_size2,laser_size,laser_max , lase_min, n_history):
        self.K = K
        self.scope = scope
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            self.X1 = tf.placeholder(tf.float32, shape=(None, image_size1,image_size1, n_history), name='X1')
            self.X2 = tf.placeholder(tf.float32, shape=(None, image_size2,image_size2, n_history), name='X2')
            self.G = tf.placeholder(tf.float32, shape=(None,), name='G')
            self.actions = tf.placeholder(tf.int32, shape=(None,), name='actions')

            #Zlaser = (self.laser - lase_min)/(laser_max - lase_min)
            #Zlaser = tf.layers.conv1d(Zlaser, 64, 5, activation=tf.nn.relu)
            #Zlaser = tf.contrib.layers.flatten(Zlaser)
            #Zlaser = tf.layers.dense(Zlaser, 128, activation=tf.nn.relu)

            Z1 = self.X1 / 255.0
            Z1 = tf.layers.conv2d(Z1, 32, [8,8], activation=tf.nn.relu)
            Z1 = tf.layers.max_pooling2d(Z1,[4,4],2)
            Z1 = tf.layers.conv2d(Z1, 64, [4,4], activation=tf.nn.relu)
            Z1 = tf.layers.max_pooling2d(Z1,[2,2],2)
            Z1 = tf.layers.conv2d(Z1, 64, [3,3], activation=tf.nn.relu)
            Z1 = tf.contrib.layers.flatten(Z1)


            Z2 = self.X2 / 255.0
            Z2 = tf.layers.conv2d(Z2, 32, [8,8], activation=tf.nn.relu)
            Z2 = tf.layers.max_pooling2d(Z2,[4,4],2)
            Z2 = tf.layers.conv2d(Z2, 64, [4,4], activation=tf.nn.relu)
            Z2 = tf.layers.max_pooling2d(Z2,[2,2],2)
            Z2 = tf.layers.conv2d(Z2, 64, [3,3], activation=tf.nn.relu)
            Z2 = tf.contrib.layers.flatten(Z2)
            

            Z = tf.concat([Z1,Z2], axis = 1)
            Z = tf.layers.dense(Z, 512, activation=tf.nn.relu)
            self.predict_op = tf.layers.dense(Z,K, activation=tf.nn.relu)
            selected_action_value = tf.reduce_sum(self.predict_op * tf.one_hot(self.actions,K), reduction_indices=[1])
            
            cost = tf.reduce_mean(tf.losses.huber_loss(self.G, selected_action_value))
            self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            self.train_op = tf.train.AdamOptimizer(5e-6).minimize(cost)
            self.cost = cost
            
            
    def copy_from(self, other):
        mine = [t for t in tf.trainable_variables() if t.name.startswith(self.scope)]
        mine = sorted(mine, key=lambda v: v.name)
        theirs = [t for t in tf.trainable_variables() if t.name.startswith(other.scope)]
        theirs = sorted(theirs, key=lambda v: v.name)
        
        ops = []
        for p,q in zip(mine, theirs):
            op = p.assign(q)
            ops.append(op)
        self.session.run(ops)
    
    def save(self):
        params = [t for t in tf.trainable_variables() if t.name.startswith(self.scope)]
        params = self.session.run(params)
        np.savez('tf_dqn_weights.npz', *params)
    
    def load(self):
        params = [t for t in tf.trainable_variables() if t.name.startswith(self.scope)]
        npz = np.load('tf_dqn_weights.npz')
        ops = []
        for p, (_, v) in zip(params, npz.iteritems()):
            ops.append(p.assign(v))
        self.session.run(ops)
        
    def set_session(self,session):
        self.session = session
    
    def predict(self, states1, states2):
        return self.session.run(self.predict_op, feed_dict = {self.X1: states1, self.X2: states2})
    
    def update(self, states1, states2, actions, targets):
        c = self.session.run(
                [self.cost, self.train_op],
                feed_dict = {self.X1: states1, self.X2: states2, self.G: targets, self.actions: actions}
                )[0]
        return c
    
    def sample_action(self,states1, states2, eps):
        if np.random.random() < eps:
            return np.random.choice(self.K)
        else:
            return np.argmax(self.predict([states1], [states2])[0])



class DQN_predator():
    def __init__(self, K, scope, image_size, n_history):
        self.K = K
        self.scope = scope
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            self.X = tf.placeholder(tf.float32, shape=(None, image_size,image_size, n_history), name='X')
            self.G = tf.placeholder(tf.float32, shape=(None,), name='G')
            self.actions = tf.placeholder(tf.int32, shape=(None,), name='actions')

            Z = self.X / 255.0
            Z = tf.layers.conv2d(Z, 32, [8,8], activation=tf.nn.relu)
            Z = tf.layers.max_pooling2d(Z,[4,4],2)
            Z = tf.layers.conv2d(Z, 64, [4,4], activation=tf.nn.relu)
            Z = tf.layers.max_pooling2d(Z,[2,2],2)
            Z = tf.contrib.layers.flatten(Z)
            Z = tf.layers.dense(Z, 512, activation=tf.nn.relu)
            self.predict_op = tf.layers.dense(Z,K, activation=tf.nn.relu)
            selected_action_value = tf.reduce_sum(self.predict_op * tf.one_hot(self.actions,K), reduction_indices=[1])
            
            cost = tf.reduce_mean(tf.losses.huber_loss(self.G, selected_action_value))
            self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            self.train_op = tf.train.AdamOptimizer(5e-6).minimize(cost)
            self.cost = cost
            
            
    def copy_from(self, other):
        mine = [t for t in tf.trainable_variables() if t.name.startswith(self.scope)]
        mine = sorted(mine, key=lambda v: v.name)
        theirs = [t for t in tf.trainable_variables() if t.name.startswith(other.scope)]
        theirs = sorted(theirs, key=lambda v: v.name)
        
        ops = []
        for p,q in zip(mine, theirs):
            op = p.assign(q)
            ops.append(op)
        self.session.run(ops)
    
    def save(self):
        params = [t for t in tf.trainable_variables() if t.name.startswith(self.scope)]
        params = self.session.run(params)
        np.savez('tf_dqn_weights.npz', *params)
    
    def load(self):
        params = [t for t in tf.trainable_variables() if t.name.startswith(self.scope)]
        npz = np.load('tf_dqn_weights.npz')
        ops = []
        for p, (_, v) in zip(params, npz.iteritems()):
            ops.append(p.assign(v))
        self.session.run(ops)
        
    def set_session(self,session):
        K.set_session(session)
        self.session = session
    
    def predict(self, states):
        return self.session.run(self.predict_op, feed_dict = {self.X: states})
    
    def update(self, states, actions, targets):
        c = self.session.run(
                [self.cost, self.train_op],
                feed_dict = {self.X: states, self.G: targets, self.actions: actions}
                )[0]
        return c
    
    def sample_action(self,states, eps):
        if np.random.random() < eps:
            return np.random.choice(self.K)
        else:
            return np.argmax(self.predict([states])[0])



class DQN_object_disposer_robot():
    def __init__(self, K, scope, image_size, n_history):
        self.K = K
        self.scope = scope
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            self.X = tf.placeholder(tf.float32, shape=(None, image_size,image_size, n_history), name='X')
            self.G = tf.placeholder(tf.float32, shape=(None,), name='G')
            self.actions = tf.placeholder(tf.int32, shape=(None,), name='actions')

            Z = self.X / 255.0
            Z = tf.layers.conv2d(Z, 32, [8,8], activation=tf.nn.relu)
            Z = tf.layers.max_pooling2d(Z,[2,2],2)
            Z = tf.layers.conv2d(Z, 64, [4,4], activation=tf.nn.relu)
            Z = tf.layers.max_pooling2d(Z,[2,2],2)
            Z = tf.layers.conv2d(Z, 64, [3,3], activation=tf.nn.relu)
            Z = tf.layers.conv2d(Z, 128, [3,3], activation=tf.nn.relu)
            Z = tf.layers.max_pooling2d(Z,[2,2],2)
            Z = tf.contrib.layers.flatten(Z)
            Z = tf.layers.dense(Z, 512, activation=tf.nn.relu)
            self.predict_op = tf.layers.dense(Z,K, activation=tf.nn.relu)
            selected_action_value = tf.reduce_sum(self.predict_op * tf.one_hot(self.actions,K), reduction_indices=[1])
            
            cost = tf.reduce_mean(tf.losses.huber_loss(self.G, selected_action_value))
            self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            self.train_op = tf.train.AdamOptimizer(1e-6).minimize(cost)
            self.cost = cost
            
            
    def copy_from(self, other):
        mine = [t for t in tf.trainable_variables() if t.name.startswith(self.scope)]
        mine = sorted(mine, key=lambda v: v.name)
        theirs = [t for t in tf.trainable_variables() if t.name.startswith(other.scope)]
        theirs = sorted(theirs, key=lambda v: v.name)
        
        ops = []
        for p,q in zip(mine, theirs):
            op = p.assign(q)
            ops.append(op)
        self.session.run(ops)
    
    def save(self):
        params = [t for t in tf.trainable_variables() if t.name.startswith(self.scope)]
        params = self.session.run(params)
        np.savez('tf_dqn_weights.npz', *params)
    
    def load(self):
        params = [t for t in tf.trainable_variables() if t.name.startswith(self.scope)]
        npz = np.load('tf_dqn_weights.npz')
        ops = []
        for p, (_, v) in zip(params, npz.iteritems()):
            ops.append(p.assign(v))
        self.session.run(ops)
        
    def set_session(self,session):
        K.set_session(session)
        self.session = session
    
    def predict(self, states):
        return self.session.run(self.predict_op, feed_dict = {self.X: states})
    
    def update(self, states, actions, targets):
        c = self.session.run(
                [self.cost, self.train_op],
                feed_dict = {self.X: states, self.G: targets, self.actions: actions}
                )[0]
        return c
    
    def sample_action(self,states, eps):
        if np.random.random() < eps:
            return np.random.choice(self.K)
        else:
            return np.argmax(self.predict([states])[0])


class DQN_predator_keras():
    def __init__(self, K, scope, image_size, n_history):
        self.K = K
        self.scope = scope
        self.X = Input(shape=(None, image_size,image_size, n_history))
        self.G = Input(shape=(None,))
        self.actions = Input(shape=(None,))
        Z = Lambda(lambda x: x /255.0)(self.X)
        Z = Conv2D(filters = 32, kernel_size = (3,3), activation='relu')(Z)
        Z = Conv2D(filters = 32, kernel_size = (3,3), activation='relu')(Z)
        Z = MaxPooling2D(pool_size=(2, 2))(Z)
        Z = BatchNormalization()(Z)
        Z = Conv2D(filters = 64, kernel_size = (3,3), activation='relu')(Z)
        Z = Conv2D(filters = 64, kernel_size = (3,3), activation='relu')(Z)
        Z = MaxPooling2D(pool_size=(2, 2))(Z)
        Z = BatchNormalization()(Z)
        Z = Conv2D(filters = 64, kernel_size = (3,3), activation='relu')(Z)
        Z = Conv2D(filters = 64, kernel_size = (3,3), activation='relu')(Z)
        Z = GlobalMaxPooling2D()(Z)
        Z = Dense(512,activation='relu', kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l1(0.01))(Z)
        self.predict_op = Dense(K,activation='relu', kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l1(0.01))
        oh = Lambda(lambda x: K.one_hot(x,K))(self.actions)
        oh = Multiply()([oh,self.predict_op(Z)])
        ReduceSum = Lambda(lambda z: K.sum(z, axis=1))(oh)
        self.output_model = Model(inputs = self.X, outputs = self.predict_op(Z))
        self.training_model = Model(inputs = [self.X,self.actions] , outputs = ReduceSum)
        self.output_model.compile(optimizer='adam', loss=huber_loss, metrics=['accuracy'])
        self.training_model.compile(optimizer='adam', loss=huber_loss, metrics=['accuracy'])
            
            
    def copy_from(self, other):
        self.training_model.set_weights(other.get_weights()) 
    
    def save(self):
        params = [t for t in tf.trainable_variables() if t.name.startswith(self.scope)]
        params = self.session.run(params)
        np.savez('tf_dqn_weights.npz', *params)
    
    def load(self):
        params = [t for t in tf.trainable_variables() if t.name.startswith(self.scope)]
        npz = np.load('tf_dqn_weights.npz')
        ops = []
        for p, (_, v) in zip(params, npz.iteritems()):
            ops.append(p.assign(v))
        self.session.run(ops)
        
    def set_session(self,session):
        K.set_session(session)
    
    def predict(self, states):
        self.output_model.set_weights(self.training_model.get_weights())
        return self.output_model.predict(states)
    
    def update(self, states, actions, targets):
        self.training_model.fit([states,actions], targets)
    
    def sample_action(self,states, eps):
        if np.random.random() < eps:
            return np.random.choice(self.K)
        else:
            return np.argmax(self.predict([states])[0])