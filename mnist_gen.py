import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.datasets import mnist



img_width, img_height = 28, 28

class DiscriminatorNet(Model):

    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        n_features = img_width * img_height

        n_hidden1 = 128
        n_hidden2 = 256

        # only fully connected layers for now

        self.l1 = layers.Dense(n_hidden1, activation=tf.nn.relu)
        self.l2 = layers.Dense(n_hidden2, activation=tf.nn.relu)

        # output is either yes or no
        self.out = layers.Dense(2, activation=tf.nn.softmax)
    
    def forward_prop(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.out(x)

        return x
    
    def cross_entropy_loss(x, y):
        y = tf.cast(y, tf.int64)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=x)

        return tf.reduce_mean(loss)
    
    def train(batch_x, batch_y):


discrim_net = DiscriminatorNet()