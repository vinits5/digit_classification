import tensorflow as tf 


class network_structure():
	# Use it to initialize weights.
	def weights(self,x,y):
		weights_dict = {'weights':tf.Variable(tf.random_normal([x,y])),'biases':tf.Variable(tf.random_normal([y]))}
		return weights_dict

	# Define the complete neural network here.
	def structure(self):
		self.x = tf.placeholder(tf.float32,shape=(None,784))
		self.x1 = tf.reshape(self.x,[-1,28,28,1])
		self.y = tf.placeholder(tf.int32,shape=None)
		self.is_training = tf.placeholder(tf.bool)

		self.batch_size = tf.placeholder(tf.float32,shape=None)

		self.conv1 = tf.layers.conv2d(inputs=self.x1,filters=32,kernel_size=2,strides=2,padding="VALID")
		self.conv2 = tf.layers.conv2d(inputs=self.conv1,filters=32,kernel_size=2,strides=2,padding="VALID")
		self.conv3 = tf.layers.conv2d(inputs=self.conv2,filters=32,kernel_size=3,strides=1,padding="VALID")

		self.conv3 = tf.contrib.layers.flatten(self.conv3)

		self.l1 = tf.contrib.layers.fully_connected(self.x,256)
		self.l1 = tf.contrib.layers.dropout(self.l1,is_training=self.is_training)

		self.output = tf.contrib.layers.fully_connected(self.l1,10,activation_fn=None)
		
		self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.output,labels=self.y)
		self.loss = tf.reduce_mean(self.loss)
		self.trainer = tf.train.AdamOptimizer(learning_rate = 0.0001)
		self.updateModel = self.trainer.minimize(self.loss)