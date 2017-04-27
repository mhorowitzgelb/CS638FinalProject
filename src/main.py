import pickle
import numpy as np
from training_set_generator import get_training_set
import matplotlib.pyplot as plt
import sys
import tensorflow as tf


def main():
    with open('/home/gabe/Data/pickled638/smooth.pkl','rb') as smooth_file:
        training_dataset = pickle.load(smooth_file)


    desired_image_width = 100
    #Gets a training set of {training_set_size}x2x100
    training_set = get_training_set(training_dataset,desired_image_width)

    #number of training exmamples
    num_training_examples = training_set.shape[0]

    #Channels should be 2
    number_of_channels = training_set.shape[1]

    #Should be same as desired_image_width
    image_width = training_set.shape[2]
    print(image_width)


    #Get noisy test set: use this for testing your gans

    with open('/home/gabe/Data/pickled638/hannes.pkl','rb') as noisy_file:
        noisy_dataset = pickle.load(noisy_file)
        noisy_test_set = get_training_set(noisy_dataset,desired_image_width)

    example_one = training_set[0,:,:]

    #plot one unique transition
    #x is values 0 ... 99 , normalized time values
    #y is the transitions intensities at each time point
    plt.plot(np.arange(image_width), example_one[0,:], label='single transition')

    #Plot average shape over all other transitions of the peptide
    plt.plot(np.arange(image_width), example_one[1,:], label='shape average')

    plt.legend()

    plt.show(block=False)

def real_main():
    g = GAN(id)

    num_eps = 1000
    for i in range(num_eps):
        # train here

        if i % 20 == 0:
            g.display_model()
            g.save_model()

BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EPOCHS = 75
Z_DIM = 100

class GAN(object):
	def __init__(self, lr=1E-3, batch_size=128, z_dim=100):
		self.lr = lr
		self.batch_size = batch_size
		self.z_dim = z_dim

		self.z_in = tf.placeholder(tf.float32, shape=[None, z_dim])
		self.x_in = tf.placeholder(tf.float32, shape=[None, 784])

		# Construct Discriminator/ Generator graph ops
		self.g_sample, self.g_weights = self.generator(z=self.z_in)
		self.d_real, self.d_weights = self.discriminator(self.x_in)
		self.d_fake, _ = self.discriminator(self.g_sample, reuse=True)

		# Loss and optimization ops
		self.d_loss, self.g_loss = self.loss()
		self.d_train, self.g_train = self.optimize()

		# Initialize session to run ops in
		self._sess = tf.Session()
		self._sess.run(tf.initialize_all_variables())

	def discriminator(self, x, reuse=False):
		with tf.variable_scope("D", reuse=reuse):
			W1_init, _ = he_xavier(784, 128, init_only=True)
			W1 = tf.get_variable("W1", initializer=W1_init)
			b1 = tf.get_variable("b1", shape=[128], initializer=tf.constant_initializer(0.0))
			d_h1 = tf.nn.elu(tf.add(tf.matmul(x, W1), b1))

			W2_init, _ = he_xavier(128, 1, init_only=True)
			W2 = tf.get_variable("W2", initializer=W2_init)
			b2 = tf.get_variable("b2", shape=[1], initializer=tf.constant_initializer(0.0))
			d_h2 = tf.add(tf.matmul(d_h1, W2), b2)

			return tf.nn.sigmoid(d_h2), [W1,b1,W2,b2]

	def generator(self, z):
		W1, b1 = he_xavier(self.z_dim, 128)
		g_h1 = tf.nn.elu(tf.add(tf.matmul(z, W1), b1))

		W2, b2 = he_xavier(128, 784)
		g_h2 = tf.add(tf.matmul(g_h1, W2), b2)

		return tf.nn.sigmoid(g_h2), [W1,b1,W2,b2]

	def loss(self):
		discriminator_loss = -tf.reduce_mean(tf.log(self.d_real) + tf.log(1. - self.d_fake))
		generator_loss = -tf.reduce_mean(tf.log(self.d_fake))
		return discriminator_loss, generator_loss

	def optimize(self):
		optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
		d_train = optimizer.minimize(self.d_loss, var_list=self.d_weights)
		g_train = optimizer.minimize(self.g_loss, var_list=self.g_weights)
		return d_train, g_train

	def sample_z(self, num_samples):
		return np.random.uniform(-1., 1., size=[num_samples, self.z_dim])

	def train_discriminator(self, x_in):
		z_sample = self.sample_z(self.batch_size)
		fetches = [self.d_train, self.d_loss]
		_, d_loss = self._sess.run(fetches, feed_dict={self.x_in: x_in, self.z_in:z_sample})
		return d_loss

	def train_generator(self):
		z_sample = self.sample_z(self.batch_size)
		fetches = [self.g_train, self.g_loss]
		_, g_loss = self._sess.run(fetches, feed_dict={self.z_in: z_sample})
		return g_loss

	def sample_g(self, num_samples):
		z_sample = self.sample_z(num_samples=num_samples)
		return self._sess.run(self.g_sample, feed_dict={self.z_in: z_sample})

def plot_grid(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

def main(_):
	print_flags(FLAGS)
	gan = GAN(lr=FLAGS.learning_rate, batch_size=FLAGS.batch_size, z_dim=FLAGS.z_dim)
	mnist = load_mnist()

	print("Started training {}".format(datetime.now().isoformat()[11:]))
	plot_index = 0
	for epoch in range(FLAGS.epochs):
		for batch in range(mnist.train.num_examples // FLAGS.batch_size):
			batch_x, _ = mnist.train.next_batch(FLAGS.batch_size)
			_ = gan.train_discriminator(x_in=batch_x)
			_ = gan.train_generator()

		if epoch % 10 == 0:
			d_loss = gan.train_discriminator(x_in=batch_x)
			g_loss = gan.train_generator()
			print("Epoch {} Discriminator Loss {} Generator loss {}".format(epoch + 1,
			                                                                d_loss,
			                                                                g_loss))
			gen_sample = gan.sample_g(num_samples=16)

			fig = plot_grid(gen_sample)
			plt.savefig('{}.png'.format(str(plot_index).zfill(3)), bbox_inches='tight')
			plot_index += 1
			plt.close(fig)
	print("Ended training {}".format(datetime.now().isoformat()[11:]))

FLAGS = None
if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument("--learning_rate", type=int, default=LEARNING_RATE)
	parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
	parser.add_argument("--epochs", type=int, default=EPOCHS)
	parser.add_argument("--z_dim", type=int, default=Z_DIM)
	FLAGS, _ = parser.parse_known_args()

	tf.app.run()

def GAN():
    """
    noise_model(inputs) = sample from the noise distribution
    """
    def __init__(self, noise_model, learning_rate=1E-4, batch_size=16, image_width=100, save_loc="/tmp/gan_model.ckpt"):
        self.noise_model = noise_model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.image_width = image_width

        #where the images get put
        self.img_in = tf.placeholder(tf.float32, shape=[batch_size, image_width])

        # g_sample is the output of sampling from g
        # g_weights is the weights of g
        self.g_sample, self.g_weights = self.init_generator()
        self.d_weights = self.init_discriminator()
        

        self._sess = tf.Session()
        self._sess.run(tf.initialize_all_variables)
        self._saver = tf.train.Saver()
        self.save_loc = save_loc

    def init_generator(self):
        pass

    def init_discriminator(self):
        pass

    def train_generator(self, x_in):
        assert(len(x_in) == self.batch_size)
        inputs = self.noise_model(x_in)

    def train_discriminator(self, x_in):
        assert(len(x_in) == self.batch_size)

    def save_model(self):
        self._saver.save(self._sess, self.save_loc)

    def display_model(self):
        pass
