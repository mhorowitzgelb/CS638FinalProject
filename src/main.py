import pickle
import random
import pdb
import numpy as np
from training_set_generator import get_training_set, do_predictions
from metrics import get_average_dot_product
import tensorflow as tf
import GPflow as gp
import matplotlib.pyplot as plt
import sys


def nm(x_width, k):
    return 0.01*np.random.multivariate_normal(np.zeros(x_width), k)


def leaky_relu(z, name=None):
    return tf.maximum(0.01 * z, z, name=name)


def dropout_leaky_relu(x, name=None):
    return tf.nn.dropout(leaky_relu(x, name), 0.5)


def c1dwrap(inputs, num_filters, kernel_size, stride, padding,
            batch_norm=True, name=""):
    with tf.name_scope(name) as scope:
        if batch_norm:
            return tf.layers.batch_normalization(
                tf.layers.conv1d(inputs, num_filters, kernel_size, stride,
                                 padding, activation=dropout_leaky_relu), axis=2)
        else:
            return tf.layers.conv1d(inputs, num_filters, kernel_size, stride,
                                    padding, activation=dropout_leaky_relu)


def c1d_transpose_wrap(inputs, num_filters, kernel_size, stride, padding, name= ""):
    with tf.name_scope(name) as scope:
        return tf.layers.batch_normalization(tf.squeeze(tf.layers.conv2d_transpose(
            tf.expand_dims(inputs, 1), num_filters, kernel_size, (1, stride),
            padding=padding, activation=dropout_leaky_relu), 1))


def samplenoise(image_width, num_imgs):
    xx = np.arange(image_width)[:, None]
    kern = gp.kernels.RBF(1)
    noise = []
    K = kern.compute_K_symm(xx)
    for i in range(num_imgs):
        noise.append(np.transpose(np.array([nm(image_width, K),
                                            np.zeros(image_width)])))
    return noise


def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    with open('data/smooth.pkl', 'rb') as smooth_file:
        training_dataset = pickle.load(smooth_file)

    with open('data/ManualHannesDataset.pkl','rb') as noisy_file:
        testing_dataset = pickle.load(noisy_file)



    desired_image_width = 100
    # Gets a training set of {training_set_size}x2x100
    training_set = get_training_set(training_dataset, desired_image_width)
    training_set = [np.transpose(x) for x in training_set]
    # training_set = training_set[:100]

    # Channels should be 2
    number_of_channels = training_set[0].shape[1]

    # Should be same as desired_image_width
    image_width = training_set[0].shape[0]

    # example_one = training_set[0, :, :]
    # plt.clf()
    # # plt.plot(xx, noise)
    # plt.plot(np.arange(image_width), example_one)
    # # plt.plot(np.arange(image_width), example_one[1])
    # plt.plot(np.arange(image_width), example_one[0] + noise)
    # plt.plot(np.arange(image_width), example_one[1] + noise)
    # plt.show()

    # Get noisy test set: use this for testing your gans

    # with open('/home/gabe/Data/pickled638/hannes.pkl', 'rb') as noisy_file:
    #     noisy_dataset = pickle.load(noisy_file)
    #     noisy_test_set = get_training_set(noisy_dataset, desired_image_width)

    batch_size = 16

    with tf.name_scope('Generator') as scope:

        # should be the non-noisy image
        gen_input = tf.placeholder(tf.float32,
                                   name="gen_input",
                                   shape=[batch_size,
                                          image_width,
                                          number_of_channels])

        # the noise
        gen_noise = tf.placeholder(tf.float32,
                                   name="gen_noise",
                                   shape=[batch_size,
                                          image_width,
                                          number_of_channels])

        just_activation = tf.slice(gen_input, [0, 0, 0], [-1, -1, 1])
        # this means that there will be 0 padding
        padding = "SAME"

        # Generator layers
        #  convolution section
        #   layer1
        num_filters1 = 64
        kernel_size1 = 8
        stride1 = 2

        layer1 = c1dwrap(gen_input + gen_noise, num_filters1, kernel_size1,
                         stride1, padding, name="Conv1")

        #   layer2
        num_filters2 = 128
        kernel_size2 = 4
        stride2 = 2
        layer2 = c1dwrap(layer1, num_filters2, kernel_size2, stride2, padding, name="Conv2")

        #   layer3
        num_filters3 = 256
        kernel_size3 = 2
        stride3 = 1
        layer3 = c1dwrap(layer2, num_filters3, kernel_size3, stride3, padding, name="Conv3")

        #  inv-convolution section

        #   layer4
        num_filters4 = 256
        kernel_size4 = 2
        stride4 = 1
        layer4 = c1d_transpose_wrap(layer3, num_filters4, kernel_size4, stride4, padding, name="DeConv1")

        #   layer5
        num_filters5 = 128
        kernel_size5 = 4
        stride5 = 2
        layer5 = c1d_transpose_wrap(layer4, num_filters5, kernel_size5, stride5, padding, name="DeConv2")

        #   layer6
        num_filters6 = 64
        kernel_size6 = 8
        stride6 = 2
        layer6 = c1d_transpose_wrap(layer5, num_filters6, kernel_size6, stride6, padding, name="DeConv3")

        #   layer7
        layer7 = tf.layers.dense(layer6, units=1, activation=tf.tanh)

        arst = np.reshape(([1, 0] * batch_size) + ([0, 1] * batch_size),
                          [batch_size * 2, 2])



        print(layer7, just_activation)

    with tf.name_scope("Discriminator") as scope:
        disc_input = tf.concat([layer7, just_activation], 0, name="disc_input")
        disc_output = tf.constant(arst, tf.float32, name="disc_output")

        # Discriminator
        dnf1 = 512
        dks1 = 8
        ds1 = 2
        dl1 = c1dwrap(disc_input, dnf1, dks1, ds1, padding, name="Conv1")

        dnf2 = 256
        dks2 = 4
        ds2 = 2
        dl2 = c1dwrap(dl1, dnf2, dks2, ds2, padding, name="Conv2")

        dnf3 = 128
        dks3 = 2
        ds3 = 1
        dl3 = c1dwrap(dl2, dnf3, dks3, ds3, padding, name="Conv3")

        dnf4 = 64
        dks4 = 2
        ds4 = 1
        dl4 = c1dwrap(dl3, dnf4, dks4, ds4, padding, name="Conv4")

        shpe = dl4.get_shape()[1:]
        flat = tf.reshape(dl4, [-1, shpe.num_elements()])
        dl5 = tf.layers.dense(flat, units=256, activation=tf.sigmoid)
        dl6 = tf.layers.dense(dl5, units=2, activation=tf.sigmoid)

    # Loss
    lb = 10.0
    dloss = tf.losses.softmax_cross_entropy(disc_output, dl6)
    dl_print = tf.Print(dloss, [dloss], message="dloss: ")

    gloss = tf.losses.mean_squared_error(just_activation, layer7) + lb * tf.log(dl_print)
    gl_print = tf.Print(gloss, [gloss], message="gloss: ")

    # Optimizers
    goptimizer = tf.train.AdamOptimizer(epsilon=0.01)  # TODO set parameters
    doptimizer = tf.train.GradientDescentOptimizer(0.01)  # TODO set parameters

    dtrain = doptimizer.minimize(dloss)
    gtrain = goptimizer.minimize(gl_print)

    num_epochs = 1
    save_loc = "log/gan_model{0}.ckpt"

    print("starting")
    log = "epoch:{0}, error:{1}"

    tf.summary.scalar("Generator Loss", gloss)
    tf.summary.scalar("Discriminator Loss", dloss)

    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('log',
                                             sess.graph)

        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()
        batch = [random.choice(training_set) for _ in range(batch_size)]


        for i in range(num_epochs):
            noise = samplenoise(image_width, batch_size)
            feed_dict = {gen_input: batch, gen_noise: noise}
            sess.run(dtrain, feed_dict)
            sess.run(gtrain, feed_dict)
            train_writer.add_graph(sess.graph)
            if (i+1) % 50 == 0:
                (summary,) = sess.run([merged],feed_dict)
                train_writer.add_summary(summary,i)
                print("saving at " + save_loc.format((i // 100) % 10))
                saver.save(sess, save_loc.format((i // 100) % 10))
            if i == num_epochs -1:
                #Get predictions
                predictions = layer7.eval(feed_dict={gen_input: batch, gen_noise: noise},session=sess)

        #model = generator_model(layer7,gen_input,gen_noise)

        #do_predictions(testing_dataset,sess,model,image_width)

        #print("Average dot products")
        #print(get_average_dot_product(testing_dataset))

        train_writer.flush()
        #plot first example
        original = batch[0][:,0]
        pred = predictions[0][:,0]
        t= np.arange(100)

        plt.plot(t,original)
        plt.show()
        plt.plot(t,pred)
        plt.show()

class generator_model:
    def __init__(self, output, input, noise):
        self.output = output
        self.input = input
        self.noise = noise

    def predict(self, image, sess):
        image = np.transpose(image)
        input_arr = np.zeros((16,100,2))
        noise_arr = np.zeros((16,100,2))
        input_arr[0,:,:] = image
        return self.output.eval(feed_dict={(self.input): input_arr, (self.noise): (noise_arr)}, session=sess)[0,:,0]



if __name__ == "__main__":
    main()
