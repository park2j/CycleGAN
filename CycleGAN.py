import tensorflow as tf
import numpy as np
import os
from PIL import Image
from PIL import ImageOps


def residual_block(x, output_dim, kernel_size=3, stride=1, name='res'):
    x1 = tf.layers.conv2d(x, output_dim, [kernel_size, kernel_size], strides=(stride, stride), padding='same',
                          activation=tf.nn.relu, kernel_regularizer=tf.layers.batch_normalization, name=name+'_1')
    x2 = tf.layers.conv2d(x1, output_dim, [kernel_size, kernel_size], strides=(stride, stride), padding='same',
                          kernel_regularizer=tf.layers.batch_normalization, name=name+'_2')
    return x + x2


# generate image
def generator(inp, name='G', reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        conv1 = tf.layers.conv2d(inp, 64, [7, 7], strides=(1, 1), padding='same',
                                 activation=tf.nn.relu, kernel_regularizer=tf.layers.batch_normalization)
        conv2 = tf.layers.conv2d(conv1, 128, [3, 3], strides=(2, 2), padding='same',
                                 activation=tf.nn.relu, kernel_regularizer=tf.layers.batch_normalization)
        conv3 = tf.layers.conv2d(conv2, 256, [3, 3], strides=(2, 2), padding='same',
                                 activation=tf.nn.relu, kernel_regularizer=tf.layers.batch_normalization)
        res1 = residual_block(conv3, 256, name='res1')
        res2 = residual_block(res1, 256, name='res2')
        res3 = residual_block(res2, 256, name='res3')
        res4 = residual_block(res3, 256, name='res4')
        res5 = residual_block(res4, 256, name='res5')
        res6 = residual_block(res5, 256, name='res6')
        res7 = residual_block(res6, 256, name='res7')
        res8 = residual_block(res7, 256, name='res8')
        res9 = residual_block(res8, 256, name='res9')
        dconv1 = tf.layers.conv2d_transpose(res9, 128, [3, 3], strides=(2, 2), padding='same',
                                            activation=tf.nn.relu, kernel_regularizer=tf.layers.batch_normalization)
        dconv2 = tf.layers.conv2d_transpose(dconv1, 64, [3, 3], strides=(2, 2), padding='same',
                                            activation=tf.nn.relu, kernel_regularizer=tf.layers.batch_normalization)
        conv_f = tf.layers.conv2d(dconv2, 3, [7, 7], strides=(1, 1), padding='same')
        o = tf.nn.sigmoid(conv_f)
    return o


# discriminate image
def discriminator(inp, name='D', reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        conv1 = tf.layers.conv2d(inp, 64, [4, 4], strides=(2, 2), padding='same',
                                 activation=tf.nn.leaky_relu, kernel_regularizer=tf.layers.batch_normalization)
        conv2 = tf.layers.conv2d(conv1, 128, [4, 4], strides=(2, 2), padding='same',
                                 activation=tf.nn.leaky_relu, kernel_regularizer=tf.layers.batch_normalization)
        conv3 = tf.layers.conv2d(conv2, 256, [4, 4], strides=(2, 2), padding='same',
                                 activation=tf.nn.leaky_relu, kernel_regularizer=tf.layers.batch_normalization)
        conv4 = tf.layers.conv2d(conv3, 512, [4, 4], strides=(2, 2), padding='same',
                                 activation=tf.nn.leaky_relu, kernel_regularizer=tf.layers.batch_normalization)
        conv5 = tf.layers.conv2d(conv4, 1, [4, 4], strides=(2, 2), padding='same')
        o = tf.nn.sigmoid(conv5)
    return o


# load data from directory
def load_data(img_dir, train=False):
    imgs_name = os.listdir(img_dir)
    imgs = np.zeros([len(imgs_name), 256, 256, 3], 'float16')
    for i in range(len(imgs_name)):
        img = Image.open(os.path.join(img_dir, imgs_name[i]))
        # augment image for training
        if train:
            # flip
            if np.random.rand() > 0.5:
                img = ImageOps.mirror(img)
            # crop
            img = img.resize((256 + 32, 256 + 32))
            x = np.random.randint(0, 32)
            y = np.random.randint(0, 32)
            img = img.crop((x, y, x + 256, y + 256))
        img = np.array(img)
        if len(img.shape) == 2:
            img = np.reshape(np.tile(img, 3), [256, 256, 3])
        imgs[i] = img
    imgs = imgs / 255
    return imgs


def save_img(img_np, p_samples, copy_np, name):
    # make folder
    if not os.path.exists('s2w/'):
        os.makedirs('s2w/')
    # save image
    p_samples = p_samples * 255
    p_samples = p_samples.astype('uint8')
    img_np = img_np * 255
    img_np = img_np.astype('uint8')
    copy_np = copy_np * 255
    copy_np = copy_np.astype('uint8')
    im = Image.fromarray(np.concatenate([img_np[0], p_samples[0], copy_np[0],
                                         img_np[1], p_samples[1], copy_np[1],
                                         img_np[2], p_samples[2], copy_np[2],
                                         img_np[3], p_samples[3], copy_np[3],
                                         img_np[4], p_samples[4], copy_np[4]], 1))
    im.save("s2w/" + name + ".jpg")
    return

def main():

    # input placeholder
    real_a = tf.placeholder(tf.float32, [None, 256, 256, 3])
    real_b = tf.placeholder(tf.float32, [None, 256, 256, 3])

    # generate image
    fake_b = generator(real_a, name='Ga2b')
    fake_a = generator(real_b, name='Gb2a')
    fake_a2 = generator(fake_b, name='Gb2a', reuse=True)
    fake_b2 = generator(fake_a, name='Ga2b', reuse=True)
    iden_a = generator(real_a, name='Gb2a', reuse=True)
    iden_b = generator(real_b, name='Ga2b', reuse=True)

    # discriminate real and fake
    d_real_a = discriminator(real_a, name='Da')
    d_real_b = discriminator(real_b, name='Db')
    d_fake_a = discriminator(fake_a, name='Da', reuse=True)
    d_fake_b = discriminator(fake_b, name='Db', reuse=True)

    # loss for input image and generated image
    d_loss_a = tf.reduce_mean(d_real_a**2 + (1 - d_fake_a)**2)
    d_loss_b = tf.reduce_mean(d_real_b**2 + (1 - d_fake_b)**2)
    d_loss = d_loss_a + d_loss_b
    g_loss_a = tf.reduce_mean(d_fake_a**2) + tf.reduce_mean(tf.abs(real_a - fake_a2))
    g_loss_b = tf.reduce_mean(d_fake_b**2) + tf.reduce_mean(tf.abs(real_b - fake_b2))
    g_loss = g_loss_a + g_loss_b

    # variables for discriminator and generator
    t_vars = tf.trainable_variables()
    d_var = [var for var in t_vars if 'D' in var.name]
    g_var = [var for var in t_vars if 'G' in var.name]

    # optimize discriminator and generator
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        d_optim = tf.train.AdamOptimizer(2e-5, 0.5).minimize(d_loss, var_list=d_var)
        g_optim = tf.train.AdamOptimizer(2e-5, 0.5).minimize(g_loss, var_list=g_var)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        init.run()
        # load data
        testA = load_data('summer2winter/testA')
        testB = load_data('summer2winter/testB')

        # training network
        for step in range(1000):
            batch_size = 1
            trainA = load_data('summer2winter/trainA', train=True)
            trainB = load_data('summer2winter/trainB', train=True)
            # feed training data
            for ite in range(trainB.shape[0] // batch_size):
                data_a_np = trainA[ite*batch_size:(ite+1)*batch_size]
                data_b_np = trainB[ite*batch_size:(ite+1)*batch_size]
                _, d_loss_pa, d_loss_pb = sess.run([d_optim, d_loss_a, d_loss_b], feed_dict={real_a: data_a_np, real_b: data_b_np})
                _, g_loss_pa, g_loss_pb = sess.run([g_optim, g_loss_a, g_loss_b], feed_dict={real_a: data_a_np, real_b: data_b_np})
                print(str(step) + '/' + str(ite))
                print('d_loss: ' + str(d_loss_pa), str(d_loss_pb))
                print('g_loss: ' + str(g_loss_pa), str(g_loss_pb))

            # generate image
            batch_size = 5
            data_a_np = testA[0:batch_size]
            data_b_np = testB[0:batch_size]
            a_samples, b_samples, a_copy, b_copy = sess.run([fake_a, fake_b, fake_a2, fake_b2], feed_dict={real_a: data_a_np, real_b: data_b_np})
            save_img(data_a_np, b_samples, a_copy, str(int(step + 1)) + '_a')
            save_img(data_b_np, a_samples, b_copy, str(int(step + 1)) + '_b')


if __name__ == '__main__':
    main()
