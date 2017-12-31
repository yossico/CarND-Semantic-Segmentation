import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import platform
import sys


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))



#1x1 convolution 
def conv_1x1(x, num_outputs, name="conv_1x1"):
    kernel_size = 1
    stride = 1
    initializer = tf.truncated_normal_initializer(stddev=0.01)
    conv_1x1_out = tf.layers.conv2d(x, num_outputs,
                        kernel_size=(1, 1),
                        strides=(1, 1),
                        padding='SAME',                       
                        kernel_initializer=initializer,
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01))
    return conv_1x1_out

# decoding/upsampling the previous layer
def decodelayer(x,filters,kernel_size,strides,name='decodelayer'):
     with tf.name_scope(name):
        initializer = tf.truncated_normal_initializer(stddev=0.01)
        decode_out = tf.layers.conv2d_transpose(x,filters,kernel_size, strides,padding = 'SAME', kernel_initializer = initializer)
        tf.summary.histogram(name, decode_out)
        return decode_out

def skiplayer(origin, destination, name="skip_Layer"):
    """
    connect encoder layers to decoder layers (skip connection)
    :origin: 4-Rank tensor
    :return: TF Operation
    """
    with tf.name_scope(name):
        skip = tf.add(origin,destination)
        return skip


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function - basic layers -done
    # Add skip layers
    # Use tf.saved_model.loader.load to load the model and weights    
    vgg_tag = 'vgg16'
    print(vgg_path)
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    graph = tf.get_default_graph()
    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return image_input, keep_prob, layer3_out, layer4_out, layer7_out
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The output layer
    """
    # TODO: Implement function
    conv7_1x1 = conv_1x1(vgg_layer7_out,num_classes)
    conv4_1x1 = conv_1x1(vgg_layer4_out,num_classes)
    conv3_1x1 = conv_1x1(vgg_layer3_out,num_classes)

    layer7dec = decodelayer(conv7_1x1,num_classes,5,2) 
    layer4skip = skiplayer(layer7dec,conv4_1x1)

    layer4dec = decodelayer(layer4skip,num_classes,5,2)
    layer3skip = skiplayer(layer4dec,conv3_1x1)

    model = decodelayer(layer3skip, num_classes,16,8)

    return model
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes, l2_const = 0.01):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    logits = tf.reshape(nn_last_layer, (-1,num_classes))
    labels = tf.reshape(correct_label,[-1,num_classes])
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = labels,logits = logits))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=0.9,beta2=0.999,epsilon=1e-08,use_locking=False,name='Adam')    
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = cross_entropy_loss + l2_const * sum(reg_losses)

    train_op = optimizer.minimize(loss=loss)

    return logits, train_op, cross_entropy_loss 
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, kprob, lrate, hparam):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    LOGDIR = '.'
    #hparam = 'mylog.txt'
    summ = tf.summary.merge_all()
    writer = tf.summary.FileWriter(hparam)
    writer.add_graph(sess.graph)
     
    counter = 1
    for epoch_i in range(epochs):

        for images, correct_labels in get_batches_fn(batch_size):

            if (counter % 10) == 0:

                # Run optimizer and get loss
                _, loss,s = sess.run([train_op, cross_entropy_loss,summ],
                                   feed_dict={input_image: images,
                                              correct_label: correct_labels,
                                              keep_prob: kprob,
                                              learning_rate: lrate})
                print("loss", loss, " epoch_i ", epoch_i, " epochs ", epochs)
                writer.add_summary(s, counter)

            else:
                _, loss = sess.run([train_op, cross_entropy_loss],
                                      feed_dict={input_image: images,
                                                 correct_label: correct_labels,
                                                 keep_prob: kprob,
                                                 learning_rate: lrate})
            print("loss", loss, " epoch_i ", epoch_i)
           
            counter += 1                

def make_hparam_string(kprob, lrate, l2_const):
    return "kp_%.0E,lr_%.0E,l2_%.0E" % (kprob, lrate, l2_const)

def run():
    num_classes = 2
    image_shape = (160, 576)
    

    print(pathname)
    myos = platform.system()
    if (myos == 'Windows'):
        data_dir = '.\data'
        runs_dir = '.\runs'
    else:
        data_dir = os.path.join(pathname,'/data')
        runs_dir = os.path.join(pathname,'/runs')

    l2_const = 0.002
    kprob = 0.9 
    lrate = 0.000075
    print(data_dir)
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)
        #model_path = "./model"
        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        epochs = 20
        batch_size = 20

         # hyperparameter search
        for l2_const in [0.002,0.005]:
            for lrate in [0.0001,0.00005]:
                for kprob in [0.8,0.9]:
                    tf.reset_default_graph()
                    with tf.Session() as sess:
                        hparam = make_hparam_string(kprob=kprob,lrate=lrate,l2_const=l2_const)
                        print('using param %s' % hparam)
                        model_path = LOGDIR + hparam + "/model"
                        print(model_path)
                        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
                        layer_output = layers(layer3_out, layer4_out, layer7_out,num_classes)
                        learning_rate = tf.placeholder(dtype = tf.float32)
                        correct_label = tf.placeholder(dtype = tf.float32, shape = (None, None, None, num_classes))

                        logits, train_op, cross_entropy_loss = optimize(layer_output, correct_label, learning_rate, num_classes, l2_const)
                         # 'Saver' op to save and restore all the variables
                        saver = tf.train.Saver()
                        # TODO: Train NN using the train_nn function

                        sess.run(tf.global_variables_initializer())
                        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
                                correct_label, keep_prob, learning_rate, kprob, lrate, hparam)

                        save_path = saver.save(sess, model_path)
                        print("Model saved in file: %s" % save_path)

                        # TODO: Save inference data using helper.save_inference_samples
                        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
                        
                        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':


    run()
