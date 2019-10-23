from train import Network
import numpy as np
import tensorflow as tf
import pprint
import os
import warnings
warnings.filterwarnings("ignore")

flags = tf.app.flags
flags.DEFINE_boolean('cpu', False, "Only use CPU to train, default use GPU")
flags.DEFINE_integer('epoch', 5000, "Number of epoch")
flags.DEFINE_integer('batch_size', 48, "The size of batch images")
flags.DEFINE_integer('image_size', 208, "The size of input image")
flags.DEFINE_float('learning_rate', 1e-4, "The learning rate of gradient descent algorithm")
flags.DEFINE_integer('c_dim', 3, "Dimension of image color. [1]")
flags.DEFINE_string('checkpoint_dir', 'checkpoint', "Name of checkpoint directory")
flags.DEFINE_integer('n_classes', 12, "number of output classes")
flags.DEFINE_string('data_dir', '/gpfs/home/stu16/shaw/Tell_Cat/data/', "training dataset")
flags.DEFINE_boolean('reset', False, "reset the training data")
flags.DEFINE_string('model','mynet',"model name")
flags.DEFINE_boolean('restore', False, "default without restore model")
flags.DEFINE_string('restore_model','default',"default load the recent model")
FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()

def main(_):
  #pp.pprint(flags.FLAGS.__flags)
  # if not os.path.exists(FLAGS.checkpoint_dir):
  #   os.makedirs(FLAGS.checkpoint_dir)
  # if not os.path.exists(FLAGS.sample_dir):
  #   os.makedirs(FLAGS.sample_dir)
  if FLAGS.cpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

  with tf.Session() as sess:
      cnn = Network(sess,
                    model=FLAGS.model,
                    image_size=FLAGS.image_size,
                    n_classes=FLAGS.n_classes,
                    batch_size=FLAGS.batch_size,
                    c_dim=FLAGS.c_dim)
      cnn.train(sess,FLAGS)
    
if __name__ == '__main__':
  tf.app.run()
