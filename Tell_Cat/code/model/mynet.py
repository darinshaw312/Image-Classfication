import tensorflow as tf

class mynet(object):

  def __init__(self, images, batch_size=32, image_size=208, c_dim=3, n_classes=2):
      self.batch_size = batch_size
      self.image_size = image_size
      self.c_dim = c_dim
      self.n_classes = n_classes
      self.images = images
    
      self.weights = {
        'w1': tf.Variable(tf.random_normal([3, 3, 3, 16], stddev=1e-1), name='w1'), #[k_size,k_size,in_channel,out_channel]
        'w2': tf.Variable(tf.random_normal([3, 3, 16, 16], stddev=1e-1), name='w2'),
        'w3': tf.Variable(tf.random_normal([self.image_size**2,128], stddev=5e-3), name='w3'),
        'w4': tf.Variable(tf.random_normal([128,128], stddev=5e-3), name='w4'),
        'w5': tf.Variable(tf.random_normal([128,self.n_classes], stddev=5e-3), name='w5')
      }

      self.biases = {
        'b1': tf.Variable(tf.zeros([16]), name='b1'),
        'b2': tf.Variable(tf.zeros([16]), name='b2'),
        'b3': tf.Variable(tf.zeros([128]), name='b3'),
        'b4': tf.Variable(tf.zeros([128]), name='b4'),
        'b5': tf.Variable(tf.zeros([self.n_classes]), name='b5')
      }

  def model(self):
      conv1 = tf.nn.relu(tf.nn.conv2d(self.images, self.weights['w1'], strides=[1,1,1,1], padding='SAME') + self.biases['b1'])
      pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
      norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

      conv2 = tf.nn.relu(tf.nn.conv2d(norm1, self.weights['w2'], strides=[1,1,1,1], padding='SAME') + self.biases['b2'])
      pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
      norm2 = tf.nn.lrn(pool2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      norm2 = tf.reshape(norm2, shape=[self.batch_size, -1])
      
      fc1   = tf.nn.relu(tf.matmul(norm2, self.weights['w3']) + self.biases['b3'])
      fc2   = tf.nn.relu(tf.matmul(fc1, self.weights['w4']) + self.biases['b4'])
      softmax = tf.matmul(fc1, self.weights['w5']) + self.biases['b5']
      
      return softmax
