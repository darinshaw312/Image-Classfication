import utils
#import imageio
#import scipy.misc
#import scipy.ndimage
#import matplotlib.pyplot as plt
#import PIL
import skimage
import imageio
import cv2
import time
import os
import numpy as np
import tensorflow as tf
import importlib
from datetime import datetime
from operator import methodcaller
#from model.mynet import mynet

class Network(object):
  def __init__(self,
               sess,
               model,
               image_size=208,
               n_classes=2,
               batch_size=32,
               c_dim=3):

    self.sess = sess
    self.model = model
    self.image_size = image_size
    self.n_classes = n_classes
    self.batch_size = batch_size
    self.c_dim = c_dim
    #self.checkpoint_dir = checkpoint_dir
    self.build_model()

  def build_model(self):
    self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images')
    self.labels = tf.placeholder(tf.int32, [None], name='labels')
    
    print('*****************************************************')
    print('Model:      ',self.model)
    print('Input:      ',self.image_size)
    print('Output:     ',self.n_classes)
    print('Batch_size: ',self.batch_size)
    print('*****************************************************')
    model_package = '.'.join(['model',self.model])
    model = importlib.import_module(model_package)
    #self.loss = tf.reduce_mean(tf.square(self.labels - self.pred))    self.pred = mynet(self.images, self.labels ,self.batch_size,self.image_size,self.c_dim,self.n_classes).model()
    #methodcaller('mynet', par1,par2,...)(model.mynet) = model.mynet.mynet(par1,par2,par3,...)
    self.pred = methodcaller(self.model,self.images,self.batch_size,self.image_size,self.c_dim,self.n_classes)(model).model()
    self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.pred, labels=self.labels))
    correct = tf.nn.in_top_k(self.pred, self.labels, 1)
    self.correct = tf.cast(correct, tf.float16)
    self.acc = tf.reduce_mean(self.correct)
    self.saver = tf.train.Saver()

    self.loss_summary = tf.summary.scalar('Train/Loss', self.loss)
    self.acc_summary = tf.summary.scalar('Train/Acc', self.acc)
    self.merge = tf.summary.merge([self.loss_summary,self.acc_summary])

  def train(self, sess, config):
    savepath = os.path.join(os.getcwd(), 'checkpoint/train.h5')
    if config.reset:
      data_dir = os.path.join(config.data_dir,'cat_12_train')
      data_list = os.path.join(config.data_dir,'train_list.txt')
      data=[]
      label=[]
      cnt=1
      with open(data_list,'r') as f:
        for fn in f:
          l=fn.split()
          label.append(int(l[1]))
          #image = cv2.imread(config.data_dir + l[0])
          #image = cv2.resize(image,(self.image_size,self.image_size),interpolation = cv2.INTER_AREA)
          image = imageio.imread(config.data_dir + l[0])
          print(config.data_dir + l[0],image.shape)

          if len(image.shape) < 3:        #gray_image
            image = np.expand_dims(image,2).repeat(3,axis=2)
              
          image = skimage.transform.resize(image[:,:,:3],(self.image_size,self.image_size))
          data.append(np.asarray(image,dtype=np.float16))


      print(type(data),type(data[0]),data[0].shape)
      temp = np.array([data,label])
      temp = temp.transpose()
      np.random.shuffle(temp)
      data=list(temp[:,0])
      label=list(temp[:,1])
      print(type(data),type(data[0]),data[0].shape)
      print('MAKE train.h5 ...')
      ts=time.time()
      utils.make_data(savepath,data,label)
      te=time.time()
      print('MAKE SUCCESS!\nTime: %.2f' %(te-ts))

    print('READ train.h5 ...')
    ts1=time.time()
    train_data, train_label = utils.read_data(savepath)
    te1=time.time()
    print('READ SUCCESS!\nTime: %.2f' %(te1-ts1))

    # Stochastic gradient descent with the standard backpropagation
    self.train_op = tf.train.GradientDescentOptimizer(config.learning_rate).minimize(self.loss)
    tf.global_variables_initializer().run()

    # coord = tf.train.Coordinator()
    # threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    #if self.load(self.checkpoint_dir):
    #  print(" [*] Load SUCCESS")
    #else:
    #  print(" [!] Load failed...")
    #if config.restore:
      #pre_model = tf.train.import_meta_graph('my-model-1000.meta')
    #loading restore model
    if config.restore:
       self.load(config.checkpoint_dir, config.restore_model)
    
    counter = 0
    print("Training ...")
    summary_dir = os.path.join('logs',datetime.now().strftime("%b%d_%H:%M:%S"))
    writer = tf.summary.FileWriter(summary_dir, sess.graph)
    for ep in range(config.epoch):
      it = len(train_data) // config.batch_size    #2160/48=45
      print("Epoch: [%5d/%d]    Learning_rate: %.e" %(ep+1,config.epoch,config.learning_rate))
      start_time = time.time()
      for idx in range(0, it):
        batch_images = train_data[idx*config.batch_size : (idx+1)*config.batch_size]
        batch_labels = train_label[idx*config.batch_size : (idx+1)*config.batch_size]
        summary, _, err, acc = self.sess.run([self.merge, self.train_op, self.loss, self.acc], feed_dict={self.images: batch_images, self.labels: batch_labels})
        counter += 1
        if (idx+1) % 9 == 0:
          print("[%3d/%d]  Loss: [%2.4f]  Acc: [%3.2f]  Time: [%.2f]  " \
            %(idx+1, it, err, acc*100, time.time()-start_time))
        if (idx+1) % it == 0:
          self.save(config.checkpoint_dir, ep+1)
      writer.add_summary(summary, ep+1)


    # else:
    #   print("Testing...")

    #   result = self.pred.eval({self.images: train_data, self.labels: train_label})

    #   result = merge(result, [nx, ny])
    #   result = result.squeeze()
    #   image_path = os.path.join(os.getcwd(), config.sample_dir)
    #   image_path = os.path.join(image_path, "test_image.png")
    #   imsave(result, image_path)

  def save(self, checkpoint_dir, step):
    model_name = self.model
    #model_dir = "%s_%s" % ("mynet", self.image_size)
    checkpoint_dir = os.path.join(checkpoint_dir, model_name)
    #if not os.path.exists(checkpoint_dir):
    #    os.makedirs(checkpoint_dir)
    #if step == 1:    #Only the first epoch we save the meta file
    #  self.saver.save(self.sess,
    #                  os.path.join(checkpoint_dir, model_name),
    #                  global_step=step)
    #else:
    #  self.saver.save(self.sess,
    #                  os.path.join(checkpoint_dir, model_name),
    #                  global_step=step,
    #                  write_meta_graph=False)
    self.saver.save(self.sess,
                os.path.join(checkpoint_dir, model_name),
                global_step=step)
    #print('SAVE SUCCESS!')

  def load(self, checkpoint_dir, restore_model):
    print("Loading ...")
    model_name = self.model
    #model_dir = "%s_%s" % ("srcnn", self.label_size)
    checkpoint_dir = os.path.join(checkpoint_dir, model_name)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if restore_model == 'default':
        if ckpt and ckpt.model_checkpoint_path:     #find recent model file
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)  #model_checkpoint_path can get the recent file name
            #saver = tf.train.import_meta_graph('checkpoint/mynet/mynet-1.meta')
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            #self.saver.restore(self.sess, mynet-150)
            print('Load [%s] SUCESS!' % ckpt_name)
            return True
        else:
            return False
    else:
        self.saver.restore(self.sess, os.path.join(checkpoint_dir, restore_model))
        print('Load [%s] SUCESS!' % restore_model)
        return True
