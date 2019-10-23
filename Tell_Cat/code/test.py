#coding=utf-8  
import tensorflow as tf 
#import cv2 as cv  
#import input_data 
import numpy as np
import model
import os
import time
from operator import methodcaller
import importlib
import csv
import imageio
import skimage
import utils
#从训练集中选取一张图片 

def test():
    BATCH_SIZE = 40
    N_CLASSES = 12
    IMAGE_SIZE = 208
    C_DIMS = 3
    data_dir = '/gpfs/home/stu16/shaw/Tell_Cat/data/cat_12_test'
    savepath = os.path.join(os.getcwd(), 'checkpoint/test.h5')
    csv_file = 'result.csv'
    data=[]
    label=[]
    reset = False
    if reset:
        with open(csv_file,newline='',encoding='UTF-8') as cf:
            rows=csv.reader(cf)
            for r in rows:
                print(r)
                image_name = r[0]
                image = imageio.imread(os.path.join(data_dir,image_name))
                if len(image.shape) < 3:
                    image = np.expand_dims(image,2).repeat(3,axis=2)

                image = skimage.transform.resize(image[:,:,:3],(IMAGE_SIZE,IMAGE_SIZE))
                data.append(np.asarray(image,dtype=np.float16))
                label.append(int(r[1]))

        temp = np.array([data,label])
        temp = temp.transpose()
        np.random.shuffle(temp)
        data=list(temp[:,0])
        label=list(temp[:,1])
        print(type(data),type(data[0]),data[0].shape)
        print('MAKE test.h5 ...')
        ts=time.time()
        utils.make_data(savepath,data,label)
        te=time.time()
        print('MAKE SUCCESS!\nTime: %.2f' %(te-ts))

    # test_image = cv.imread('/gpfs/home/stu16/shaw/new/Test/5.jpg')
    # test_image = cv.resize(test_image,(IMAGE_SIZE,IMAGE_SIZE))
    # cv.imwrite('/gpfs/home/stu16/shaw/new/sample/5.jpg',test_image)
    # #test_image = tf.expand_dims(test_image,0)
    # print(type(test_image))
    # test_image = np.asarray(test_image,dtype=np.float32)
    # test_image = np.expand_dims(test_image,0)
    # print(type(test_image),test_image.shape)

    print('READ test.h5 ...')
    ts1=time.time()
    test_data, test_label = utils.read_data(savepath)
    te1=time.time()
    print('READ SUCCESS!\nTime: %.2f' %(te1-ts1))

    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 208, 208, 3])

    model_package = '.'.join(['model','mynet'])
    model = importlib.import_module(model_package)
    pred = methodcaller('mynet',x, BATCH_SIZE,IMAGE_SIZE,C_DIMS,N_CLASSES)(model).model()
    logit = tf.nn.softmax(pred)

    
    
    # 我门存放模型的路径
    logs_train_dir = 'checkpoint/mynet'
    checkpoint_dir = '/gpfs/home/stu16/shaw/Tell_Cat/code/checkpoint'
    # 定义saver 
    saver = tf.train.Saver()  
    result=[]
    pre_label=[]
    true_label=[]
    M=np.zeros(shape=(12,12))
    with tf.Session() as sess:
        print("从指定的路径中加载模型。。。。")
        # 将模型加载到sess 中 
        ckpt = tf.train.get_checkpoint_state(logs_train_dir)
        saver.restore(sess, ckpt.model_checkpoint_path)
        print(ckpt.model_checkpoint_path)
        it = len(test_data) // BATCH_SIZE    #2160/48=45
        start_time = time.time()
        print(len(test_data),BATCH_SIZE,it)
        for idx in range(0, it):
            batch_images = test_data[idx*BATCH_SIZE : (idx+1)*BATCH_SIZE]
            batch_labels = test_label[idx*BATCH_SIZE : (idx+1)*BATCH_SIZE]
            #summary, _, err, acc = self.sess.run([self.merge, self.train_op, self.loss, self.acc], feed_dict={self.images: batch_images, self.labels: batch_labels})
            prediction = sess.run(logit, feed_dict={x: batch_images})
            #print(idx+1,prediction.shape,max_index,type(prediction),prediction[0,:],type(prediction[0,:]))
            maxvalue = [max(i) for i in prediction]
            maxindex = [np.argmax(i) for i in prediction]
            for i in range(0,BATCH_SIZE):
                P=maxindex[i]
                T=batch_labels[i]
                pre_label.append(P)
                true_label.append(T)
                M[T,P]+=1
        acc = sum(M.diagonal())/np.sum(M)
        print('ACC: %.2f' %(acc*100.0))

    # with open('prediction.txt','w') as ff:
    #     for ll in range(len(pre_label)):
    #         ff.writelines(str(pre_label[ll]))
    #         ff.writelines('\n')
    #         ff.writelines(str(true_label[ll]))
    #         ff.writelines('\n')
    #         ff.writelines('\n')
    # print('write ok!')

# 测试
test()