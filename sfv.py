import tensorflow as tf
import numpy as np
import scipy.io as sio
import time
import sys
import h5py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "cpu"

meta_path = sys.argv[1]
meta = h5py.File(meta_path,'r+')

multirate = 10.0

mset  = np.array(meta['set']).astype(np.int32)
mlabel= np.array(meta['label'],np.int32)

svm_b = np.array(meta['rho'],np.float32)
numWords = meta['numWords'][0,0].astype(np.int32)
psc = np.array(meta['psc']).astype(np.float32)
ssc = np.array(meta['ssc']).astype(np.float32)
v2 = np.array(meta['V']).astype(np.float32)

weight = np.array(range(mlabel.shape[0])).astype(np.float32)
weight[mset[:,0]==1] = 0.0
weight[mset[:,0]==2] = 1.0

if mlabel.size == ssc.shape[1]:
    label = np.zeros([svm_b.shape[0],mlabel.shape[0]])
    for idx in range(svm_b.shape[0]):
        label[idx, mlabel[:,0]==idx+1]=1
    label = label*2-1
else:
    label = mlabel.transpose()


config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.InteractiveSession(config=config)

v2 = tf.Variable(v2)
v2 = tf.nn.relu(v2)

nc = tf.sqrt(tf.matmul(v2 * v2, ssc))
sc = tf.reshape(tf.matmul(v2, psc.reshape(numWords,-1)),[svm_b.shape[0], -1])
sc = tf.divide(sc,nc)
sc = sc-svm_b
y_out = tf.nn.tanh(sc*multirate)

correct_prediction = tf.equal(tf.argmax(y_out,0), tf.argmax(label,0))
#pce = tf.nn.softmax_cross_entropy_with_logits(labels=label.transpose(), logits=tf.transpose(y_out))

pce = tf.reduce_mean(tf.abs(label.transpose()-tf.transpose(y_out)), axis=-1)
cross_entropy = (pce * weight)

learningrate = tf.placeholder(tf.float32)
train_step = tf.train.AdamOptimizer(learningrate).minimize(cross_entropy)

accuracy = tf.reduce_mean(tf.cast(tf.transpose(correct_prediction), tf.float32))
sess.run(tf.global_variables_initializer())

lr = 1
for idx in range(300):
    if idx%100==0:
        lr=lr*0.1
    train_step.run(feed_dict={learningrate: lr})
V = v2.eval()

meta.pop('V')
meta['V'] = V
meta.close()
