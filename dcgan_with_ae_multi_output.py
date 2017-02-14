# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import sys, os
from skimage import io
from skimage import transform
from PIL import Image




#AE: input image -> 5 conv(encoder) -> 5 deconv(decoder) -> output image
#generator: input vector -> 5 deconv -> output image
#DCGAN: input image -> 4 conv -> 4 prob




def weight_variable(shape):
	initial = tf.truncated_normal(shape,stddev=0.02)
	return tf.Variable(initial)
def bias_variable(shape):
	initial = tf.truncated_normal(shape,stddev=0.02)
	return tf.Variable(initial)
def conv2d(x,W,s):
	return tf.nn.conv2d(x,W,strides=s,padding='SAME')
def deconv2d(x,W,ops,s):
	return tf.nn.conv2d_transpose(x,W,output_shape=ops,strides=s,padding='SAME')




#parameters
img_size = 96
batch_size = 10
num_step_ae_init = 10000
num_step_dcgan_pretrain = 10000
num_step_dcgan = 200000
image_save_interval = 1000

#image channel
DIM1 = 3

#AE(encoder,decoder), generator
DIM2 = 120
DIM3 = 90
DIM4 = 60
DIM5 = 30
DIM6 = 100

#discriminiator(DCGAN)
DIM7 = 30
DIM8 = 60
DIM9 = 90
DIM10 = 120

model_file_path = "./"
model_file_name = os.path.join(model_file_path ,"dcgan_with_ae_multi_output.ckpt")
img_path = "./img/"




#load images
x_train = []
data_files = os.listdir(img_path)
data_files.sort()
for data_file in data_files:
	for fl in range(2):
		#load image
		img = io.imread(os.path.join(img_path,data_file))

		#flip image
		if fl == 1:
			img = np.array(Image.fromarray(np.uint8(img)).transpose(Image.FLIP_LEFT_RIGHT))
		img = transform.resize(img, (img_size,img_size))
		x_train.append(img)

x_train = 2*np.array(x_train)-1




input_x = tf.placeholder(tf.float32,[None,img_size,img_size,3])
input_z = tf.placeholder(tf.float32,[None,1,1,DIM6])

bs_x = tf.shape(input_x)[0]
bs_z = tf.shape(input_z)[0]

#convolution(AE img)
ae_W_conv1 = weight_variable([4,4,DIM1,DIM2])
ae_b_conv1 = bias_variable([DIM2])
ae_h_conv1 = tf.nn.relu(tf.nn.bias_add(conv2d(input_x,ae_W_conv1,[1,2,2,1]),ae_b_conv1))

ae_W_conv2 = weight_variable([4,4,DIM2,DIM3])
ae_b_conv2 = bias_variable([DIM3])
ae_h_conv2 = tf.nn.relu(tf.nn.bias_add(conv2d(ae_h_conv1,ae_W_conv2,[1,2,2,1]),ae_b_conv2))

ae_W_conv3 = weight_variable([4,4,DIM3,DIM4])
ae_b_conv3 = bias_variable([DIM4])
ae_h_conv3 = tf.nn.relu(tf.nn.bias_add(conv2d(ae_h_conv2,ae_W_conv3,[1,2,2,1]),ae_b_conv3))

ae_W_conv4 = weight_variable([4,4,DIM4,DIM5])
ae_b_conv4 = bias_variable([DIM5])
ae_h_conv4 = tf.nn.relu(tf.nn.bias_add(conv2d(ae_h_conv3,ae_W_conv4,[1,2,2,1]),ae_b_conv4))

ae_W_conv5 = weight_variable([img_size/16,img_size/16,DIM5,DIM6])
ae_b_conv5 = bias_variable([DIM6])
ae_h_conv5 = tf.nn.tanh(tf.nn.bias_add(conv2d(ae_h_conv4,ae_W_conv5,[1,img_size/16,img_size/16,1]),ae_b_conv5))


#deconvolution(AE img)
ae_W_deconv1 = weight_variable([img_size/16,img_size/16,DIM5,DIM6])
ae_b_deconv1 = bias_variable([DIM5])
ae_h_deconv1 = tf.nn.relu(tf.nn.bias_add(deconv2d(ae_h_conv5,ae_W_deconv1,[bs_x,img_size/16,img_size/16,DIM5],[1,img_size/16,img_size/16,1]),ae_b_deconv1))

ae_W_deconv2 = weight_variable([4,4,DIM4,DIM5])
ae_b_deconv2 = bias_variable([DIM4])
ae_h_deconv2 = tf.nn.relu(tf.nn.bias_add(deconv2d(ae_h_deconv1,ae_W_deconv2,[bs_x,img_size/8,img_size/8,DIM4],[1,2,2,1]),ae_b_deconv2))

ae_W_deconv3 = weight_variable([4,4,DIM3,DIM4])
ae_b_deconv3 = bias_variable([DIM3])
ae_h_deconv3 = tf.nn.relu(tf.nn.bias_add(deconv2d(ae_h_deconv2,ae_W_deconv3,[bs_x,img_size/4,img_size/4,DIM3],[1,2,2,1]),ae_b_deconv3))

ae_W_deconv4 = weight_variable([4,4,DIM2,DIM3])
ae_b_deconv4 = bias_variable([DIM2])
ae_h_deconv4 = tf.nn.relu(tf.nn.bias_add(deconv2d(ae_h_deconv3,ae_W_deconv4,[bs_x,img_size/2,img_size/2,DIM2],[1,2,2,1]),ae_b_deconv4))

ae_W_deconv5 = weight_variable([4,4,DIM1,DIM2])
ae_b_deconv5 = bias_variable([DIM1])
ae_h_deconv5 = tf.nn.tanh(tf.nn.bias_add(deconv2d(ae_h_deconv4,ae_W_deconv5,[bs_x,img_size,img_size,DIM1],[1,2,2,1]),ae_b_deconv5))

#deconvolution(generator z)
gen_W_deconv1 = weight_variable([img_size/16,img_size/16,DIM5,DIM6])
gen_b_deconv1 = bias_variable([DIM5])
gen_h_deconv1 = tf.nn.relu(tf.nn.bias_add(deconv2d(input_z,gen_W_deconv1,[bs_z,img_size/16,img_size/16,DIM5],[1,img_size/16,img_size/16,1]),gen_b_deconv1))

gen_W_deconv2 = weight_variable([4,4,DIM4,DIM5])
gen_b_deconv2 = bias_variable([DIM4])
gen_h_deconv2 = tf.nn.relu(tf.nn.bias_add(deconv2d(gen_h_deconv1,gen_W_deconv2,[bs_z,img_size/8,img_size/8,DIM4],[1,2,2,1]),gen_b_deconv2))

gen_W_deconv3 = weight_variable([4,4,DIM3,DIM4])
gen_b_deconv3 = bias_variable([DIM3])
gen_h_deconv3 = tf.nn.relu(tf.nn.bias_add(deconv2d(gen_h_deconv2,gen_W_deconv3,[bs_z,img_size/4,img_size/4,DIM3],[1,2,2,1]),gen_b_deconv3))

gen_W_deconv4 = weight_variable([4,4,DIM2,DIM3])
gen_b_deconv4 = bias_variable([DIM2])
gen_h_deconv4 = tf.nn.relu(tf.nn.bias_add(deconv2d(gen_h_deconv3,gen_W_deconv4,[bs_z,img_size/2,img_size/2,DIM2],[1,2,2,1]),gen_b_deconv4))

gen_W_deconv5 = weight_variable([4,4,DIM1,DIM2])
gen_b_deconv5 = bias_variable([DIM1])
gen_h_deconv5 = tf.nn.tanh(tf.nn.bias_add(deconv2d(gen_h_deconv4,gen_W_deconv5,[bs_z,img_size,img_size,DIM1],[1,2,2,1]),gen_b_deconv5))


#convolution(DCGAN img)
dcgan_W_conv1 = weight_variable([4,4,DIM1,DIM7])
dcgan_b_conv1 = bias_variable([DIM7])
dcgan_h_conv1_img = tf.nn.relu(tf.nn.bias_add(conv2d(input_x,dcgan_W_conv1,[1,2,2,1]),dcgan_b_conv1))

dcgan_W_conv2 = weight_variable([4,4,DIM7,DIM8])
dcgan_b_conv2 = bias_variable([DIM8])
dcgan_h_conv2_img = tf.nn.relu(tf.nn.bias_add(conv2d(dcgan_h_conv1_img,dcgan_W_conv2,[1,2,2,1]),dcgan_b_conv2))

dcgan_W_conv3 = weight_variable([4,4,DIM8,DIM9])
dcgan_b_conv3 = bias_variable([DIM9])
dcgan_h_conv3_img = tf.nn.relu(tf.nn.bias_add(conv2d(dcgan_h_conv2_img,dcgan_W_conv3,[1,2,2,1]),dcgan_b_conv3))

dcgan_W_conv4 = weight_variable([4,4,DIM9,DIM10])
dcgan_b_conv4 = bias_variable([DIM10])
dcgan_h_conv4_img = tf.nn.relu(tf.nn.bias_add(conv2d(dcgan_h_conv3_img,dcgan_W_conv4,[1,2,2,1]),dcgan_b_conv4))


#convolution(DCGAN generator)
dcgan_h_conv1_gen = tf.nn.relu(tf.nn.bias_add(conv2d(gen_h_deconv5,dcgan_W_conv1,[1,2,2,1]),dcgan_b_conv1))
dcgan_h_conv2_gen = tf.nn.relu(tf.nn.bias_add(conv2d(dcgan_h_conv1_gen,dcgan_W_conv2,[1,2,2,1]),dcgan_b_conv2))
dcgan_h_conv3_gen = tf.nn.relu(tf.nn.bias_add(conv2d(dcgan_h_conv2_gen,dcgan_W_conv3,[1,2,2,1]),dcgan_b_conv3))
dcgan_h_conv4_gen = tf.nn.relu(tf.nn.bias_add(conv2d(dcgan_h_conv3_gen,dcgan_W_conv4,[1,2,2,1]),dcgan_b_conv4))


#discriminator(DCGAN img)
dcgan_W_id1 = weight_variable([img_size/2,img_size/2,DIM7,1])
dcgan_b_id1 = bias_variable([1])
dcgan_h_id1_img = tf.nn.bias_add(conv2d(dcgan_h_conv1_img,dcgan_W_id1,[1,img_size/2,img_size/2,1]),dcgan_b_id1)

dcgan_W_id2 = weight_variable([img_size/4,img_size/4,DIM8,1])
dcgan_b_id2 = bias_variable([1])
dcgan_h_id2_img = tf.nn.bias_add(conv2d(dcgan_h_conv2_img,dcgan_W_id2,[1,img_size/4,img_size/4,1]),dcgan_b_id2)

dcgan_W_id3 = weight_variable([img_size/8,img_size/8,DIM9,1])
dcgan_b_id3 = bias_variable([1])
dcgan_h_id3_img = tf.nn.bias_add(conv2d(dcgan_h_conv3_img,dcgan_W_id3,[1,img_size/8,img_size/8,1]),dcgan_b_id3)

dcgan_W_id4 = weight_variable([img_size/16,img_size/16,DIM10,1])
dcgan_b_id4 = bias_variable([1])
dcgan_h_id4_img = tf.nn.bias_add(conv2d(dcgan_h_conv4_img,dcgan_W_id4,[1,img_size/16,img_size/16,1]),dcgan_b_id4)


#discriminator(DCGAN generator)
dcgan_h_id1_gen = tf.nn.bias_add(conv2d(dcgan_h_conv1_gen,dcgan_W_id1,[1,img_size/2,img_size/2,1]),dcgan_b_id1)
dcgan_h_id2_gen = tf.nn.bias_add(conv2d(dcgan_h_conv2_gen,dcgan_W_id2,[1,img_size/4,img_size/4,1]),dcgan_b_id2)
dcgan_h_id3_gen = tf.nn.bias_add(conv2d(dcgan_h_conv3_gen,dcgan_W_id3,[1,img_size/8,img_size/8,1]),dcgan_b_id3)
dcgan_h_id4_gen = tf.nn.bias_add(conv2d(dcgan_h_conv4_gen,dcgan_W_id4,[1,img_size/16,img_size/16,1]),dcgan_b_id4)



#tensorflow variable lists
ae_W_conv_list = [ae_W_conv1,ae_W_conv2,ae_W_conv3,ae_W_conv4,ae_W_conv5]
ae_b_conv_list = [ae_b_conv1,ae_b_conv2,ae_b_conv3,ae_b_conv4,ae_b_conv5]
ae_W_deconv_list = [ae_W_deconv1,ae_W_deconv2,ae_W_deconv3,ae_W_deconv4,ae_W_deconv5]
ae_b_deconv_list = [ae_b_deconv1,ae_b_deconv2,ae_b_deconv3,ae_b_deconv4,ae_b_deconv5]

dcgan_W_conv_list = [dcgan_W_conv1,dcgan_W_conv2,dcgan_W_conv3,dcgan_W_conv4]
dcgan_b_conv_list = [dcgan_b_conv1,dcgan_b_conv2,dcgan_b_conv3,dcgan_b_conv4]

dcgan_W_id_list = [dcgan_W_id1,dcgan_W_id2,dcgan_W_id3,dcgan_W_id4]
dcgan_b_id_list = [dcgan_b_id1,dcgan_b_id2,dcgan_b_id3,dcgan_b_id4]

gen_W_deconv_list = [gen_W_deconv1,gen_W_deconv2,gen_W_deconv3,gen_W_deconv4,gen_W_deconv5]
gen_b_deconv_list = [gen_b_deconv1,gen_b_deconv2,gen_b_deconv3,gen_b_deconv4,gen_b_deconv5]




#loss(AE initialization)
loss_ae = tf.reduce_mean(tf.square(input_x-ae_h_deconv5))
optimizer_ae = tf.train.AdamOptimizer(learning_rate=0.0001)
train_ae = optimizer_ae.minimize(loss_ae,var_list=(ae_W_conv_list+ae_b_conv_list+ae_W_deconv_list+ae_b_deconv_list))

#loss(DCGAN discriminator)
loss_dcgan = tf.reduce_mean( tf.concat(0, [
tf.nn.sigmoid_cross_entropy_with_logits(dcgan_h_id1_img,tf.ones_like(dcgan_h_id1_img)),
tf.nn.sigmoid_cross_entropy_with_logits(dcgan_h_id2_img,tf.ones_like(dcgan_h_id2_img)),
tf.nn.sigmoid_cross_entropy_with_logits(dcgan_h_id3_img,tf.ones_like(dcgan_h_id3_img)),
tf.nn.sigmoid_cross_entropy_with_logits(dcgan_h_id4_img,tf.ones_like(dcgan_h_id4_img)),
tf.nn.sigmoid_cross_entropy_with_logits(dcgan_h_id1_gen,tf.zeros_like(dcgan_h_id1_gen)),
tf.nn.sigmoid_cross_entropy_with_logits(dcgan_h_id2_gen,tf.zeros_like(dcgan_h_id2_gen)),
tf.nn.sigmoid_cross_entropy_with_logits(dcgan_h_id3_gen,tf.zeros_like(dcgan_h_id3_gen)),
tf.nn.sigmoid_cross_entropy_with_logits(dcgan_h_id4_gen,tf.zeros_like(dcgan_h_id4_gen))]))
optimizer_dcgan = tf.train.AdamOptimizer(learning_rate=0.0001)
train_dcgan = optimizer_dcgan.minimize(loss_dcgan,var_list=(dcgan_W_conv_list+dcgan_b_conv_list+dcgan_W_id_list+dcgan_b_id_list))

#loss(generator)
loss_gen = tf.reduce_mean( tf.concat(0,[
tf.nn.sigmoid_cross_entropy_with_logits(dcgan_h_id1_gen,tf.ones_like(dcgan_h_id1_gen)),
tf.nn.sigmoid_cross_entropy_with_logits(dcgan_h_id2_gen,tf.ones_like(dcgan_h_id2_gen)),
tf.nn.sigmoid_cross_entropy_with_logits(dcgan_h_id3_gen,tf.ones_like(dcgan_h_id3_gen)),
tf.nn.sigmoid_cross_entropy_with_logits(dcgan_h_id4_gen,tf.ones_like(dcgan_h_id4_gen))]))
optimizer_gen = tf.train.AdamOptimizer(learning_rate=0.0001)
train_gen = optimizer_gen.minimize(loss_gen,var_list=(gen_W_deconv_list+gen_b_deconv_list))




#replace generator weight and bias to AE ones
ae2gW = []
ae2gb = []
for i in range(len(gen_W_deconv_list)):
	ae2gW.append(tf.assign(gen_W_deconv_list[i],ae_W_deconv_list[i]))
	ae2gb.append(tf.assign(gen_b_deconv_list[i],ae_b_deconv_list[i]))




#variable initialization
saver = tf.train.Saver()
sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

#load model
if tf.train.get_checkpoint_state(model_file_path) :
	ckpt = tf.train.get_checkpoint_state(model_file_path)
	last_model = ckpt.model_checkpoint_path
	saver.restore(sess,last_model)

#AE initialization
for step in xrange(num_step_ae_init):
	batch_mask = np.random.choice(x_train.shape[0],batch_size)
	x_batch = x_train[batch_mask]

	print "AE init : ",step
	if (step+1) % image_save_interval == 0:
		output_image = tf.image.convert_image_dtype((ae_h_deconv5+1.0)/2.0, tf.uint8,saturate=True)
		with open("img_AE_init.jpg", 'wb') as f:
			f.write(sess.run(tf.image.encode_jpeg(output_image[0],quality=100,chroma_downsampling=False),feed_dict={input_x: x_batch[:1]}))

	sess.run([train_ae],feed_dict={input_x: x_batch})

saver = tf.train.Saver()
saver.save(sess, model_file_name)


#replace generator weight and bias to AE ones
for i in range(len(ae2gW)):
	sess.run(ae2gW[i])
	sess.run(ae2gb[i])

#DCGAN pretraining
for step in xrange(num_step_dcgan_pretrain):
	batch_mask = np.random.choice(x_train.shape[0],batch_size)
	x_batch = x_train[batch_mask]
	z_batch = 2.0*np.random.rand(batch_size,1,1,DIM6)-1.0

	print "DCGAN pretraining: ",step

	sess.run([train_dcgan],feed_dict={input_x: x_batch, input_z: z_batch})

saver = tf.train.Saver()
saver.save(sess, model_file_name)


#DCGAN(discriminator) vs generator
for step in xrange(num_step_dcgan):
	batch_mask = np.random.choice(x_train.shape[0],batch_size)
	x_batch = x_train[batch_mask]
	z_batch = 2.0*np.random.rand(batch_size,1,1,DIM6)-1.0

	print "DCGAN : ",step

	if (step+1) % image_save_interval == 0:
		output_image = tf.image.convert_image_dtype((gen_h_deconv5+1.0)/2.0, tf.uint8,saturate=True)
		with open("img_DCGAN.jpg", 'wb') as f:
			f.write(sess.run(tf.image.encode_jpeg(output_image[0],quality=100,chroma_downsampling=False),feed_dict={input_z: z_batch[:1]}))

	sess.run([train_dcgan],feed_dict={input_x: x_batch, input_z: z_batch})
	sess.run([train_gen],feed_dict={input_z: z_batch})


saver = tf.train.Saver()
saver.save(sess, model_file_name)


"""
#use below if you want to generate images from random vectors
row = 5
z_batch = 2.0*np.random.rand(row*row,1,1,DIM6)-1.0

imgs = sess.run(gen_h_deconv5,feed_dict={input_z: z_batch})

all_img = np.zeros((row*img_size,row*img_size,3))
for y in range(row):
	for x in range(row):
		all_img[y*img_size:(y+1)*img_size,x*img_size:(x+1)*img_size] = imgs[y*row+x]
io.imsave("img_z_AE_mo.jpg",(all_img+1.0)/2.0)
"""
sess.close()

