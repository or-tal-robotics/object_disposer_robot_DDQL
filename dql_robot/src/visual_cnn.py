#!/usr/bin/env python
# load vgg model
from dqn_model_visual import DQN
#from imagetranformer import transform
from matplotlib import pyplot
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from numpy import expand_dims
import  tensorflow as ts

# load the model
# load the model
model = VGG16()

# redefine model to output right after the first hidden layer
model = Model(inputs=model.inputs, outputs=model.layers[1].output)
model.summary()
# load the image with the required shape
img = load_img('car_top.jpg', target_size=(224, 224))
# convert the image to an array
img = img_to_array(img)
# expand dimensions so that it represents a single 'sample'
img = expand_dims(img, axis=0)
# prepare the image (e.g. scale pixel values for the vgg)
img = preprocess_input(img)
# get feature map for first hidden layer
feature_maps = model.predict(img)
print(feature_maps)
# plot all 64 maps in an 8x8 squares
square = 8
ix = 1
for _ in range(square):
	for _ in range(square):
		# specify subplot and turn of axis
		ax = pyplot.subplot(square, square, ix)
		ax.set_xticks([])
		ax.set_yticks([])
		# plot filter channel in grayscale
		pyplot.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
		ix += 1
# show the figure
#pyplot.show()

model = DQN()

# redefine model to output right after the first hidden layer
model = model.model
#model.summary()
# load the image with the required shape
img = load_img('car_top_new_low.jpg', target_size=(64, 64))

#img=transform(img,  [64, 64])

# convert the image to an array
img = img_to_array(img)
#print(type(img))

#img== ts.keras.preprocessing.image.array_to_img(img)
# expand dimensions so that it represents a single 'sample'

img = expand_dims(img, axis=0)
#print(img)
# prepare the image (e.g. scale pixel values for the vgg)
#print(img)
img = preprocess_input(img)
#print(type(img))
# get feature map for first hidden layer
#img=ts.image.rgb_to_grayscale(img)
#img=img.numpy()
feature_maps = model.predict(img)
#print(type(feature_maps))
#print(feature_maps)
# plot all 64 maps in an 8x8 squares
square = 8
ix = 1
for _ in range(square):
	for _ in range(square):
		# specify subplot and turn of axis
		ax = pyplot.subplot(square, square, ix)
		ax.set_xticks([])
		ax.set_yticks([])
		# plot filter channel in grayscale
		#pyplot.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
		pyplot.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
		ix += 1
# show the figure
pyplot.show()
print(model)