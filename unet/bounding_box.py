#!/bin/python

import cv2
import numpy as np

from unet import augmentation
from unet.helper import full_path

class BoundingBox(object):
	def __init__(self, xmin, xmax, ymin, ymax, label):
		self.xmin = int(xmin)
		self.xmax = int(xmax)
		self.ymin = int(ymin)
		self.ymax = int(ymax)
		self.label = label

class ImageFactory(object):
	def __init__(self, bounding_boxes):
		self.image_file = bounding_boxes.image_file.iloc[0]
		self.bounding_boxes = []
		for row in bounding_boxes.iterrows():
			self.bounding_boxes.append(BoundingBox(row[1]['xmin'], row[1]['xmax'], row[1]['ymin'], row[1]['ymax'], row[1]['label']))

	def image(self, augment=False, trans_range=20, scale_range=20, size=(640,400)):
		image, bounding_boxes = image_bounding_boxes(self.image_file, self.bounding_boxes, augment, trans_range, scale_range, size)
		return Image(self.image_file, image, bounding_boxes)

class Image(object):
	def __init__(self, image_file, image, bounding_boxes):
		self.image_file = image_file
		self.image = image
		self.bounding_boxes = bounding_boxes
		self.image_mask = create_image_mask(bounding_boxes, image.shape)

def image_bounding_boxes(image_file, bounding_boxes, augment=False, trans_range=20, scale_range=20, size=(640,400)):
	img = cv2.imread(full_path(image_file))
	img_size = np.shape(img)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = cv2.resize(img, size)
	img_size_post = np.shape(img)

	if augmentation == True:
		img, bounding_boxes = augmentation.trans_image(img, bounding_boxes, trans_range)
		img, bounding_boxes = augmentation.stretch_image(img, bounding_boxes, scale_range)
		img = augmentation.augment_brightness_camera_images(img)

	new_bounding_boxes = []
	for i in range(len(bounding_boxes)):
		bb = bounding_boxes[i]
		new_bounding_boxes.append(BoundingBox(
			np.round(bb.xmin * 1. /img_size[1]*img_size_post[1]),
			np.round(bb.xmax * 1. /img_size[1]*img_size_post[1]),
			np.round(bb.ymin * 1. /img_size[0]*img_size_post[0]),
			np.round(bb.ymax * 1. /img_size[0]*img_size_post[0]),
			bb.label))

	return img, new_bounding_boxes

def create_image_mask(bounding_boxes, image_shape):
	labels = {
	'car': 0, 
	'pedestrian' : 1,
	'truck': 2,
	'trafficlight': 3,
	'biker': 4
	}

	image_mask = np.zeros((image_shape[0], image_shape[1], 5))

	for i in range(len(bounding_boxes)):
		bb = bounding_boxes[i]
		if labels[bb.label] is not None:
			mask_index = labels[bb.label]
			image_mask[bb.ymin:bb.ymax, bb.xmin:bb.xmax, mask_index] = 1.

	return image_mask
