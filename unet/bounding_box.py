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

	def image(self, augment=False, trans_range=20, scale_range=20, size=(640,300)):
		print(self.image_file)
		image, bounding_boxes = image_bounding_boxes(self.image_file, self.bounding_boxes, augment, trans_range, scale_range, size)
		return Image(self.image_file, image, bounding_boxes)

class Image(object):
	def __init__(self, image_file, image, bounding_boxes):
		self.image_file = image_file
		self.image = image
		self.bounding_boxes = bounding_boxes

	def image_mask(self):
		labels = {
		'car': 0, 
		'pedestrian' : 1,
		'truck': 2,
		'trafficlight': 3,
		'biker': 4
		}

		image_mask = np.zeros((5, self.image.shape[0], self.image.shape[1], 1))

		for i in range(len(self.bounding_boxes)):
			bb = self.bounding_boxes[i]
			if labels[bb.label] is not None:
				mask_index = labels[bb.label]
				image_mask[mask_index, bb.xmin:bb.xmax, bb.ymin:bb.ymax, 0] = 1.

		return image_mask

def image_bounding_boxes(image_file, bounding_boxes, augment=False, trans_range=20, scale_range=20, size=(640,300)):
	img = cv2.imread(full_path(image_file))
	img_size = np.shape(img)

	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = cv2.resize(img, size)
	img_size_post = np.shape(img)

	if augmentation == True:
		img, bounding_boxes = augmentation.trans_image(img,bounding_boxes,trans_range)
		img, bounding_boxes = augmentation.stretch_image(img,bounding_boxes,scale_range)
		img = augmentation.augment_brightness_camera_images(img)

	for i in range(len(bounding_boxes)):
		bb = bounding_boxes[i]
		bounding_boxes[i] = BoundingBox(
			np.round(bb.xmin/img_size[1]*img_size_post[1]),
			np.round(bb.xmax/img_size[1]*img_size_post[1]),
			np.round(bb.ymin/img_size[1]*img_size_post[1]),
			np.round(bb.ymax/img_size[1]*img_size_post[1]),
			bb.label)

	return img, bounding_boxes


