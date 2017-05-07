#!/bin/python

import numpy as np



### Augmentation functions
### Based on functions defined in https://github.com/udacity/self-driving-car/tree/master/vehicle-detection/u-net

def augment_brightness_camera_images(img):
  ### Augment brightness
  img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
  random_bright = .25+np.random.uniform()
  img[:,:,2] = img[:,:,2]*random_bright
  img = cv2.cvtColor(img,cv2.COLOR_HSV2RGB)
  return img

def trans_image(img, bounding_boxes, trans_range):
  # Translation augmentation
  tr_x = trans_range*np.random.uniform()-trans_range/2
  tr_y = trans_range*np.random.uniform()-trans_range/2

  Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
  rows,cols,channels = img.shape

  for i in range(len(bounding_boxes)):
  	bb = bounding_boxes[i]
  	bounding_boxes[i] = BoundingBox(
  		bb.xmin + tr_x,
  		bb.xmax + tr_x,
  		bb.ymin + tr_y,
  		bb.ymax + tr_y,
  		bb.label)

  img = cv2.warpAffine(img,Trans_M,(cols,rows))

  return img, bounding_boxes


def stretch_image(img, bounding_boxes, scale_range):
  # Stretching augmentation
  tr_x1 = scale_range*np.random.uniform()
  tr_y1 = scale_range*np.random.uniform()
  p1 = (tr_x1, tr_y1)
  tr_x2 = scale_range*np.random.uniform()
  tr_y2 = scale_range*np.random.uniform()
  p2 = (img.shape[1]-tr_x2,tr_y1)

  p3 = (img.shape[1]-tr_x2,img.shape[0]-tr_y2)
  p4 = (tr_x1,img.shape[0]-tr_y2)

  pts1 = np.float32([[p1[0],p1[1]],
                 [p2[0],p2[1]],
                 [p3[0],p3[1]],
                 [p4[0],p4[1]]])
  pts2 = np.float32([[0,0],
                 [img.shape[1],0],
                 [img.shape[1],img.shape[0]],
                 [0,img.shape[0]] ]
                 )

  M = cv2.getPerspectiveTransform(pts1,pts2)
  img = cv2.warpPerspective(img,M,(img.shape[1],img.shape[0]))
  img = np.array(img,dtype=np.uint8)

  for i in range(len(bounding_boxes)):
  	bb = bounding_boxes[i]
  	bounding_boxes[i] = BoundingBox(
  		(bb.xmin - p1[0])/(p2[0]-p1[0])*img.shape[1],
  		(bb.xmax - p1[0])/(p2[0]-p1[0])*img.shape[1],
  		(bb.ymin - p1[1])/(p3[1]-p1[1])*img.shape[0],
  		(bb.ymax - p1[1])/(p3[1]-p1[1])*img.shape[0],
  		bb.label)

  return img, bounding_boxes
