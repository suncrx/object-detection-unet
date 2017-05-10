#!/bin/python

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from unet import helper
from unet.bounding_box import ImageFactory

def dataframe():
	dir_label = ['object-dataset', 'object-detection-crowdai']
	df_1_file = helper.full_path("data/" + dir_label[0] +'/labels.csv')
	df_2_file = helper.full_path("data/" + dir_label[1] + "/labels.csv")

	df_1 = pd.read_csv(df_1_file, sep=" ", header=None, names=['Frame', 'xmin', 'xmax', 'ymin', 'ymax', 'ind', 'Label', 'RM'])
	df_1 = df_1.drop('RM', 1)
	df_1 = df_1.drop('ind', 1)
	df_1["image_file"] = "data/" + dir_label[0] + "/" + df_1['Frame']
	df_1["frame"] = df_1["Frame"]
	df_1["label"] = df_1["Label"].str.lower()


	df_2 = pd.read_csv(df_2_file, header=0)
	df_2 = df_2.drop("Preview URL", 1)
	df_2["image_file"] = "data/" + dir_label[1] + "/" + df_2['Frame']
	df_2["label"] = df_2["Label"].str.lower()
	df_2["frame"] = df_2["Frame"]

	df_vehicles = pd.concat([df_1,df_2]).reset_index()
	df_vehicles = df_vehicles.drop('index', 1)
	df_vehicles = df_vehicles.drop('Label', 1)
	df_vehicles = df_vehicles.drop('Frame', 1)
	df_vehicles.columns =['frame','image_file','label','ymin','xmin','ymax','xmax']

	return df_vehicles

def generator(image_factories, batch_size):
	image_obj = image_factories[0].image()
	image = image_obj.image
	mask = image_obj.image_mask
	batch_images = np.zeros((batch_size, image.shape[0], image.shape[1], image.shape[2]))
	batch_masks = np.zeros((batch_size, mask.shape[0], mask.shape[1], mask.shape[2]))

	while 1:
		for i_batch in range(batch_size):
			i_line = np.random.randint(len(image_factories))
			image_obj = image_factories[i_line].image(augment=True, trans_range=50, scale_range=50)
			batch_images[i_batch] = image_obj.image
			batch_masks[i_batch] = image_obj.image_mask

		yield batch_images, batch_masks

def image_factories(df, count=None):
	image_factories = []

	unique = df.frame.unique()
	if count is None:
		count = unique.shape[0]

	for i in range(count):
		frame = unique[i]
		matching_frames = df[df.frame.isin([frame])]
		image_factories.append(ImageFactory(matching_frames))

	return np.array(image_factories)
			
def train_test_generator(df, batch_size, test_size, is_test):
	if is_test:
		count = 100
	else:
		count = None

	images = image_factories(df, count)

	train, test = train_test_split(images, test_size = test_size)

	return generator(train, batch_size), generator(test, batch_size), len(train), len(test)






