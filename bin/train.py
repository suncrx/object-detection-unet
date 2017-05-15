#!/bin/python 

import tensorflow as tf
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
import os
import sys
import time

project_path, x = os.path.split(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(project_path)

from unet.model import IOU_calc_loss, IOU_calc, unet_model, unet_test_model
from unet import data
from unet.helper import full_path, upload_s3, stop_instance

def train_model(batch_size, model_filepath, csv_filepath, min_delta, patience, test_size, epochs, is_test):
	t = time.time()
	print("Creating generators.")
	df = data.dataframe()
	train_generator, test_generator, train_size, validation_size = data.train_test_generator(df, batch_size, test_size, is_test)
	print("Done creating generators, took", time.time() - t, "seconds.")

	batch = next(train_generator)
	image_shape = batch[0][0].shape
	segments = batch[1][0].shape[2]

	if is_test:
		model = unet_test_model(image_shape, segments)
	else:
		model = unet_model(image_shape, segments)

	checkpoint = ModelCheckpoint(model_filepath, verbose=1, save_best_only=True)
	earlystop = EarlyStopping(min_delta=min_delta, patience=patience, verbose=1)
	logger = CSVLogger(csv_filepath)

	model.compile(optimizer=Adam(lr=1e-4), loss=IOU_calc_loss, metrics=[IOU_calc])

	print("Compiled model!")

	print(model.summary())

	model.fit_generator(train_generator, 
		steps_per_epoch=train_size, 
		epochs=epochs,
		callbacks=[checkpoint, earlystop, logger], 
		validation_data=test_generator, 
		validation_steps=validation_size,
		verbose=2)

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 1, 'The batch size for the generator')
flags.DEFINE_string('model_filepath', 'model.ckpt', 'The saved model filepath.')
flags.DEFINE_string('csv_filepath', 'training.log', 'The saved logs from training.')
flags.DEFINE_float('test_size', 0.3, 'The percentage of testing/validation data.')
flags.DEFINE_integer('epochs', 1, 'Number of training examples.')
flags.DEFINE_float('min_delta', 1.0, 'Early stopping minimum change value.')
flags.DEFINE_integer('patience', 10, 'Early stopping epochs patience to wait before stopping.')
flags.DEFINE_boolean('is_test', True, 'Whether this is a test run or the real thing.')
flags.DEFINE_boolean('stop', True, 'Stop aws instance after finished running.')

def main(_):
	print("Using batchsize", FLAGS.batch_size)
	train_model(FLAGS.batch_size, 
		full_path(FLAGS.model_filepath), 
		full_path(FLAGS.csv_filepath), 
		FLAGS.min_delta, 
		FLAGS.patience, 
		FLAGS.test_size, 
		FLAGS.epochs, 
		FLAGS.is_test)

	print("Done training!")

	if FLAGS.is_test is False and FLAGS.stop:
		stop_instance()

if __name__ == '__main__':
	tf.app.run()

