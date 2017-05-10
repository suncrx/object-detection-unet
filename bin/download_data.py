#!/bin/python

import os
import sys
import time

project_path, x = os.path.split(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(project_path)

from unet.helper import full_path, untar_data, download_file

def download():
	urls = [("http://bit.ly/udacity-annotations-autti", "data/object-dataset.tar.gz", "data/object-dataset"),
					("http://bit.ly/udacity-annoations-crowdai", "data/object-detection-crowdai.tar.gz", "data/object-detection-crowdai")]

	for url, file, directory in urls:
		if os.path.isdir(full_path(directory)) == False:
			download_file(url, full_path(file))
			untar_data(full_path(file))
		else:
			print("Directory", directory, "is already present. If this is an error, remove the directory and rerun.")

download()
