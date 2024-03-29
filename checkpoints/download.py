from keras.utils.data_utils import get_file
import requests
from tqdm import tqdm
import argparse
import os

#put here the links to the models to be downloaded
SYSTEMS = {
	'turkish_10_word' : {'weights' : {'rgb':'https://storage.googleapis.com/beautiful-sign-language-tr-checkpoints/20181129-0800-chalearn035-rgb-i3d-entire-best_acc_98.h5',
										'oflow':"https://storage.googleapis.com/beautiful-sign-language-tr-checkpoints/20181129-1002-chalearn035-oflow-i3d-entire-best.h5",
										'lstm':None},
										'classes' : 'https://storage.googleapis.com/beautiful-sign-language-tr-checkpoints/turkish_classes.csv'},

	'turkish_20_word' : {'weights' : {'rgb':None,
										'oflow':None,
										'lstm':None},
										'classes' : None}
}
			

def download(url,save_to):
	import urllib.request
	urllib.request.urlretrieve(url,save_to)
	

def download_sys(sys, dumb_folder):
	root = os.getcwd()
	folder_dir = os.path.join(root,dumb_folder)
	weight_dir = folder_dir + os.sep + "weights" + os.sep
	classes_dir = folder_dir + os.sep + "classes" + os.sep

	os.makedirs(folder_dir, exist_ok=True)
	os.makedirs(weight_dir, exist_ok=True)
	os.makedirs(classes_dir, exist_ok=True)

	if SYSTEMS[sys] is None:
		raise ValueError(f"{sys} is not yet uploaded to a cloud storage.")

	# download weights
	weights = SYSTEMS[sys]['weights']

	for model_name,weight_url in weights.items():
		if weight_url:
			print(f"downloading {model_name} weights from {weight_url} ...")
			downloaded_weights_path = get_file(model_name+".h5", weight_url, cache_subdir=weight_dir)
		"""
		print(f"dumbing weights to {downloaded_weights_path} ...")
		with open(model_name,"wb") as dumb_f:
			dumb_f.write(downloaded_weights_path)	
		"""

	# download labels
	classes_url = SYSTEMS[sys]['classes']

	download(classes_url,classes_dir+"classes.csv")

"""

if __name__ == "__main__" :

	parser = argparse.ArgumentParser()
	parser.add_argument(
		'-sys',
		'--system',
		dest='sys',
		type=str,
		default='turkish_10_word',
		help='which system to download.')
	parser.add_argument(
		'-f',
		'--folder',
		dest='folder',
		type=str,
		default='turkish_10_word',
		help='the name of the folder that will hold the downloaded system')

	args = parser.parse_args()

	download_sys(args.sys,args.folder)


"""