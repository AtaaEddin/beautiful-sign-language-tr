import random
import os
import pandas as pd
import numpy as np
import shutil
import cv2
from glob import glob


"""
main path:
	first we take all videos from the sql file put it on text file
	train.txt for training and val.txt for evaluating put all videos 
	in same directory as spciefied in sql file 
	first run right_labeling function output train.csv and val.csv
	then move the videos by runing mv_videos function then convert to map4 
	with convert_.... function then delete webm videos

	we can delete every think you used from text files to csv files
to add more videos:
	run add_to_csv function and then continue as main path where you run mv_videos function
make labels path:
	edit the list inside makelooktable function before you run it 
"""


#current_dataset_dir = "./dataset/" #not used


def convert_wbm_file_to_mp4_file():
	# check to see if the list is empty, if not proceed
	webmFiles = glob(new_dataset_dir + "*/*/*.webm")
	if len(webmFiles) <= 0:
		print("No files to convert!")
		return
	for webm_file in webmFiles:
		mp4_file = webm_file.replace('.webm','.mp4')
		cmd_string = 'ffmpeg -i "' + webm_file + '" "' + mp4_file + '"'
		print('converting ' + webm_file + ' to ' + mp4_file)
		os.system(cmd_string)

def delet_webm():
	webmFiles = glob(new_dataset_dir + "*/*/*.webm")
	print(len(webmFiles))
	i = 0
	for webm_file in webmFiles:
		os.remove(webm_file)
		i += 1
	print(f"{i} file removed")

def right_labeling():
	#create dirs
	if not os.path.exists(new_dataset_dir):
		os.mkdir(new_dataset_dir)
	if not os.path.exists(train_dir):
		os.mkdir(train_dir)
	if not os.path.exists(val_dir):
		os.mkdir(val_dir)

	#our dataset
	cols = ['id', 'video_orginal_path', 'word_id', 'createdAt', 'size']
	train_csv = None
	val_csv = None

	with open(train_txt_dir) as train_txt, open(val_txt_dir) as val_txt:
		train_labels = list(train_txt)
		train_labels = list(map(lambda s : s.split(')')[0].replace("(","").strip().replace(" ","").split(',')
			, train_labels))
		val_labels = list(val_txt)
		val_labels = list(map(lambda s : s.split(')')[0].replace("(","").strip().replace(" ","").split(',')
			, val_labels))
		#print(labels[-1])
		#data_length = len(labels)
		#train_rows = int(data_length * 9/10)
		#val_rows = int(data_length * 1/10)
		
		
		train_csv = pd.DataFrame(train_labels, columns =cols)
		val_csv = pd.DataFrame(val_labels, columns =cols)
		new_path = []

		train_csv["video_orginal_path"].apply(lambda s: s.replace("'",""))
		#train_csv["new_path"].apply(lambda s: s.replace("'",""))
		val_csv["video_orginal_path"].apply(lambda s: s.replace("'",""))
		#val_csv["new_path"].apply(lambda s: s.replace("'",""))

		for idx,row in train_csv.iterrows():

			word_id = row["word_id"]
			video_name = row["video_orginal_path"].split("/")[-1]

			new_path.append("train/c{:02d}/{}".format(int(word_id),video_name))

		train_csv["new_path"] = new_path

		new_path = []
		for idx,row in val_csv.iterrows():

			word_id = row["word_id"]
			video_name = row["video_orginal_path"].split("/")[-1]

			new_path.append("val/c{:02d}/{}".format(int(word_id),video_name))

		val_csv["new_path"] = new_path

	train_csv = train_csv[["video_orginal_path", "new_path", "word_id", "size"]]
	val_csv = val_csv[["video_orginal_path", "new_path", "word_id", "size"]]
	
	train_csv.to_csv(train_csv_file)
	val_csv.to_csv(val_csv_file)

	print(train_csv.head(3))

def mv_vidoes(mode="train"):
	#move train file
	csv = None
	if mode == "train":
		csv = pd.read_csv(train_csv_file)
	else:
		csv = pd.read_csv(val_csv_file)

	for idx,row in csv.iterrows():
		current_dir = "./" + row["video_orginal_path"].replace("'","")
		next_dir = new_dataset_dir + row["new_path"].replace("'","")
		class_id = next_dir.split("/")[-2]

		class_dir = new_dataset_dir + mode + "/" + class_id +"/"
		if not os.path.exists(class_dir):
			os.mkdir(class_dir)
		if not os.path.exists(next_dir) and os.path.exists(current_dir):
			shutil.copy(current_dir,next_dir)
		else:
			print(f"{next_dir} file already exists!")
		

def show_video():
	class_idx = random.randint(0,len(os.listdir(train_dir)))
	class_file = os.listdir(train_dir)[class_idx]
	
	video_idx = random.randint(0,len(os.listdir(os.path.join(train_dir,class_file))))

	video_file = os.listdir(os.path.join(train_dir,class_file))[video_idx]
	video_dir = os.path.join(train_dir,class_file,video_file)
	print(video_dir)
	cap = cv2.VideoCapture(video_dir)

	while True:
		ret,img = cap.read()
		img = cv2.flip(img, 1)
		if not ret:
			break

		cv2.imshow("img",img)

		if cv2.waitKey(25) & 0xFF == ord('q'):
			break

	cv2.destroyAllWindows()
	cap.release()

def add_to_csv(mode="train", file="./hilalabi.txt"):
	csv = None
	cols = ['id', 'video_orginal_path', 'word_id', 'createdAt', 'size']

	if mode == "train":
		csv = pd.read_csv(train_csv_file)
	else:
		csv = pd.read_csv(val_csv_file)

	with open(file) as txt_file:
		labels = list(txt_file)
		labels = list(map(lambda s : s.split(')')[0].replace("(","").strip().replace(" ","").split(',')
			, labels))

	temp_csv = pd.DataFrame(labels, columns=cols)


	new_path = []
	for idx,row in temp_csv.iterrows():

		word_id = row["word_id"]
		video_name = row["video_orginal_path"].split("/")[-1]

		new_path.append("{}/c{:02d}/{}".format(mode,int(word_id),video_name))
	
	temp_csv["new_path"] = new_path

	csv = csv[["video_orginal_path", "new_path", "word_id", "size"]]
	temp_csv = temp_csv[["video_orginal_path", "new_path", "word_id", "size"]]
	print("temp_csv:\n", temp_csv.head(3))
	print("csv:\n",csv.head(3))
	print(len(csv))
	csv = csv.append(temp_csv, ignore_index=True)
	print(len(csv))


	if mode == "train":
		csv.to_csv(train_csv_file)
	else:
		csv.to_csv(val_csv_file)

def make_lookuptable(classes, save_to="turkish_classes.csv"):
	sclasses = pd.DataFrame(classes, columns=["sClass", "sWord"])
	sclasses["sClass"] = sclasses["sClass"].apply(lambda s: "c{:02d}".format(s))
	print(sclasses.head(10))
	sclasses.to_csv(save_to)

def main():
<<<<<<< HEAD
	pass
	#make_lookuptable()
=======
	make_lookuptable()
>>>>>>> c4bd54b146f1a134824e6b346b5c39b01a00cb3f
	#mv_vidoes("train")
	#right_labeling()
	#show_video()
	#convert_wbm_file_to_mp4_file()
	#delet_webm()
	#add_to_csv("train", "./new_hilal.txt")

if __name__ == "__main__":
<<<<<<< HEAD
	
	#make labels path 
	classes = [[10, "iyi"],[34, "ev"],[43, "gel"],[45, "Merhaba"],[47, "iyi akşamlar"],[48, "teşekkür edirim"],[49, "nasılsın"],[64, "evet"],[65, "hayır"],[81, "yemek"]]
	make_lookuptable(classes)
	exit(0)
=======
	#make labels path 
	classes = [[10, "güzel - iyi"],[34, "ev"],[43, "gel"],[45, "Merhaba"],[47, "iyi akşamlar"],[48, "teşekkür etmek"],[49, "nasılsın"],[64, "evet"],[65, "hayır"],[81, "yemek"]]
>>>>>>> c4bd54b146f1a134824e6b346b5c39b01a00cb3f
	classes_dir = "turkish_x_word.csv"

	#main path
	train_txt_dir = "./train.txt"
	val_txt_dir = "./val.txt"
	new_dataset_dir = "./newdataset/"
	train_dir = new_dataset_dir + "train/"
	val_dir = new_dataset_dir + "val/"
	train_csv_file = "./train.csv"
	val_csv_file = "./val.csv"


	#to add more videos 
	txt_to_add = "addition.txt"

	#make labels path
	if not os.path.exists(classes_dir):
		print("making labels ...")
		make_lookuptable(classes,classes_dir)

	else:
		print(f"found labels in {classes_dir} directory")

	phases = ["train", "val"]
	if not os.path.exists(train_csv_file):
		print("making train and val csv files")
		right_labeling()
	else:
		for p in phases:
			print(f"adding {p} line to its specified csv file")
			add_to_csv(p, txt_to_add)
	
	for p in phases:
		print(f"moving new {p} videos to {new_dataset_dir}")
		mv_videos(p)
	print(f"converting all videos in {new_dataset_dir} directory from webm to mp4.")
	convert_wbm_file_to_mp4_file()
	print(f"deleting all webm videos in {new_dataset_dir} directory.")
	delet_webm()
	print("finish executing.")
