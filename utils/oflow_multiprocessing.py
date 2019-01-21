from opticalflow import OpticalFlow
import numpy as np
from multiprocessing import Process, Manager
from collections import OrderedDict


def calculate_oflow(rgbFrames, return_dict, indx, prev_f):
	#print(f"proess {indx} running got rgbFrames {rgbFrames.shape}")
	liFlows = []
	#if ind

	oOpticalFlow = OpticalFlow(sAlgorithm = "farnback", bThirdChannel = False, fBound = 20)
	
	if indx != 0:
		oOpticalFlow.arPrev = cv2.cvtColor(prev_f, cv2.COLOR_BGR2GRAY)#np.array(prev_f)
		#print(f"oOpticalFlow.arPrev shape {oOpticalFlow.arPrev.shape}")

	for i in range(len(rgbFrames)):
		arFlow = oOpticalFlow.next(rgbFrames[i])

		liFlows.append(arFlow)
		#print("in loop")


#	print(f"proess {i} got oflow = {liFlows.shape}")
	return_dict[indx] = np.array(liFlows)


def process_oflow(rgbFrames, pro_num=4):
	# take normilized rgb Frames and return optical flow Frames
	# [TODO] multiprocessing
	rgbFrames = np.array(rgbFrames)
	#print(f"rgbFrame array is {rgbFrames.shape}")
	manager = Manager()
	return_dict = manager.dict()
	jobs = []
	l =int(len(rgbFrames)/pro_num)
	for i in range(pro_num):
		p = Process(target=calculate_oflow, 
					args=(rgbFrames[i*l:(i+1)*l,...],return_dict,i,rgbFrames[(i*l)-1,...])) 
		p.start()
		jobs.append(p)

	if len(rgbFrames) % pro_num != 0 :
		r = len(rgbFrames) % pro_num
		p = Process(target=calculate_oflow, 
					args=(rgbFrames[pro_num*l:(pro_num*l)+r, ...],return_dict,pro_num,rgbFrames[(pro_num*l)-1,...])) 
		p.start()
		jobs.append(p)

	#print("waiting...")
	for i in range(len(jobs)):
		jobs[i].join()

	#print("everthing is ok")

	od = OrderedDict(sorted(return_dict.items()))

	oflowFrames = []
	for k,v in od.items():
		#print(f" process {k} returned {v.shape}")
		oflowFrames.extend(v)

	return np.array(oflowFrames)