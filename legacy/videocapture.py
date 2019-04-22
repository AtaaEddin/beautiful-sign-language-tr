"""
https://github.com/FrederikSchorr/sign-language

Utilites to launch webcam, capture/record video, show rectangles & text on screen.

"""


# import the necessary packages
import time

import numpy as np
from datagenerator import VideoClasses
import cv2
from Streamer import streamer
from timer import Timer
from frame import image_crop, images_crop, frames_show
from opticalflow import OpticalFlow, frames2flows, flow2colorimage, flows2colorimages, unittest_fromfile
from predict import predict
from multiprocessing.pool import ThreadPool

import keras


def video_start(device = 0, tuResolution =(320, 240), nFramePerSecond = 30):
	""" Returns videocapture object/stream

	Parameters:
		device: 0 for the primary webcam, 1 for attached webcam
	"""
	
	# try to open webcam device
	oStream = cv2.VideoCapture(device) 
	if not oStream.isOpened():
		# try again with inbuilt camera
		print("Try to initialize inbuilt camera ...")
		device = 0
		oStream = cv2.VideoCapture(device)
		if not oStream.isOpened(): raise ValueError("Could not open webcam")

	# set camera resolution
	nWidth, nHeight = tuResolution
	oStream.set(3, nWidth)
	oStream.set(4, nHeight)

	# try to set camera frame rate
	oStream.set(cv2.CAP_PROP_FPS, nFramePerSecond)

	#print("Initialized video device %d, with resolution %s and target frame rate %d" % \
		#(device, str(tuResolution), nFramePerSecond))

	return oStream



def rectangle_text(arImage, sColor, sUpper, sLower = None, tuRectangle = (224, 224)):
	""" Returns new image (not altering arImage)
	"""
	
	nHeigth, nWidth, _ = arImage.shape
	nRectHeigth, nRectWidth = tuRectangle
	x1 = int((nWidth - nRectWidth) / 2)
	y1 = int((nHeigth - nRectHeigth) / 2)

	if sColor == "green": bgr = (84, 175, 25)
	elif sColor == "orange": bgr = (60, 125, 235)
	else: #sColor == "red": 
		bgr = (27, 13, 252)

	arImageNew = np.copy(arImage)
	cv2.rectangle(arImageNew, (x1, y1), (nWidth-x1, nHeigth-y1), bgr, 3)

	# display a text to the frame 
	font = cv2.FONT_HERSHEY_SIMPLEX
	fFontSize = 0.4
	textSize = cv2.getTextSize(sUpper, font, 1.0, 2)[0]
	cv2.putText(arImageNew, sUpper, (x1 + 7, y1 + textSize[1] + 7), font, fFontSize, bgr, 2)	

	# 2nd text
	if (sLower != None):
		bgr = (27, 13, 252)
		textSize = cv2.getTextSize(sLower, font, 1.0, 2)[0]
		cv2.putText(arImageNew, sLower, (x1 + 7, nHeigth - y1 - 7), font, fFontSize, bgr, 2)

	return arImageNew


def video_show(oStream, sColor, sUpper, sLower = None, tuRectangle = (224, 224), nCountdown = 0): 
	
	if nCountdown > 0: 
		fTimeTarget = time.time() + nCountdown
	
	# loop over frames from the video file stream
	s = sUpper
	while True:
		# grab the frame from the threaded video file stream
		(bGrabbed, arFrame) = oStream.read()
		arFrame = rescale_frame(arFrame, 320, 240)
		if bGrabbed == False: continue

		if nCountdown > 0:
			fCountdown = fTimeTarget - time.time()
			s = sUpper + str(int(fCountdown)+1) + " sec"

		# paint rectangle & text, show the (mirrored) frame
		arFrame = rectangle_text(cv2.flip(arFrame, 1), sColor, s, sLower, tuRectangle)

		#arFrame = cv2.resize(arFrame, (320, 240), 0, 0, cv2.INTER_CUBIC)
		cv2.imshow("Video", arFrame)
	
		# stop after countdown
		if nCountdown > 0 and fCountdown <= 0.0:
			key = -1
			break

		# Press 'q' to exit live loop
		key = cv2.waitKey(1) & 0xFF
		if key != 0xFF: break
	return key


def video_capture(oStream, sColor, sText, tuRectangle = (224, 224), nTimeDuration = 3, bOpticalFlow = False) -> \
	(float, np.array, np.array):
	
	
	if bOpticalFlow:
		oOpticalFlow = OpticalFlow(bThirdChannel = False)

	liFrames = []
	liFlows = []
	fTimeStart = time.time()

	# loop over frames from the video file stream
	while True:
		# grab the frame from the threaded video file stream
		(bGrabbed, arFrame) = oStream.read()
		arFrame = rescale_frame(arFrame, 320, 240)
		arFrame = cv2.flip(arFrame, 1)
		liFrames.append(arFrame)

		fTimeElapsed = time.time() - fTimeStart
		s = sText + str(int(fTimeElapsed)+1) + " sec"

		# paint rectangle & text, show the frame
		arFrameText = rectangle_text(arFrame, sColor, s, "", tuRectangle)
		cv2.imshow("Video", arFrameText)

		# display optical flow
		if bOpticalFlow:
			arFlow = oOpticalFlow.next(image_crop(arFrame, *tuRectangle))
			liFlows.append(arFlow)
			h,w,_ = arFlow.shape
			arZeros = np.zeros((h, w, 1), dtype = np.float32)
			arFlow = np.concatenate((arFlow, arZeros), axis=2) 
			cv2.imshow("Optical flow", flow2colorimage(arFlow))

		# stop after nTimeDuration sec
		if fTimeElapsed >= nTimeDuration: break

		# Press 'q' for early exit
		key = cv2.waitKey(1) & 0xFF
		if key == ord('q'): break
		cv2.waitKey(1)

	return fTimeElapsed, np.array(liFrames), np.array(liFlows)



def frame_show(oStream, sColor:str, sText:str, tuRectangle = (224, 224)):
	""" Read frame from webcam and display it with box+text """

	(bGrabbed, oFrame) = oStream.read()
	oFrame = rescale_frame(oFrame, 320, 240)
	oFrame = rectangle_text(cv2.flip(oFrame, 1), sColor, sText, "", tuRectangle)
	cv2.imshow("Video", oFrame)
	cv2.waitKey(1)

	return

def rescale_frame(frame, percentx=75, percenty=75):
    width = percentx#int(frame.shape[1] * percentx/ 100)
    height = percenty#int(frame.shape[0] * percenty/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

def unittest_videocapture():
	print("--> Loading Nerual Network Model ...")
	i3d_model = keras.models.load_model( "./model/20181011-1229-chalearn249-oflow-i3d-entire-best.h5")

	pool = ThreadPool(processes=1)
	cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
	cv2.resizeWindow("Video", 320, 240)

	cv2.namedWindow("Optical flow", cv2.WINDOW_NORMAL)
	cv2.resizeWindow("Optical flow", 224, 224)

	# open a pointer to the video stream
	print("--> Opening the stream...")
	oStream = video_start(device = "rtsp://192.168.1.45:8080/h264_ulaw.sdp", tuResolution = (320, 240), nFramePerSecond = 15)
	#oStream.set(3, 320)
	#oStream.set(4, 240)
	#liFrames = []
	
	# loop over action states
	sResults = ""
	while True:
		# show live video and wait for key stroke
		key = video_show(oStream, "green", "Press <blank> to start", sResults)
		
		# start!
		if key == ord(' '):
			# countdown n sec
			video_show(oStream, sColor = "orange", sUpper = "Recording starts in ", sLower = None, 
				tuRectangle = (224, 224), nCountdown = 3)
			
			# record video for n sec
			fElapsed, liFrames, liFlows = video_capture(oStream, "red", "Recording ", nTimeDuration=5, bOpticalFlow=True)

			# show orange wait box
			frame_show(oStream, "orange", "Translating sign ...")
			#_thread.start_new_thread( predict, (liFlows, "./model/20181011-1058-chalearn249-oflow-i3d-entire-best.h5", VideoClasses("./my_classes.csv")) )
			#async_result = pool.apply_async(predict, (liFlows, i3d_model, VideoClasses("./my_classes.csv")) )
			#top = async_result.get()
			# run NN to translate video to label
			top = predict(liFlows, i3d_model, VideoClasses("./my_classes.csv"))
			#time.sleep(3)
			#predict(liFlows, "./model/20181011-1058-chalearn249-oflow-i3d-entire-best.h5", VideoClasses("./my_classes.csv"))
			oStream = video_start(device = "rtsp://192.168.1.45:8080/h264_ulaw.sdp", tuResolution = (320, 240), nFramePerSecond = 15)
			#sResults = "Video duration {:.1f} sec, {} frames recorded, {:.1f} fps". \
				#format(fElapsed, len(liFrames), len(liFrames)/fElapsed)
			sResults = f"label:{top['detail']}-{top['confidence']}"

			#video info
			print("Video duration {:.1f} sec, {} frames recorded, {:.1f} fps". \
				format(fElapsed, len(liFrames), len(liFrames)/fElapsed))

			# ready for next video	

		elif key == ord("+"):
			fFPS *= 2.
			print("Frame per second increased from %.1f to %.1f" % (oStream.get(cv2.CAP_PROP_FPS),fFPS))
			oStream.set(cv2.CAP_PROP_FPS, fFPS)

		elif key == ord("-"):
			fFPS /= 2.
			print("Frame per second decreased from %.1f to %.1f" % (oStream.get(cv2.CAP_PROP_FPS), fFPS))
			oStream.set(cv2.CAP_PROP_FPS, fFPS)

		# quit
		elif key == ord('q'):
			break

		cv2.waitKey(1)

	# do a bit of cleanup
	oStream.release()
	cv2.destroyAllWindows()

	return


def unittest_opticalflow_fromcamera():

    timer = Timer()

    # start video capture from webcam
    oStream = video_start(1, (320, 240), 15)

    # loop over action states
    while True:
        # show live video and wait for key stroke
        key = video_show(oStream, "green", "Press <blank> to start", "")
        
        # start!
        if key == ord(' '):
            # countdown n sec
            video_show(oStream, "orange", "Recording starts in ", sLower = None, \
				tuRectangle = (224, 224), nCountdown = 3)
            
            # record video for n sec
            fElapsed, arFrames, _ = video_capture(oStream, "red", "Recording ", \
				tuRectangle = (224, 224), nTimeDuration = 5, bOpticalFlow = True)
            print("\nCaptured video: %.1f sec, %s, %.1f fps" % \
                (fElapsed, str(arFrames.shape), len(arFrames)/fElapsed))

            # show orange wait box
            frame_show(oStream, "orange", "Calculating optical flow ...")

			# calculate and show optical flow
            arFrames = images_crop(arFrames, 224, 224)
            timer.start()
            arFlows = frames2flows(arFrames, bThirdChannel=True)
            print("Optical flow per frame: %.3f" % (timer.stop() / len(arFrames)))
            frames_show(flows2colorimages(arFlows), int(5 * 1000 / len(arFrames)))    

        elif key == ord('f'):
            unittest_fromfile()

        # quit
        elif key == ord('q'):
            break

    # do a bit of cleanup
    oStream.release()
    cv2.destroyAllWindows()

    return


if __name__ == '__main__':
	unittest_videocapture()