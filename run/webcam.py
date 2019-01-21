from utils.util import handler,load_models
import time
import cv2

def record_video():
	# record a video from webcam 

	video_name = "./outpy.avi"
	cap = cv2.VideoCapture(0)
	video_size = (int(cap.get(3)),int(cap.get(4)))

	out = cv2.VideoWriter(video_name,cv2.VideoWriter_fourcc('M','J','P','G'), 10, video_size)

	while True:
		
		ret,img = cap.read()
		
		if not ret:
			continue

		cv2.imshow("img",img)

		out.write(img)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break # press 'q' btn to stop
	
	cap.release()
	out.release()
	cv2.destroyAllWindows()

	return video_name

"""
			models,
			labels,
			pred_type,
			nTop):
"""
def test(models, labels, pred_type, nTop):
	
	rgb_model,oflow_model,lstmModel = models["rgb"],models["oflow"],models["lstm"]

	video_dir = record_video()
	print(video_dir)
	predictions = handler(video_dir,lstmModel,rgb_model,oflow_model,
	labels,pred_type,nTop)

	print(predictions)

