from flask import Flask,request,url_for,render_template,jsonify
from werkzeug.utils import secure_filename
from filetype.match import video
import filetype
import os

#from pymediainfo import MediaInfo
#from moviepy.editor import VideoFileClip
#os.environ['PATH'] = os.path.dirname(os.path.join(os.getcwd(),'run','REST_API','MediaInfo/')) + ';' + os.environ['PATH']

try:
	from util import handler,json_to_kiwi,IsCorrupted
except ImportError:
	print("you should run server from the 'main.py' in home directory.")

TMP_DIR = "./tmp/"

app = Flask(__name__)

class request_handler(object):
	"""docstring for flaskServer"""

	@classmethod
	def init(cls, models,labels,pred_type,nTop,mul_oflow,oflow_pnum,mul_2stream):
		
		cls.models = models
		cls.labels = labels
		cls.pred_type = pred_type
		cls.nTop = nTop
		cls.mul_oflow = mul_oflow
		cls.oflow_pnum = oflow_pnum
		cls.mul_2stream = mul_2stream

	@classmethod
	def handle(cls, vid_dir):

		reuslts,process_time = handler(vid_dir,
							cls.models['lstm'],
							cls.models['rgb'],
							cls.models['oflow'],
							cls.labels,
							cls.pred_type,
							cls.nTop,
							cls.mul_oflow,
							cls.oflow_pnum,
							cls.mul_2stream)

		return reuslts,process_time


		
@app.route("/")
def index():
	return render_template("predict.html")
			

@app.route("/predict",methods=['GET','POST'])
def predict():
	results = None
	message = ""
	process_time = 0
	success = False
	Isvideo = False
	if request.method == "POST":
		f = request.files["file"]
		filename = secure_filename(f.filename)
		ext = os.path.basename(filename).split('.')[-1]
		save_dir = str(TMP_DIR+"tmp."+ext)
		f.save(save_dir)
		#check if it's a video or not
		#try:
			#clip = VideoFileClip(save_dir)
		#except Exception as e:
			#print(e)
			#Isvideo = False
		#Isvideo = not IsCorrupted(save_dir)
		Isvideo = video(save_dir) is not None
		if Isvideo:
			try:
				results,process_time = request_handler.handle(save_dir)
				message = "File processed successfully"
				success = True
			except:
				message = "Entrenal Error occured while trying to process the file. \nplease contact with @kiwi-team at http://kiwi.wiki.com.tr/support."
				success = False
			
		else:
			kind = filetype.guess(save_dir)
			if kind is None:
				message = "Unrecognized or unsupported file type sent to server... \nplease send video files with general supported formats 'mp4','webm','avi',...."
			else:
				message = f"""server received unexpected file (type: {kind.mime if hasattr(kind,'mime') else 'Unknown'} - extention: {kind.extention if hasattr(kind,'extention') else 'Unknown'}).\nplease send video files with general supported formats 'mp4','webm','avi',...."""
				
		json_result = json_to_kiwi(results,success,message,process_time)
		return app.response_class(
						response=json_result,
						status=200,
						content_type='application/json; charset=utf-8',
						mimetype='application/json')
	else:
		return "you should do POST request"



def run(models,labels,pred_type,nTop,mul_oflow,oflow_pnum,mul_2stream,host="0.0.0.0"):
	request_handler.init(models,labels,pred_type,nTop,mul_oflow,oflow_pnum,mul_2stream)
	app.run(host=host,threaded=True)


if __name__ == '__main__' :
	run(*"lolo")

