from flask import Flask,request,url_for,render_template
from werkzeug.utils import secure_filename

try:
	from util import handler
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
		print(f"lstm: {cls.models['lstm']}, rgb: {cls.models['rgb']}, oflow: {cls.models['oflow']}")

		reuslts = handler(vid_dir,
							cls.models['lstm'],
							cls.models['rgb'],
							cls.models['oflow'],
							cls.labels,
							cls.pred_type,
							cls.nTop,
							cls.mul_oflow,
							cls.oflow_pnum,
							cls.mul_2stream)

		return reuslts


		
@app.route("/")
def index():
	return render_template("predict.html")
			

@app.route("/predict",methods=['GET','POST'])
def predict():
	if request.method == "POST":
		f = request.files["file"]
		save_dir = str(TMP_DIR+secure_filename(f.filename))
		f.save(save_dir)
		results = request_handler.handle(save_dir)
		return str(results)
	else:
		return "you should do POST request"



def run(models,labels,pred_type,nTop,mul_oflow,oflow_pnum,host="127.0.0.1",port=5000):
	request_handler.init(models,labels,pred_type,nTop,mul_oflow,oflow_pnum,mul_2stream)
	app.run(host=host,port=port)


if __name__ == '__main__' :
	run(*"lolo")

