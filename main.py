import argparse
import os 
import glob
import time 
import sys


sys.path.insert(0,'./utils')
from globalVariables import ret_dict,data,res_dict,LABELS_SWORD_COL,_2stream

CHEKPOINT = "./checkpoints"
WEIGHTS = "weights"
LABELS = "classes"

# settings for WampServer 
php_webservice = "http://localhost/combine/webservices.php"
wamp_folder = '/opt/lampp/htdocs/combine/'

def get_sys_info(sys_name):
	
	rgb_dir = None
	oflow_dir = None
	lstm_dir = None
	labels = None

	# find which words folder been chosen.
	systems = glob.glob(os.path.join(CHEKPOINT,'*'))	
	systems = list(map(lambda s: s.rsplit(f'{os.sep}',1)[-1],systems))

	if not sys_name in systems or len(systems) == 0:
		raise ValueError(f"ERROR : could not find {sys_name} in {CHEKPOINT} directory.")

	sys_path = os.path.join(CHEKPOINT,sys_name)

	# get weights.
	sys_weights = glob.glob(os.path.join(sys_path,WEIGHTS,'*.h5'))

	if len(sys_weights) == 0:
		raise ValueError(f"ERROR : no weights has been found in {WEIGHTS} folder.")

	# find rgb,oflow,lstm,lstm_cpu
	h5_files = ['rgb','oflow','lstm','cpu']
	h5_dirs = {}
	for h5_file in h5_files:

		h5_dir = [weights for weights in sys_weights if h5_file in weights.lower()]
		if len(h5_dir) > 1:
			raise ValueError(f"ERROR : In {h5_dir[0].rsplit(os.sep,1)[0]} directory more than one {h5_file} file found.")
		
		h5_dirs[h5_file] = h5_dir[0] if len(h5_dir) > 0 else None

	# get labels file
	sys_labels = glob.glob(os.path.join(sys_path,LABELS,'*.csv'))


	if len(sys_labels) != 1:
		raise ValueError(f"ERROR : something wrong with {LABELS} folder.")

	return h5_dirs,sys_labels[0]	

def print_sys_info(args):

	print("running the system with:")
	for arg in vars(args):
		print(' '*3,f'{arg} = {getattr(args,arg)}')

if __name__ == '__main__' :

	parser = argparse.ArgumentParser()

	# --run 
	parser.add_argument(
		'-run',
		'--run',
		dest='run_method',
		type=str,
		default='REST_API',
		help='choose a way to test the sign language system.')
	parser.add_argument(
		'-sys',
		'--system',
		dest='system_name',
		type=str,
		default='turkish_10_word',
		help='choose which sign language system to run.')
	parser.add_argument(
		'-use_lstm',
		'--use_lstm',
		dest='use_lstm',
		type=bool,
		default=False,
		help='add lstm on top of stream network.')
	parser.add_argument(
		'-rgb',
		'--rgb_only',
		dest='use_rgb',
		type=bool,
		default=False,
		help='just use rgb stream.')
	parser.add_argument(
		'-oflow',
		'--oflow_only',
		dest='use_oflow',
		type=bool,
		default=True,
		help='just use optical flow stream.')
	parser.add_argument(
		'-on_cpu',
		'--use_cpu',
		dest='on_cpu',
		type=bool,
		default=True,
		help='run the system on cpu.')
	parser.add_argument(
		'-pred_type',
		'--prediction_type',
		dest='pred_type',
		type=str,
		default='word',
		help='define how the system output will be, either word or sentence.')
	parser.add_argument(
		'-nTop',
		'--top_predictions',
		dest='nTop',
		type=int,
		default=3,
		help='how many result(output) should the system give.')
	parser.add_argument(
		'-download',
		'--download',
		dest='download',
		type=bool,
		default=False,
		help='download weights and classes to checkpoints directory.')
	parser.add_argument(
		'-mul_oflow',
		'--multiprocessing_opticalflow',
		dest='mul_oflow',
		type=bool,
		default=False,
		help="faster optical flow calculation with multiprocessing.")
	parser.add_argument(
		'-oflow_pnum',
		'--oflow_process_num',
		dest='oflow_pnum',
		type=int,
		default=4,
		help="number of processes to calculate optical flow.")
	parser.add_argument(
		'-mul_2stream',
		'--multiprocessing_two_stream',
		dest='mul_2stream',
		type=bool,
		default=False,
		help='run two stream on different processes.')
	# CPU OR GPU
	# HOW MUCH FRACTION ON GPU DO YOU WANT TO USE 
	# WHICH GPU TO RUN ON
	# WORDS OR SENTENCES
	# SINGLE CPU OR MULTIPULE
	# use just rgb or just oflow
	# don't use lstm
	args = parser.parse_args()

	# run test script 
	run_method = args.run_method
	use_lstm = args.use_lstm
	use_rgb = args.use_rgb	
	use_oflow = args.use_oflow
	on_cpu = args.on_cpu	
	pred_type = args.pred_type
	nTop = args.nTop
	download = args.download
	mul_oflow = args.mul_oflow
	oflow_pnum = args.oflow_pnum
	mul_2stream = args.mul_2stream
	system_name = args.system_name

	# download model weights and labels
	if download:
		from checkpoints.download import download_sys
		Dir = CHEKPOINT+os.sep+system_name
		print(f"downloading weights and lables for {system_name} system to {Dir}.")
		download_sys(system_name,Dir)

	#load checkpoints and labels
	models_dir,labels_dir = get_sys_info(system_name)
	# informative message
	print(f"In {args.system_name} folder:")
	for k,v in models_dir.items():
		if v is not None:
			# informative message
			print(f"{' '*4}{k.upper()} WEIGHTS found : {v.rsplit(os.sep,1)[-1]}")
	# informative message
	print(f"{' '*4}labels : {labels_dir.rsplit(os.sep,1)[-1]}")

	
	# make sure that flags are set properlly
	if use_rgb and use_oflow:
		raise ValueError("""ERROR : both rgb and oflow flags are on.
						 trying to use both? set both flag to 'False'""")
	if not pred_type == "word" and not pred_type == "sentence":
		raise ValueError("ERROR : pred_type should be 'word' or 'sentence'")
	con = mul_oflow and not oflow_pnum > 0
	#notcon = not mul_oflow and oflow_pnum > 0 
	if con:
		raise ValueError("ERROR : check mul_oflow and oflow_pnum flags.")
	if not on_cpu and mul_2stream:
		raise ValueError("ERROR : you can't use multiprocessing on streams while the system is running on gpu.")  
	if (use_rgb or use_oflow) and mul_2stream:
		raise ValueError("ERROR : you can't do multiprocessing while using just one stream!.")
	# print informative messages for what will be used next
	print_sys_info(args) 


	

	if on_cpu:
		os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
		os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


	from collections import defaultdict

	models = defaultdict(lambda : None)

	from utils.util import load_models,csv_to_dict
	from multiprocessing import Manager
	from multiprocessing import Process

	labels = csv_to_dict(labels_dir,LABELS_SWORD_COL)
		
	if not mul_2stream:

		# load labels
		print(f"loading labels from {labels_dir}.")
		labels = csv_to_dict(labels_dir,LABELS_SWORD_COL)
		print(f"{len(labels)} word found in {labels_dir}")


		# load models
		uploading_time = time.time()
		print("Initializing models")
		models = load_models(models_dir,
								on_cpu,
								use_rgb,
								use_oflow,
								use_lstm,
								False)
		print(f"Uploading took {round(time.time()-uploading_time,2)} sec")
	else:
		models['oflow'] = 1
		from utils.parallel_streams import nn_work

		_2stream.append(Process(target=nn_work, args=('oflow',models_dir,labels_dir,pred_type,nTop,mul_oflow,oflow_pnum)))
		_2stream.append(Process(target=nn_work, args=('rgb',models_dir,labels_dir,pred_type,nTop,mul_oflow,oflow_pnum)))
		if use_lstm:
			_2stream.append(Process(target=nn_work, args=('oflow',models_dir,labels_dir,pred_type,nTop,mul_oflow,oflow_pnum)))

		for p in _2stream:
			p.start()

		print(f"{len(_2stream)} process has been initialized.")

	# run some server with flags cpu gpu pred_type nTop
	# if wamp
	if run_method == "wamp":
		print("running wamp server.")
		from run.wamp import run_server 

		if not os.path.exists(wamp_folder):
			raise ValueError(f"ERROR : can't find wamp service in {wamp_folder} directory")

		# running wamp server
		run_server(php_webservice,
				wamp_folder,
				models,
				labels,
				pred_type,
				nTop,
				mul_oflow,
				oflow_pnum,
				mul_2stream)
	
	elif run_method == "webcam":
		print("testing system on webcam, to close webcam press 'q'.")
		from run.webcam import test

		test(models,
			labels,
			pred_type,
			nTop,
			mul_oflow,
			oflow_pnum,
			mul_2stream)

	elif run_method == "REST_API":
		print("Initiate REST API server ...")
		from run.REST_API import server

		server.run(models,
					labels,
					pred_type,
					nTop,
					mul_oflow,
					oflow_pnum,
					mul_2stream,
					host="0.0.0.0")
