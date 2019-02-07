from globalVariables import data,LABELS_SWORD_COL

def nn_work(model_type,models_dir,labels_dir,pred_type,nTop,mul_oflow,oflow_pnum):
	import time
	from utils.util import load_models,csv_to_dict,handler
	global data
	global LABELS_SWORD_COL

	labels = csv_to_dict(labels_dir,LABELS_SWORD_COL)
	use_rgb,use_oflow,only_lstm = False,False,False
	if model_type == 'rgb':
		use_rgb = True
	elif model_type == 'oflow':
		use_oflow = True
	else:
		only_lstm = True
	models = load_models(models_dir,
						on_cpu=True,
						use_rgb=use_rgb,
						use_oflow=use_oflow,
						use_lstm=False,
						only_lstm=only_lstm)

	from collections import defaultdict
	
	dd = defaultdict(lambda : None)
	
	dd[model_type] = models[model_type]

	assert dd[model_type] != None
	# to use lstm with two stream
	if dd['lstm'] is None and only_lstm:
		dd['lstm'] = 1

	if model_type == 'rgb':
		time.sleep(2)
	
	print(f"process {model_type} waiting for data")

	while True:
		time.sleep(0.1)
		if len(data[model_type]) == 0:
			continue

		print(f"{model_type} handling now ...")
		results = handler(None,
						dd['lstm'],
						dd['rgb'],
						dd['oflow'],
						labels,
						pred_type,
						nTop,
						mul_oflow,
						oflow_pnum,
						False,
						from_worker=True)
		
		data[model_type] = []
