from multiprocessing import Process, Manager

_2stream = []
LABELS_SWORD_COL = "sWord"
data = Manager().dict()
data['rgb'],data['oflow'],data['lstm'] = [],[],[]
data['lstm'] = [None] * 2
ret_dict = Manager().dict()
res_dict = Manager().dict()
res_dict['rgb'],res_dict['oflow'],res_dict['lstm'] = [],[],[]
