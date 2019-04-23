<<<<<<< HEAD
#from multiprocessing import Manager

_2stream = []
LABELS_SWORD_COL = "sWord"
data = {}#Manager().dict()
data['rgb'],data['oflow'],data['lstm'] = [],[],[]
data['lstm'] = [None] * 2
ret_dict = {}#Manager().dict()
res_dict = {}#Manager().dict()
=======
from multiprocessing import Manager

_2stream = []
LABELS_SWORD_COL = "sWord"
data = Manager().dict()
data['rgb'],data['oflow'],data['lstm'] = [],[],[]
data['lstm'] = [None] * 2
ret_dict = Manager().dict()
res_dict = Manager().dict()
>>>>>>> c4bd54b146f1a134824e6b346b5c39b01a00cb3f
res_dict['rgb'],res_dict['oflow'],res_dict['lstm'],res_dict['rgb_time'],res_dict['oflow_time'] = [],[],[],[],[]
