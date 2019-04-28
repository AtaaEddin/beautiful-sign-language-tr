from functools import reduce
import os
import glob
import sys
import warnings
import time
import numpy as np
import pandas as pd

from datagenerator import FeaturesGenerator
from frame import frames_downsample
from globalVariables import ret_dict,res_dict,data
from util import concate

def i3d_LSTM_prediction(rgbFrames,
                        oflowFrames,
                        labels,
                        LSTM_model,
                        rgb_model,
                        oflow_model,
                        nTop,
                        mul_2stream,
                        from_worker):
    global data
    global res_dict
    rgbProbas = None
    oflowProbas = None

    if not from_worker: 
        rgbProbas = rgb_model.predict(np.expand_dims(rgbFrames, axis=0))
        oflowProbas = oflow_model.predict(np.expand_dims(oflowFrames, axis=0))
    else:
        rgbProbas,oflowProbas = data['lstm'] 
        
    LSTM_input = np.concatenate((rgbProbas,oflowProbas), axis=-1)

    arProbas_i = LSTM_model.predict(LSTM_input)
    arProbas = arProbas_i[0]
    indx = arProbas.argsort()[-nTop:][::-1]
    arTopProbas = arProbas[indx]
    
    results = []

    for i in range(nTop):
        results.append({labels[indx[i]] : round(arTopProbas[i]*100.,2)})

    if from_worker:
        res_dict['lstm'] = results
    else:
        return results


def get_predicts(rgbFrames, oflowFrames, labels, oflow_model, rgb_model, nTop, from_worker):
    global ret_dict
    global res_dict
    oflow_arProbas = None
    rgb_arProbas = None
    results = []
    for_worker_acc = False

    if oflow_model is not None:
        if rgb_model is None and not from_worker:
            results,_ = predict(oflowFrames,oflow_model,labels,nTop)
            return results
        else:
            _,oflow_arProbas = predict(oflowFrames,oflow_model,labels,nTop)
    if rgb_model is not None:
        if oflow_model is None and not from_worker:
            results,_ = predict(rgbFrames, rgb_model,labels,nTop)
            return results
        else:
            _,rgb_arProbas = predict(rgbFrames, rgb_model,labels,nTop)

    if from_worker:
            #if len(oflowFrames) > 0 and oflowFrames is not None:
            if rgb_model:
                res_dict['rgb'] = rgb_arProbas
                data['lstm'][0] = rgb_arProbas
            #if len(rgbFrames) > 0 and rgbFrames is not None:
            if oflow_model:
                res_dict['oflow'] = oflow_arProbas
                data['lstm'][1] = oflow_arProbas
            """    
            if (ret_dict['rgb'] is not None and len(ret_dict['rgb']) > 0) and\
                (ret_dict['oflow'] is not None and len(ret_dict['oflow']) > 0):
                                                
                ret_dict['lstm'] = (ret_dict['rgb'],ret_dict['oflow'])                     
                for_worker_acc = True
                oflow_arProbas = ret_dict['oflow']
                rgb_arProbas = ret_dict['rgb']
            """
    if oflow_model is not None and rgb_model is not None :
        results = concate(oflow_arProbas,rgb_arProbas,labels,nTop)
        """
        arProbas = np.concatenate((oflow_arProbas,rgb_arProbas), axis=0)
        arProbas = np.mean(arProbas, axis=0)
        index = arProbas.argsort()[-nTop:][::-1]
        arTopProbas = arProbas[index]

        for i in range(nTop):    
            results.append({labels[index[i]] : round(arTopProbas[i]*100.,2)})
               
        return results
        """
    return results


def predict(Frames, i3d_model, labels, nTop):

    results = []
    print("i will predict ")
    arProbas_i = i3d_model.predict(np.expand_dims(Frames, axis=0))

    arProbas = arProbas_i[0]
    indx = arProbas.argsort()[-nTop:][::-1]
    arTopProbas = arProbas[indx]
    print("finsih predicting")
    for i in range(nTop):    
        results.append({labels[indx[i]] : round(arTopProbas[i]*100.,2)})
    
    return results,arProbas_i



def sent_preds(rgbs,oflows,frames_count,labels,lstmModel,rgb_model,oflow_model,
    nTop,frames_to_process=30,stride=10,threshold=10):
    pos = 0
    results = []
    #print("rgbs.shape:",rgbs.shape)
    Next = 0
    while rgbs[Next:Next+stride].shape[0] != 0:

        rgbs_p = rgbs[pos * frames_to_process-(pos*stride):(pos+1)*frames_to_process-(pos*stride)]

        #print(f"from {pos*frames_to_process-(pos*stride)} to {(pos+1)*frames_to_process-(pos*stride)}")
        
        oflows_p = oflows[pos*frames_to_process-(pos*stride):(pos+1)*frames_to_process-(pos*stride)]
        #print("rgbs_p.shape:",rgbs_p.shape)
        #print(rgbs[pos*frames_to_process:pos*frames_to_process+stride])
        rgbs_p = frames_downsample(np.array(rgbs_p), 40)
        oflows_p = frames_downsample(np.array(oflows_p), 40)

        if lstmModel is not None:
            predictions = i3d_LSTM_prediction(rgbs_p, oflows_p, labels, lstmModel, rgb_model, oflow_model, nTop)
        else:
            predictions = get_predicts(rgbs_p, oflows_p, labels, oflow_model, rgb_model, nTop, False)

        #print("predictions:", predictions)
        tmp = [d for item in predictions for d in item.items()]
        
        if tmp[0][1] > threshold:
            keys = list(list(zip(*tmp))[0])
            vals = list(list(zip(*tmp))[1])
            if len(results) == 0:
                results.append(predictions)
            else:
                tmp_list = []
                added = False
                for result in results:
                    pred = [d for item in result for d in item.items()]
                    pred_keys = list(list(zip(*pred))[0])
                    pred_vals = list(list(zip(*pred))[1])
                    #if len([i for i,j in zip(pred_keys,keys) if i==j]) == 3:
                    #print("pred_keys[0]:",pred_keys[0])
                    #print("keys[0]:",keys[0])
                    if str(pred_keys[0]).strip() == str(keys[0]).strip() :
                        
                        avg = reduce(lambda x,y: x+y, vals) / len(vals)
                        pred_avg = reduce(lambda x,y: x+y, pred_vals) / len(pred_vals)
                        if avg > pred_avg:
                            results.remove(result)
                            #tmp_list.remove()
                            tmp_list.append(predictions)
                        added = True
                    
                if not added:
                    #print("no similar to this:",keys[0])
                    tmp_list.append(predictions)
                    added = True
                results.extend(tmp_list)
        
        Next = (pos+1)*frames_to_process-(pos*stride)
        pos += 1
        
        #print(f"is there from {Next} to {Next+stride}")
    

    print(results)
    def Phase(results):
        new_res = []
        key = [""] * 3
        val = [0] * 3
        idx = 0
        for result in results:
            for d in result:
                for i in d.items():
                    tmp_key,tmp_val = i
                    key[idx] += tmp_key + "-"
                    val[idx] += tmp_val
                    #print(key)
                    #print(val)
                idx += 1
                #print(idx)
            idx = 0
        
        for i in range(len(key)):
            new_res.append({key[i].rsplit("-",1)[0]:val[i]/len(results)})

            #print("new_res:", new_res)
        return new_res

    if len(results) > 0:
        predictions = Phase(results)
    else:
        predictions = [{'Unknown': 0.}, {'Unknown': 0.}, {'Unknown': 0.}]

    return predictions
    


"""
turk_labels = ["iki", "Tamam", "üç", "Kolay", "zor", "anlamak", "acıkmak", "dört", 
    "beş", "bir", "kötü", "tünel", "uçak", "onlar", "Duvar", "gitmek", "Tamam", "kötü", "durmak", "Kavşak"]

def turkish_classes():
    return dict(enumerate(list(map(lambda s: s.split(" ", 1)[1].strip(),
    open("turkishClasses.txt", encoding = "utf-8")))))
"""
