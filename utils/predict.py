import os
import glob
import sys
import warnings
import time
from multiprocessing import Process, Manager
import numpy as np
import pandas as pd

from datagenerator import FeaturesGenerator
from frame import frames_downsample



ret_dict = Manager().dict()


def i3d_LSTM_prediction(rgbFrames, oflowFrames, labels, LSTM_model, rgb_model, oflow_model, nTop, mul_2stream):
    
    rgbProbas = rgb_model.predict(np.expand_dims(rgbFrames, axis=0))
    oflowProbas = oflow_model.predict(np.expand_dims(oflowFrames, axis=0))

    LSTM_input = np.concatenate((rgbProbas,oflowProbas), axis=-1)

    arProbas_i = LSTM_model.predict(LSTM_input)
    arProbas = arProbas_i[0]
    indx = arProbas.argsort()[-nTop:][::-1]
    arTopProbas = arProbas[indx]
    
    results = []

    for i in range(nTop):
        results.append({labels[indx[i]] : round(arTopProbas[i]*100.,2)})

    return results


def get_predicts(rgbFrames, oflowFrames, labels, oflow_model, rgb_model, nTop, mul_2stream):
    import keras
    oflow_arProbas = []
    rgb_arProbas = []
    results = []
    jobs = []
    if oflow_model is not None:
        if rgb_model is None:
            results,_ = predict(oflowFrames,oflow_model,labels,nTop,mul_2stream)
            return results
        else:
            if mul_2stream:
                p1 = Process(target=predict, args=(oflowFrames,oflow_model,labels,nTop,mul_2stream))
                p1.start()   
                jobs.append(p1)
                p1.join()
            else:
                _,oflow_arProbas = predict(oflowFrames,oflow_model,labels,nTop,mul_2stream)
    if rgb_model is not None:
        if oflow_model is None:
            results,_ = predict(rgbFrames, rgb_model,labels,nTop,mul_2stream)
            return results
        else:
            if mul_2stream:
                p2 = Process(target=predict, args=(rgbFrames,rgb_model,labels,nTop,mul_2stream))
                p2.start()   
                jobs.append(p2)
            else:
                _,rgb_arProbas = predict(rgbFrames, rgb_model,labels,nTop,mul_2stream)

    if oflow_model is not None and rgb_model is not None:
        if mul_2stream:
            print("2 process are working")
            for p in jobs:
                p.join()
            oflow_arProbas = ret_dict["oflow"]
            rgb_arProbas = ret_dict["rgb"]
        print("we did it !!")
        arProbas = np.concatenate((oflow_arProbas,rgb_arProbas), axis=0)
        arProbas = np.mean(arProbas, axis=0)
        index = arProbas.argsort()[-nTop:][::-1]
        arTopProbas = arProbas[index]

        for i in range(nTop):    
            results.append({labels[index[i]] : round(arTopProbas[i]*100.,2)})

        return results
    else:
        raise ValueError("[Prediction Error]: No model found.")

def predict(Frames, i3d_model, labels, nTop,mul_2stream):

    results = []
    print("i will predict ")
    try:
        arProbas_i = i3d_model.predict(np.expand_dims(Frames, axis=0))
    except Exception as e:
        print(e)

    arProbas = arProbas_i[0]
    indx = arProbas.argsort()[-nTop:][::-1]
    arTopProbas = arProbas[indx]
    print("finsih predicting")
    for i in range(nTop):    
        results.append({labels[indx[i]] : round(arTopProbas[i]*100.,2)})

    if mul_2stream:
        if Frames.shape[-1] == 2:
            ret_dict["oflow"] = arProbas_i
        else:
            ret_dict["rgb"] = arProbas_i
    else:
        return results,arProbas_i



def sent_preds(rgbs,oflows,frames_count,labels,lstmModel,rgb_model,oflow_model,
    nTop,mul_2stream,frames_to_process=30,stride=10,threshold=40):
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
            predictions = get_predicts(rgbs_p, oflows_p, labels, oflow_model, rgb_model, nTop)

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

    return Phase(results)
    


"""
turk_labels = ["iki", "Tamam", "üç", "Kolay", "zor", "anlamak", "acıkmak", "dört", 
    "beş", "bir", "kötü", "tünel", "uçak", "onlar", "Duvar", "gitmek", "Tamam", "kötü", "durmak", "Kavşak"]

def turkish_classes():
    return dict(enumerate(list(map(lambda s: s.split(" ", 1)[1].strip(),
    open("turkishClasses.txt", encoding = "utf-8")))))
"""
