import numpy as np 
import json 
import datetime
import time 
import os.path as osp
import os
import glob
from util import *

#global variables
w = 1280
h= 720


class getDetectionBot(object):

    def genSimRes(self):
        bo = [np.random.rand(1)[0]/20 for _ in range(4)]
        box = [(0.21+bo[1])*w, (0.26+bo[3])*w, (0.59+bo[0])*h, (0.73+bo[2])*h]
        res={}
        res["label"] = "Standing person"
        res["box"] = box
        res["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return res 



    def genRes(self):
        bo = [np.random.rand(1)[0] for _ in range(4)]
        box = [bo[1]*w, bo[3]*w, bo[0]*h, bo[2]*h]
        res={}
        res["label"] = "Standing person"
        res["box"] = box
        res["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return res 

    def getDetections(self):
        while True:
            num = np.random.randint(0,8)
            output = {}
            for i in range(num):
                res = self.genSimRes()
                output[str(i)] = res
            result_boxes =filt(output,num)
            time_stamp = datetime.datetime.now().strftime("%S:%f")
            jsonFileName = time_stamp+"result.json"
            jsonFilePath = osp.join("./res",jsonFileName)
            with open(jsonFilePath,'w') as f:
                json.dump(result_boxes,f)
            time.sleep(1)
            #detection_result = self.parseResult()
            #print(detection_result)
           # print("detection_result count is {}".format(len(detection_result)))

    def getDetectionLocal(self):
        while True:
            num = np.random.randint(0,3)
            output = {}
            for i in range(num):
                res = self.genSimRes()
                output[str(i)] = res
            result_boxes = filt(output,num)
            time.sleep(1)
            return result_boxes 

        
    def parseResult(self):
        resultJsonList = glob.glob("./res/*.json")
        print(resultJsonList)
        result = []
        for resultJson in resultJsonList:
            with open(resultJson) as json_file:
                result.append(json.load(json_file))
            rm = "rm {}".format(resultJson)
            os.system(rm)
        return result

    def printResult(self):
        while True:
            print(self.parseResult())

if __name__ == "__main__":
    jsonFileName = 'detectionResult.json'
    bot = getDetectionBot()
    #get_thread = threading.Thread(target=bot.getDetections)
    #get_thread.start()
    bot.getDetections()
