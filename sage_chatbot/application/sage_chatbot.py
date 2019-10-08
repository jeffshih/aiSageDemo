from flask import Flask, request, abort
from bot_config import CAT, Channel_secret 
import reply_text
import json 
#from getDetectionBot import getDetectionBot 
import threading
import time
import glob
import os 
from flask_sqlalchemy import SQLAlchemy 
from util import *

baseDir = os.path.abspath(os.path.dirname(__file__))
from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage, FollowEvent
)


detection_result = {}
app = Flask(__name__)
line_bot_api = LineBotApi(CAT)
handler = WebhookHandler(Channel_secret)
register = set()
#db = SQLAlchemy(app)
#global accountDB
accountDB = {}


def addLocalDB(user_id):
    if user_id not in accountDB:
        accountDB[user_id] = False

def updateLocalDB():
    localDB = open("./db/localDb.json",'w')
    json.dump(accountDB,localDB)
    localDB.close()

def deleteLocalDB(user_id):
    if user_id in accountDB:
        del accountDB[user_id]

def readLocalDB():
    tmp = {}
    with open("./db/localDb.json") as dbjs:
        tmp = json.load(dbjs)
    return tmp

def checkUser(user_id):
    return user_id in accountDB

def checkUserStatus(user_id):
    if checkUser(user_id):
        return accountDB[user_id]
    else:
        return False

def updateUserStatus(user_id,flag):
    if checkUser(user_id):
        accountDB[user_id] = flag 

def pushMessage():
    def parseResult(jsonFile):
        with open(jsonFile) as json_file:
            result = json.load(json_file)
        return result
    while True: 
        resultJsonList = glob.glob("./res/*.json")
        dets = []
        box2filt = {}
        for jsonFile in resultJsonList:
            dets.append(parseResult(jsonFile))
            rm = "rm {}".format(jsonFile) 
            os.system(rm)
        i = 0
        for det in dets:
            for idx, res in det.items():
                box2filt[str(i)] = res
                i+=1
        result = filt(box2filt, len(box2filt))
        detsCnt = len(result)
        registerCnt = len(accountDB)
        #print("account db {}".format(accountDB))
        #print("dets {}".format(dets))
        if registerCnt == 0 or detsCnt == 0:
            time.sleep(1)
            continue
        date = dets[0]["0"]["timestamp"]
        for ID, mode in accountDB.items():
           # print(ID, mode)
           # print(type(mode))
           # print(mode=="True")
            if mode == True:
                line_bot_api.push_message(ID,TextSendMessage(text="在{}時偵測到{}個站立".format(date,detsCnt)))
        updateLocalDB()
        time.sleep(1)

#def pushMessageLocal():
#    bot = getDetectionBot() 
#    while True: 
#        dets = bot.getDetectionLocal()
#        detsCnt = len(dets)
#        registerCnt = len(register)
#        if registerCnt == 0 or detsCnt == 0:
#            continue
#        for usr in register:
#            line_bot_api.push_message(usr,TextSendMessage(text="偵測到{}個站立".format(detsCnt)))
#

@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return 'OK'

@app.route("/hello",methods=['GET'])
def hello():
    return "Hello, line chatbot"




@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    #print(event.source.user_id)
    #line_bot_api.reply_message(event.reply_token, TextSendMessage(text="ABC"))
    text2reply = []
    msg = event.message.text 
    user_id = event.source.user_id
    if  msg == "註冊":
        if(checkUser(user_id)):
            text2reply.append(TextSendMessage(text="您已為註冊用戶"))
        else:
            text2reply.append(TextSendMessage(text="已將您加入註冊用戶"))
            addLocalDB(user_id)         
    elif msg == "移除":
        if(checkUser(user_id)):
            text2reply.append(TextSendMessage(text="已將您移除註冊"))
            deleteLocalDB(user_id)
        else:
            text2reply.append(TextSendMessage(reply_text.notReg))
 
    if "開啟" in msg:
        if(checkUser(user_id)):
            if checkUserStatus(user_id):
                text2reply.append(TextSendMessage(reply_text.isOpen))
            else:
                updateUserStatus(user_id,True)
                text2reply.append(TextSendMessage(text="功能已啟用"))
        else:
            text2reply.append(TextSendMessage(reply_text.notReg))
    elif "停止" in msg:
        if(checkUser(user_id)):
            if checkUserStatus(user_id):
                text2reply.append(TextSendMessage(text="功能已關閉"))
                updateUserStatus(user_id,False)
            else:
                text2reply.append(TextSendMessage(reply_text.isClosed))
        else:
            text2reply.append(TextSendMessage(reply_text.notReg))
    elif msg == "Hi" or msg == "hi":
        text2reply.append(TextSendMessage(text=reply_text.Hi))
        text2reply.append(TextSendMessage(reply_text.intro))
    elif msg == "介紹":
        text2reply.append(TextSendMessage(reply_text.helpMessage2))
        text2reply.append(TextSendMessage(reply_text.helpMessage3))
        text2reply.append(TextSendMessage(reply_text.helpMessage))
    elif "笑話" in msg or "joke" in msg:
        text2reply.append(TextSendMessage(reply_text.joke))
    elif msg == "我的狀態" :
        if(checkUser(user_id)):
            if checkUserStatus(user_id):
                text2reply.append(TextSendMessage(text="您已註冊，目前功能為開啟"))
            else:
                text2reply.append(TextSendMessage(text="您已註冊，目前功能為關閉"))
    elif len(text2reply)==0:
        text2reply.append(TextSendMessage(text=reply_text.generalResponse))
        text2reply.append(TextSendMessage(reply_text.intro))
    
    if len(text2reply)>0:
        line_bot_api.reply_message(event.reply_token, messages=text2reply)
    #line_bot_api.push_message("Ucd82fda579561396b75c340fe72e1342",TextSendMessage(text="安安"))
    #line_bot_api.push_message("U253c8bf25a08de15f97f7c47472ea840",TextSendMessage(text="安安"))

@handler.add(FollowEvent)
def handle_follow(event):
    text2reply = []
    app.logger.info("Got Follow Event:" + event.source.user_id)
    text2reply.append(TextSendMessage(reply_text.greeting))
    text2reply.append(TextSendMessage(reply_text.intro))
    line_bot_api.reply_message(event.reply_token,messages=text2reply)

if __name__ == "__main__":
    accountDB = readLocalDB()
#    detBot = getDetectionBot.getDetectionBot()
#    det_thread = mp.Process(target=detBot.getDetections,args=(jsonFileName,))
#    det_thread.start()
    push_thread = threading.Thread(target=pushMessage)
    push_thread.start()
    #app_thread = threading.Thread(target=app.run)
    #app_thread.start()
    #manager.run()
    app.run()
