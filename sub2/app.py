import pickle
from threading import Thread
import sqlite3
import numpy as np
import time
import json


from konlpy.tag import Okt
from flask import Flask, request, make_response, Response
from slack import WebClient
from slackeventsapi import SlackEventAdapter
from scipy.sparse import lil_matrix

import requests
from slack.web.classes import extract_json
from slack.web.classes.blocks import *
from slack.web.classes.elements import *
from slack.web.classes.interactions import MessageInteractiveEvent

from scipy.sparse import lil_matrix
from retrain import read_data, tokenize
from sklearn.naive_bayes import MultinomialNB
from sklearn import linear_model


# slack 연동 정보 입력 부분
SLACK_TOKEN = "xoxb-731614402629-733495701111-6QglObMVmrUpPNSJz4bob0Vo"
SLACK_SIGNING_SECRET = "33d0b00dfeb6ab2a156b78392ccb01b1"

app = Flask(__name__)

slack_events_adaptor = SlackEventAdapter(SLACK_SIGNING_SECRET, "/listening", app)
slack_web_client = WebClient(token=SLACK_TOKEN)

# Req 2-2-1. pickle로 저장된 model.clf 파일 불러오기
pickle_obj = open('model.clf', 'rb')
clf = pickle.load(pickle_obj) # naive bayes
clf2 = pickle.load(pickle_obj) # Logistic Regression
clf3 = pickle.load(pickle_obj) # SVM
word_indices = pickle.load(pickle_obj)
clf4 = pickle.load(pickle_obj) # 의사결정트리

neg = 0
pos = 0
msg = ""

# Req 2-2-2. 토큰화 및 one-hot 임베딩하는 전 처리
pos_tagger = Okt()

def preprocess(doc):
    # 토큰화
    text_docs = []
    result = []
    result = ['/'.join(t) for t in pos_tagger.pos(doc, norm=True, stem=True)]
    text_docs += [result]
    #one-hot 임베딩
    for idx in range(len(text_docs)):
        temp = [0]*len(word_indices)
        for verb in text_docs[idx]:
            part = verb.split('/')[0]
            if word_indices.get(part)!=None:
                temp[word_indices[part]]=1
        text_docs[idx]=temp

    return text_docs


def predict(test_doc, test_clf):
    predict_result = test_clf.predict(test_doc)[0]
    return predict_result

# # Req 2-2-3. 긍정 혹은 부정으로 분류
def classify(test_doc, test_clf):
    global neg
    global pos
    predict_result = predict(test_doc, test_clf)
    if predict_result == 0.0:
        neg += 1
        return "negative"
    else:
        pos += 1
        return "positive"

# # Req 2-2-4. app.db 를 연동하여 웹에서 주고받는 데이터를 DB로 저장
def save_text_to_db(text):
    # db에 저장
    con = sqlite3.connect('./app.db')
    cur = con.cursor()

    msg = text.split("> ")[1]
    cur.execute('INSERT INTO search_history(query) VALUES(?)', (msg,))
    con.commit()
    cur.close()

# 결과값이 틀린 경우 데이터를 DB에 저장
def add_data(message):
    chk = True
    ## db저장 구현

    return chk

# 추가 데이터 트레이닝
def data_training():
    chk = True
    # DB에 저장된 데이터 개수 확인
    # DB에 데이터가 10개 미만일 경우 chk -> false


    # DB에 데이터가 10개 이상일 경우 chk -> true
    # 추가 데이터 트레이닝
    # DB 데이터 삭제
    """
    train_data = read_date()
    train_docs = tokenize(train_data)

    X = lil_matrix((len(train_docs), len(word_indices)))
    Y = np.zeros(len(train_docs))

    for idx in range(len(train_docs)):
        temp = [0]*len(word_indices)
        for verb in train_docs[idx]:
            part = verb.split('/')[0]
            if word_indices.get(part)!=None:
                temp[word_indices[part]]=1
        X[idx]=temp

    for idx in range(len(train_data)):
        part = train_data[idx][2].split('\n')[0]
        Y[idx]=part
    
    clf.partial_fit(X, Y) # naive Bayes
    clf2.partial_fit(X, Y) # Logistic
    clf3.partial_fit(X, Y) # SVM
    """
    return chk

def send_message(text, ch):
    global neg, pos
    
    test_doc = preprocess(text.split("> ")[1])
    predict_NB = classify(test_doc, clf)
    predict_LR = classify(test_doc, clf2)
    
    if neg > pos:
        result = "negative"
        img = "https://i.pinimg.com/originals/2c/21/8f/2c218fa1247ce35d20cb618e9f3049d4.gif"
    else:
        result = "positive"
        img = "https://img1.daumcdn.net/thumb/R800x0/?scode=mtistory2&fname=https%3A%2F%2Ft1.daumcdn.net%2Fcfile%2Ftistory%2F99A4654C5C63B09028"

    neg = 0
    pos = 0

    attachement = {
            "color": "#fe6f5e",
            "image_url": img,
            "title": "RESULT",
            'pretext': text.split("> ")[1],
            "fallback": "Status Monitor",
            "callback_id": "button_event",
            "text": result,
            "fields":[
                {
                    "title": "Naive baysian model",
                    "value": predict_NB,
                    "short": True
                },
                {
                    "title": "Logistic regresion model",
                    "value": predict_LR,
                    "short": True
                },
                {

                }
            ],
            "actions": [
                {
                    "name": "edit",
                    "text": "EDIT",
                    "type": "button",
                    "value": "edit",
                    "style": "danger"
                },
                {
                    "name": "trainig",
                    "text": "TRAINING",
                    "type": "button",
                    "value": "training",
                    "style": "danger"
                }
            ],
        }
    slack_web_client.chat_postMessage(
        channel=ch, 
        text=None,
        attachments=[attachement], 
        as_user=False)

@app.route("/click", methods=["GET", "POST"])
def on_button_click():
    payload = request.values["payload"]
    clicked = json.loads(payload)["actions"][0]['value']
    my_ch = json.loads(payload)["channel"]["id"]
    
    if clicked == "edit":
        print("edit")
        add_data(msg)
    elif clicked == "training":
        print("train")
        if data_training():
            print("Success Training")
        else:
            print("Save more Data")
    
    slack_web_client.chat_postMessage(
        channel=my_ch,
        # channel = test,
        text = clicked
        # blocks=extract_json(message_blocks)
    )
    return make_response("", 200)

# 챗봇이 멘션을 받았을 경우
@slack_events_adaptor.on("app_mention")
def app_mentioned(event_data):
    global msg
    channel = event_data["event"]["channel"]
    text = event_data["event"]["text"]
    msg = text.split("> ")[1]
    # DB에 데이터 저장
    save_text_to_db(text)
    # 메세지 보내기
    send_message(text, channel)

@app.route("/", methods=["GET"])
def index():
    return "<h1>Server is ready.</h1>"


if __name__ == '__main__':
    app.run()