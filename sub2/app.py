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
SLACK_TOKEN = "xoxb-720220358483-738701955364-q3tCkTnPKzSFEQbW2a8vnrWm"
SLACK_SIGNING_SECRET = "61d52b36a564138f59046147325dcfe4"

app = Flask(__name__)

slack_events_adaptor = SlackEventAdapter(
    SLACK_SIGNING_SECRET, "/listening", app)
slack_web_client = WebClient(token=SLACK_TOKEN)

# Req 2-2-1. pickle로 저장된 model.clf 파일 불러오기
pickle_obj = open('origin_model.clf', 'rb')
clf = pickle.load(pickle_obj) # naive bayes
clf2 = pickle.load(pickle_obj) # Logistic Regression
clf3 = pickle.load(pickle_obj) # SVM
clf4 = pickle.load(pickle_obj) # 의사결정트리
word_indices = pickle.load(pickle_obj)


neg = 0
pos = 0
msg = ""

output = -1  # 트레이닝결과값 0/1을 db에 쓰기위한 초기화 변수
beforeTrainDataCnt = 0  # 까지 학습했다.를 저장하는 전역변수로 사용

# Req 2-2-2. 토큰화 및 one-hot 임베딩하는 전 처리
pos_tagger = Okt()


def preprocess(doc):
    # 토큰화
    text_docs = []
    result = []
    result = ['/'.join(t) for t in pos_tagger.pos(doc, norm=True, stem=True)]
    text_docs += [result]
    # one-hot 임베딩
    for idx in range(len(text_docs)):
        temp = [0]*len(word_indices)
        for verb in text_docs[idx]:
            part = verb.split('/')[0]
            if word_indices.get(part) != None:
                temp[word_indices[part]] = 1
        text_docs[idx] = temp

    return text_docs

def n_preprocess(doc, word_indices):
    # 토큰화
    text_docs = []
    result = []
    result = ['/'.join(t) for t in pos_tagger.pos(doc, norm=True, stem=True)]
    text_docs += [result]
    # one-hot 임베딩
    for idx in range(len(text_docs)):
        temp = [0]*len(word_indices)
        for verb in text_docs[idx]:
            part = verb.split('/')[0]
            if word_indices.get(part) != None:
                temp[word_indices[part]] = 1
        text_docs[idx] = temp

    return text_docs


def predict(test_doc, test_clf):
    predict_result = test_clf.predict(test_doc)[0]
    return predict_result

# # Req 2-2-3. 긍정 혹은 부정으로 분류


def classify(test_doc, test_clf, model_name):
    global neg
    global pos
    
    predict_result = predict(test_doc, test_clf)
    if predict_result == 0.0:
        if model_name=="NB" or model_name=="DTC":
            print(test_clf.predict_proba(test_doc))
            neg += test_clf.predict_proba(test_doc)[0][0]
            pos += test_clf.predict_proba(test_doc)[0][1]
        else:
            print(test_clf.decision_function(test_doc))
            neg += 0.7
            pos += 0.3
        return "negative"
    else:
        if model_name=="NB" or model_name=="DTC":
            print(test_clf.predict_proba(test_doc))
            neg += test_clf.predict_proba(test_doc)[0][0]
            pos += test_clf.predict_proba(test_doc)[0][1]
        else:
            print(test_clf.decision_function(test_doc))
            pos += 0.7
            neg += 0.3
        return "positive"

# # Req 2-2-4. app.db 를 연동하여 웹에서 주고받는 데이터를 DB로 저장


def save_text_to_db(text):
    # db에 저장
    global output

    con = sqlite3.connect('./app.db')
    cur = con.cursor()

    msg = text.split("> ")[1]
    cur.execute(
        'INSERT INTO search_history(question, answer) VALUES(?,?)', (msg, output,))
    con.commit()

    output = -1
    cur.close()


# 결과값이 틀린 경우 데이터를 DB에 저장
def edit_data():
    chk = False
    con = sqlite3.connect('./app.db')
    cur = con.cursor()

    recent = cur.execute(
        'SELECT max(id), answer FROM search_history')
    idx = 0
    answer = -1
    for row in recent:
        idx = row[0]
        answer = row[1]

    if(answer == 0):
        cur.execute('UPDATE search_history SET answer = 1 WHERE id = %s' % idx)
        print("1로 수정완료")
        chk = True
    elif(answer == 1):
        cur.execute('UPDATE search_history SET answer = 0 WHERE id = %s ' % idx)
        print("0으로 수정완료")
        chk = True
    else:
        print("errer")

    con.commit()
    cur.close()

    # db 업데이트 성공유무를 리턴
    return chk


# 추가 데이터 트레이닝
def data_training():
    global beforeTrainDataCnt
    global clf1, clf2, clf3, clf4, word_indices
    chk = True

    con = sqlite3.connect('./app.db')
    cur = con.cursor()
    # DB에 저장된 데이터 개수 확인
    recent = cur.execute(
        'SELECT max(id), answer FROM search_history')

    cnt = 0
    for row in recent:
        cnt = row[0]

    # DB에 데이터가 10개 미만일 경우 chk -> false
    if((cnt - beforeTrainDataCnt) < 10):
        print("추가로 저장된 데이터가 10개 미만입니다")
        chk = False

    # DB에 데이터가 10개 이상일 경우 chk -> true
    else:
        print("학습하겠습니다")
        print(beforeTrainDataCnt+1, " 부터 학습합니다")

        f = open('retrain.txt', mode='wt', encoding='utf-8')
        readData = cur.execute(
            'SELECT * FROM search_history WHERE id > %s ' % beforeTrainDataCnt
        )
        temp = ""
        for data in readData:
            row = str(data[1]) + "\t" + data[2] + "\t" + str(data[3]) + "\n"
            temp += row

        f.write(temp)
        print("쓰기완료")
        f.close()

        beforeTrainDataCnt = cnt  # 현재 데이터까지 학습했음을 저장
        print(beforeTrainDataCnt, " idx 까지 학습완료")
        chk = True
        
        # 추가 데이터 트레이닝
        train_data = read_data('retrain.txt')
        train_docs = tokenize(train_data)
        print('read_data, tokenize')

        X = lil_matrix((len(train_docs), len(word_indices)))
        Y = np.zeros(len(train_docs))

        print('X, Y init')
        for idx in range(len(train_docs)):
            temp = [0]*len(word_indices)
            for verb in train_docs[idx]:
                part = verb.split('/')[0]
                if word_indices.get(part)!=None:
                    temp[word_indices[part]]=1
            X[idx]=temp
        print('X one hot embedding ')
        for idx in range(len(train_data)):
            part = train_data[idx][2].split('\n')[0]
            Y[idx]=part
        print('Y label')
        clf.partial_fit(X, Y) # naive Bayes
        print('naive')
        clf2.partial_fit(X, Y) # Logistic
        print('logistic')
        clf3.partial_fit(X, Y) # SVM
        print('SVM')

        fl = open('origin_model.clf', 'wb')
        pickle.dump(clf, fl)
        pickle.dump(clf2, fl)
        pickle.dump(clf3, fl)
        pickle.dump(clf4, fl)
        pickle.dump(word_indices, fl)
        fl.close()
    cur.close()
   
    # DB 데이터 삭제
    return chk


def send_message(text, ch):
    global neg, pos
    global output

    test_doc = preprocess(text.split("> ")[1])
    predict_NB = classify(test_doc, clf, "NB")
    predict_LR = classify(test_doc, clf2, "LR")
    predict_SVM = classify(test_doc, clf3, "SVM")
    predict_DTC = classify(test_doc, clf4, "DTC")

    if neg > pos:
        result = "negative"
        img = "https://imgur.com/L3ZrqYS.gif"
        output = 0
    else:
        result = "positive"
        img = "https://imgur.com/oLCYpsU.gif"
        output = 1

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
                    "title": "Naive Baysian model",
                    "value": predict_NB,
                    "short": True
                },
                {
                    "title": "Logistic Regresion model",
                    "value": predict_LR,
                    "short": True
                },
                {
                    "title": "Support Vector Machine model",
                    "value": predict_SVM,
                    "short": True
                },
                {
                    "title": "Decision Tree Classifier model",
                    "value": predict_DTC,
                    "short": True
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
                },
                {
                    "name": "naver",
                    "text": "NAVER SHOW",
                    "type": "button",
                    "value": "naver",
                    "style": "danger"
                }
            ],
        }
    slack_web_client.chat_postMessage(
        channel=ch,
        text=None,
        attachments=[attachement],
        as_user=False)

def send_naver_message(ch):
    global neg, pos
    global output
    global msg

    pickle_obj = open('naver_model.clf', 'rb')
    n_clf = pickle.load(pickle_obj) # naive bayes
    n_clf2 = pickle.load(pickle_obj) # Logistic Regression
    n_clf3 = pickle.load(pickle_obj) # SVM
    n_clf4 = pickle.load(pickle_obj) # 의사결정트리
    n_word_indices = pickle.load(pickle_obj)
    print("model create")
    print(msg)
    test_doc = n_preprocess(msg.split("> ")[1], n_word_indices)
    predict_NB = classify(test_doc, n_clf, "NB")
    predict_LR = classify(test_doc, n_clf2, "LR")
    predict_SVM = classify(test_doc, n_clf3, "SVM")
    predict_DTC = classify(test_doc, n_clf4, "DTC")
    print('predict')
    if neg > pos:
        result = "negative"
        img = "https://i.pinimg.com/originals/2c/21/8f/2c218fa1247ce35d20cb618e9f3049d4.gif"
        output = 0
    else:
        result = "positive"
        img = "https://img1.daumcdn.net/thumb/R800x0/?scode=mtistory2&fname=https%3A%2F%2Ft1.daumcdn.net%2Fcfile%2Ftistory%2F99A4654C5C63B09028"
        output = 1

    neg = 0
    pos = 0

    attachement = {
            "color": "#fe6f5e",
            "title": "NAVER RESULT",
            'pretext': msg.split("> ")[1],
            "fallback": "Status Monitor",
            "callback_id": "button_event",
            "text": result,
            "fields":[
                {
                    "title": "Naive Baysian model",
                    "value": predict_NB,
                    "short": True
                },
                {
                    "title": "Logistic Regresion model",
                    "value": predict_LR,
                    "short": True
                },
                {
                    "title": "Support Vector Machine model",
                    "value": predict_SVM,
                    "short": True
                },
                {
                    "title": "Decision Tree Classifier model",
                    "value": predict_DTC,
                    "short": True
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
        edit_data()
    elif clicked == "training":
        print("train")
        if data_training():
            print("Success Training")
        else:
            print("Save more Data")
    else:
        print("naver start")
        send_naver_message(my_ch)
        print("sending naver")
        return make_response("", 200)

    slack_web_client.chat_postMessage(
        channel=my_ch,
        # channel = test,
        text=clicked
        # blocks=extract_json(message_blocks)
    )
    return make_response("", 200)

# 챗봇이 멘션을 받았을 경우
@slack_events_adaptor.on("app_mention")
def app_mentioned(event_data):
    global msg
    retry_reason = request.headers.get("x-slack-retry-reason")
    retry_count = request.headers.get("x-slack-retry-num")
    if retry_count:
        return make_response('No', 200, {"X-Slack-No-Retry": 1})
    else:
        channel = event_data["event"]["channel"]
        text = event_data["event"]["text"]
        msg = text
        # DB에 데이터 저장
        # 메세지 보내기
        send_message(text, channel)
        save_text_to_db(text)
    make_response('No', 200, {"X-Slack-No-Retry": 1})
    

@app.route("/", methods=["GET"])
def index():
    return "<h1>Server is ready.</h1>"


if __name__ == '__main__':
    app.run()
