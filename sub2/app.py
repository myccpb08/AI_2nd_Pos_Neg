import pickle
from threading import Thread
import sqlite3
import numpy as np
import time

from konlpy.tag import Okt
from flask import Flask, request, make_response, Response
from slack import WebClient
from slackeventsapi import SlackEventAdapter
from scipy.sparse import lil_matrix

import pprint
import json
# slack 연동 정보 입력 부분
SLACK_TOKEN = "xoxb-731614402629-733495701111-6QglObMVmrUpPNSJz4bob0Vo"
SLACK_SIGNING_SECRET = "33d0b00dfeb6ab2a156b78392ccb01b1"

app = Flask(__name__)

slack_events_adaptor = SlackEventAdapter(SLACK_SIGNING_SECRET, "/listening", app)
slack_web_client = WebClient(token=SLACK_TOKEN)

# Req 2-2-1. pickle로 저장된 model.clf 파일 불러오기
pickle_obj = open('model.clf', 'rb')

clf = pickle.load(pickle_obj)
clf2 = pickle.load(pickle_obj)
word_indices = pickle.load(pickle_obj)

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

# # Req 2-2-3. 긍정 혹은 부정으로 분류
def classify(doc):
    text_docs = preprocess(doc)
    predict_result = clf.predict(text_docs)[0]
    print(clf.predict(text_docs)[0])
    if predict_result == 0.0:
        return "negative"
    else:
        return "positive"
    
# # Req 2-2-4. app.db 를 연동하여 웹에서 주고받는 데이터를 DB로 저장


# 챗봇이 멘션을 받았을 경우
@slack_events_adaptor.on("app_mention")
def app_mentioned(event_data):
    channel = event_data["event"]["channel"]
    text = event_data["event"]["text"]
    pprint.pprint(event_data)
    # DB에 데이터 저장
    # save_text_to_db(text)
    # 메세지 보내기
    send_message(text, channel)

# @slack_events_adaptor.on("reaction_added")
# def reaction_added(event_data):
#   emoji = event_data["event"]["reaction"]
#   print(emoji)
@app.route("/click", methods=["POST"])
def message_options():
    form_json = json.loads(request.form["payload"])
    ch = form_json["channel"]["id"]
    pprint.pprint(form_json)
    keyword = form_json["actions"][0]["value"]
    #selected_option = form_json['actions'][0]["selected_option"]["text"]["text"]
    slack_web_client.chat_postMessage(
        channel=ch,
        text=keyword,
    )
    return make_response("", 200)

def save_text_to_db(text):
    # db에 저장
    con = sqlite3.connect('./app.db')
    cur = con.cursor()

    msg = text.split("> ")[1]
    # print(msg)
    cur.execute('INSERT INTO search_history(query) VALUES(?)', (msg,))
    con.commit()
    cur.close()

def send_message(text, ch):
    # response = slack_web_client.files_upload(
    #     channels=ch,
    #     file="GoodOmpangi.gif"
    # )
    # assert response["ok"]dd
    
    print(text.split("> ")[1])
    keyword = classify(text.split("> ")[1])
    
    attachement = {
            "color": "#fe6f5e",
            "image_url":"https://img1.daumcdn.net/thumb/R800x0/?scode=mtistory2&fname=https%3A%2F%2Ft1.daumcdn.net%2Fcfile%2Ftistory%2F99A4654C5C63B09028",
            "title": "RESULT",
            'pretext': text.split("> ")[1],
            "fallback": "Status Monitor",
            "callback_id": "button_event",
            "text": keyword,
            "fields":[
                {
                    "title": "Naive baysian model",
                    "value": "predict_NB",
                    "short": True
                },
                {
                    "title": "Logistic regresion model",
                    "value": "predict_LR",
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
                    "style": "good"
                },
                {
                    "name": "trainig",
                    "text": "TRAINING",
                    "type": "button",
                    "value": "training",
                    "style": "good"
                },
                {
                    "name": "close",
                    "text": "CLOSE",
                    "type": "button",
                    "value": "close",
                    "style": "danger"
                }
            ],
        }


    print('시작')
    slack_web_client.chat_postMessage(
        channel=ch,
        text=keyword,
        attachments=[attachement],
        blocks = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "You can add a button alongside text in your message. "
            },
            "accessory": {
                "type": "button",
                "text": {
                    "type": "plain_text",
                    "text": "Button",
                    "emoji": True
                },
                "value": "click_me_123"
            }
        }
        ]
    )
    print('끝')


@app.route("/", methods=["GET"])
def index():
    return "<h1>Server is ready.</h1>"


if __name__ == '__main__':
    app.run()
