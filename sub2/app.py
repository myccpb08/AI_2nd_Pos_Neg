import pickle
from threading import Thread
import sqlite3
import numpy as np
import time

from konlpy.tag import Okt
from flask import Flask
from slack import WebClient
from slackeventsapi import SlackEventAdapter
from scipy.sparse import lil_matrix

# slack 연동 정보 입력 부분
SLACK_TOKEN = "xoxb-720220358483-738701955364-q3tCkTnPKzSFEQbW2a8vnrWm"
SLACK_SIGNING_SECRET = "61d52b36a564138f59046147325dcfe4"

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
    # DB에 데이터 저장
    save_text_to_db(text)
    # 메세지 보내기
    send_message(text, channel)

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
    # assert response["ok"]
    print(text.split("> ")[1])
    keyword = classify(text.split("> ")[1])

    slack_web_client.chat_postMessage(
        channel=ch,
        text=keyword
    )


@app.route("/", methods=["GET"])
def index():
    return "<h1>Server is ready.</h1>"


if __name__ == '__main__':
    app.run()
