import pickle
from threading import Thread
import sqlite3
import numpy as np

from PIL import Image

from konlpy.tag import Okt
from flask import Flask
from slack import WebClient
from slackeventsapi import SlackEventAdapter
from scipy.sparse import lil_matrix


# slack 연동 정보 입력 부분
SLACK_TOKEN = "xoxb-731166603924-741441211334-Utv5GC3CFUPRnGA30LiKqLXk"
SLACK_SIGNING_SECRET = "a7569a13748b1861460d8d41db1a24be"

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

BADURL = Image.open('./img/BadOmpangi.gif')
GOODURL = Image.open('./img/GoodOmpangi.gif')

import json
send_data = {
    "attachments": [
        {
            
            "imgPath": "img/prods/levisbluedemin01.jpg"


        }
    ]
}
json_data = json.dumps(send_data)

# # Req 2-2-3. 긍정 혹은 부정으로 분류
def classify(doc):
    predict_result = clf.predict(preprocess(doc))[0]
    if predict_result == 0.0:
        return json_data
    else:
        return json_data
# print(classify("이 영화 정말 재밌다."))
    
# # Req 2-2-4. app.db 를 연동하여 웹에서 주고받는 데이터를 DB로 저장


# 챗봇이 멘션을 받았을 경우
@slack_events_adaptor.on("app_mention")
def app_mentioned(event_data):
    channel = event_data["event"]["channel"]
    text = event_data["event"]["text"]
    # keywords = "hello"
    keywords = classify(text.split(">")[1])
        
    # slack_web_client.chat_postMessage(
    #     channel=channel,
    #     text=keywords
    # )
    print(keywords)
    response = slack_web_client.files_upload(
        channels=channel,
        file="GoodOmpangi.gif"
    )
    assert response["ok"]

@app.route("/", methods=["GET"])
def index():
    return "<h1>Server is ready.</h1>"

if __name__ == '__main__':
    app.run()