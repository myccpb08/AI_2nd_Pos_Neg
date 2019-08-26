import pickle
from threading import Thread
import sqlite3

import numpy as np
from konlpy.tag import Okt
from flask import Flask
from slack import WebClient
from slackeventsapi import SlackEventAdapter
from scipy.sparse import lil_matrix


# slack 연동 정보 입력 부분
SLACK_TOKEN = "xoxb-619154581858-619130324707-N5IswN8gy8ZO9mcIxVYPhrkP"
SLACK_SIGNING_SECRET = "2e319790605d216c629f5c107aa16c2a"

app = Flask(__name__)

slack_events_adaptor = SlackEventAdapter(SLACK_SIGNING_SECRET, "/listening", app)
slack_web_client = WebClient(token=SLACK_TOKEN)

# Req 2-2-1. pickle로 저장된 model.clf 파일 불러오기
pickle_obj = None
word_indices = None
clf = None

# Req 2-2-2. 토큰화 및 one-hot 임베딩하는 전 처리
def preprocess():

    return None

# Req 2-2-3. 긍정 혹은 부정으로 분류
def classify():

    return None
    
# Req 2-2-4. app.db 를 연동하여 웹에서 주고받는 데이터를 DB로 저장
    

# 챗봇이 멘션을 받았을 경우
@slack_events_adaptor.on("app_mention")
def app_mentioned(event_data):
    channel = event_data["event"]["channel"]
    text = event_data["event"]["text"]


@app.route("/", methods=["GET"])
def index():
    return "<h1>Server is ready.</h1>"

if __name__ == '__main__':
    app.run()
