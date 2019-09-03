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

# creawling.py의 함수 쓰기위해서 import
import crawling
from slacker import Slacker

# slack 연동 정보 입력 부분
SLACK_TOKEN = "xoxb-724397827219-739240775904-5fQc8gC6RkZRx0MTFu2ol2Ia"
SLACK_SIGNING_SECRET = "f8cdb41515e9fa9fafdcf64c57ac3850"

# 챗봇 응답 메세지 띄우기위함
slacktest = Slacker(SLACK_TOKEN)

app = Flask(__name__)

slack_events_adaptor = SlackEventAdapter(
    SLACK_SIGNING_SECRET, "/listening", app)
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
    # one-hot 임베딩
    for idx in range(len(text_docs)):
        temp = [0]*len(word_indices)
        for verb in text_docs[idx]:
            part = verb.split('/')[0]
            if word_indices.get(part) != None:
                temp[word_indices[part]] = 1
        text_docs[idx] = temp

    return text_docs

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


# print(classify(preprocess("이 영화 노잼"), clf))


# # Req 2-2-4. app.db 를 연동하여 웹에서 주고받는 데이터를 DB로 저장


# 챗봇이 멘션을 받았을 경우
@slack_events_adaptor.on("app_mention")
def app_mentioned(event_data):
    channel = event_data["event"]["channel"]
    text = event_data["event"]["text"]

    # # DB에 데이터 저장
    # save_text_to_db(text)
    # # 메세지 보내기
    send_message(text, channel)

    # search_mv = text.split("> ")[1]
    # slack_response_movieLink(search_mv)


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
        "text": result,
        "fields": [
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
    slack_web_client.chat_postMessage(channel=ch, text=None, attachments=[
                                      attachement],  as_user=False)


def slack_response_movieLink(search_mv):  # 영화를 입력했을때 관련 페이지로 이동하는 링크 제공

    print(search_mv)
    link_address = crawling.get_movie_links_bySearch(search_mv)

    attachments_dict = dict()
    attachments_dict['pretext'] = "여기에서 결과를 띄울꺼에요"
    attachments_dict['title'] = "'"+search_mv+"'" + " 에 대한 영화 정보입니다"
    attachments_dict['title_link'] = link_address
    attachments_dict['fallback'] = "클라이언트에서 노티피케이션에 보이는 텍스트 입니다. attachment 블록에는 나타나지 않습니다"
    attachments_dict['text'] = "본문 텍스트! 5줄이 넘어가면 *show more*로 보이게 됩니다."
    # 마크다운을 적용시킬 인자들을 선택합니다.
    attachments_dict['mrkdwn_in'] = ["text", "pretext"]
    attachments = [attachments_dict]

    slacktest.chat.post_message(channel="#general", text=None,
                                attachments=attachments, as_user=True)


@app.route("/", methods=["GET"])
def index():
    return "<h1>Server is ready.</h1>"


if __name__ == '__main__':
    app.run()
