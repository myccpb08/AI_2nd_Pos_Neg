import pickle
import numpy as np

from flask import Flask
from slack import WebClient
from slackeventsapi import SlackEventAdapter


SLACK_TOKEN = 'xoxb-731614402629-733495701111-6QglObMVmrUpPNSJz4bob0Vo'
SLACK_SIGNING_SECRET = '33d0b00dfeb6ab2a156b78392ccb01b1'

app = Flask(__name__)

slack_events_adaptor = SlackEventAdapter(SLACK_SIGNING_SECRET, "/listening", app)
slack_web_client = WebClient(token=SLACK_TOKEN)

# Req. 2-1-1. pickle로 저장된 model.clf 파일 불러오기
fl = open('model.clf', 'rb')
clf_from_pickle = pickle.load(fl)
# Req. 2-1-2. 입력 받은 광고비 데이터에 따른 예상 판매량을 출력하는 lin_pred() 함수 구현
def lin_pred(test_str):
    data = test_str.split(' ')
    data = list(map(float, data[1:]))
    x = np.array(data)
    w = clf_from_pickle.coef_
    sales = np.dot(x, w) + clf_from_pickle.intercept_
    return sales

# 챗봇이 멘션을 받았을 경우
@slack_events_adaptor.on("app_mention")
def app_mentioned(event_data):
    channel = event_data["event"]["channel"]
    text = event_data["event"]["text"]
    keywords = lin_pred(text)
    
    slack_web_client.chat_postMessage(
        channel=channel,
        text=keywords
    )
    

@app.route("/", methods=["GET"])
def index():
    return "<h1>Server is ready.</h1>"

if __name__ == '__main__':
    app.run()
