# Movie Review Slack-Chatbot

<hr>

```markdown
Movie Review Slack-chatbot 은
챗봇에게 어떠한 영화 리뷰를 보냈을 때,
해당 리뷰가 `긍정인지` `부정인지` 알려주는 기능을 구현한 코드입니다

만약
`재미있다` 라는 문장이 `부정` 으로 응답받은 경우
해당 문장을 `긍정으로 변경` 하여 학습데이터에 추가할 수 있도록
`Edit` 버튼이 구현되어 있습니다.

어느 정도 데이터가 쌓였을 경우,
기존의 학습데이터에 사용자가 만든 `추가학습데이터`를
training 시킬 수 있도록 `training` 버튼을 구현하였습니다.
retrain 을 시킬 수 있는 추가학습데이터의 수로 최소 10개가 요구됩니다.

기본 학습DB 는
IMDB 에서 얻을 수 있는 15만개 가량의 영화리뷰를 기반으로 하였고,
`NAVER SHOW` 버튼을 누르는 경우
IMDB 데이터가 아닌, `네이버에서 수집한 댓글들` 을 기반으로 학습한 model을 바탕으로
추정된 결과를 볼 수 있습니다.
```





## Setup(window) 

1. python 설치 (https:www.python.org/downloads/)

   > > 해당 프로젝트에서, `3.6버전` 사용

   

2. Flask 설치
   `pip install flask==0.12.2`

   

3. Numpy 설치

   `pip install numpy1.16.3`



4. Scikit-learn 설치

   `pip install -U scikit-learn==0.20.3`



5. slackclient 설치

   `pip install slackclient==2.1.0`

   `pip install slackeventsapi==2.1.0`



6. Ngrok 설치
   https://ngrok.com/  접속하여, `ngrok.exe` 파일 다운



7. KoNLPy 설치

   7-1. Open JDK 1.8 버전 설치 : https://jdk.java.net/

   7-2. pip 업그레이드
   	   `pip install --upgrade pip`

   7-3. JPypel 설치
   		https://www.lfd.uci.edu/~gohlke/pythonlibs/#jpype

   ​		해당 프로젝트에서는 `JPypel-0.6.3-cp36-cp36m-wub_amd64.whl` 사용

   ​		`pip install JPypel-0.6.3-cp36-cp36m-wub_amd64.whl`

   ​		`pip install KoNLPy==0.5.1`

   

8. SCIPY 설치
   `pip install scipy==1.2.1`



9. SQLite browser 설치

   https://sdqlitebrowser.org/dl







## Slack App 연동

1. 슬랙 워크스페이스 가입 후, 슬랙 앱 생성(https://api.slack.com/apps)

   `Create New App`

2. 챗봇 이름 설정

3. 슬랙 앱 생성 후, 슬랙 앱 설정 페이지 진입

   ![](https://i.postimg.cc/cJ1Sybtd/1.png)

4. 아래 두 가지 설정을 완료한 후 `authorize` 클릭

   ![](https://i.postimg.cc/Dy3R3MRR/2-1.png)

   ![](https://i.postimg.cc/9X7sShgG/2-2.png)

5. flask 서버 사용 위해 `app.py` 에 작성
   `Basic information` >> `App Credentials` >> `Slack_token` 과 `slack_signing_secret` 발급

   ```python
   SLACK_TOKEN = Bot User OAuth Access Token
   SLACK_SIGNING_SECRET = signing secret
   ```

   

6. `Scopes` 메뉴에서 슬랙 챗봇이 메시지를 보낼 수 있는 `chat:write:bot` 권한을 추가

7. `Event Subscriptions` >> `Enable Events` >> `on`
8. Request Url : ngrok 의 `Forwarding주소/listening` 입력
   * ngrok 사용법은 하단 <b>RUN</b> 부분에 있음
   * EX) `https://165eada1/.ngrok.io/listening`

9. `subscribe to bot events` >> `add bot user event`

10. `Interactive Components` 의 Request Url

    `Forwarding주소/click`







## RUN

### Slack APP 실행

1. CMD 에서 `Ngrok.exe` 파일이 위치한 디렉토리로 이동하여 `ngrok http 5000` 실행

2. 생성된 `Forwarding` 주소 확인 후 `Slack App Request URL` 에 해당 정보 입력

   ![](https://i.postimg.cc/s2LDTmBw/image.png)

   

3. 프로젝트 내 `app.py` 경로로 이동하여 Flask 서버 실행

   `python app.py`



4. Slack 워크스페이스에 접속하여 Slack App 호출 및 데이터 입력하여 결과 확인
   ![](https://i.postimg.cc/dQLxCYcb/Kakao-Talk-20190906-110614007.png)

   4-1) `NAVER SHOW 버튼` 눌렀을 경우 결과 확인

   ​		![](https://i.postimg.cc/8Pks4ttg/Kakao-Talk-20190906-110614007.png)





