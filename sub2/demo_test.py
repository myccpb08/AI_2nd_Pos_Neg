import numpy as np
import json
import pickle
import sqlite3

from konlpy.tag import Okt
from scipy.sparse import lil_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn import linear_model

from retrain import read_data, tokenize


# # Req 2-2-4. app.db 를 연동하여 웹에서 주고받는 데이터를 DB로 저장
def save_text_to_db(sentence):
    # db에 저장
    global output

    con = sqlite3.connect('./app.db')
    cur = con.cursor()

    cur.execute(
        'INSERT INTO search_history(question, answer) VALUES(?,?)', (sentence, output,))
    con.commit()

    output = -1
    cur.close()


# 결과값이 틀린 경우 데이터를 DB에 저장
def edit_data():
    chk = False
    con = sqlite3.connect('./app.db')
    cur = con.cursor()

    recent_record = cur.execute(
        'SELECT max(id), answer FROM search_history')
    idx = 0
    answer = -1
    for value in recent_record:
        idx = value[0]
        answer = value[1]

    if(answer == 0):
        cur.execute('UPDATE search_history SET answer = 1 WHERE id = %s' % idx)
        print("|\t긍정(1)으로 수정완료")
        chk = True
    elif(answer == 1):
        cur.execute('UPDATE search_history SET answer = 0 WHERE id = %s ' % idx)
        print("|\t부정(0)으로 수정완료")
        chk = True
    else:
        print("error")

    con.commit()
    cur.close()

    # db 업데이트 성공유무를 리턴
    return chk


# 추가 데이터 트레이닝
def data_training():
    global beforeTrainDataIdx

    chk = True
    con = sqlite3.connect('./app.db')
    cur = con.cursor()
    # DB에 저장된 데이터 개수 확인
    recent_record = cur.execute(
        'SELECT max(id), answer FROM search_history')

    recent_idx = 0
    for value in recent_record:
        recent_idx = value[0]

    # DB에 데이터가 10개 미만일 경우 chk -> false
    if((recent_idx - beforeTrainDataIdx) < 10):
        print("|\t추가로 저장된 데이터가 10개 미만입니다")
        print("|\t이전 문장으로 10번 학습시키겠습니까?(y/n)",end=" ")
        ans = input()
        if ans == "y":
            recent_record = cur.execute(
            'SELECT max(id), question, answer FROM search_history')

            for value in recent_record:
                temp_sentence = value[1]
                temp_ans = value[2]
            for _ in range(10):
                cur.execute(
                'INSERT INTO search_history(question, answer) VALUES(?,?)', (temp_sentence, temp_ans,))
                con.commit()

        chk = False

    # DB에 데이터가 10개 이상일 경우 chk -> true
    else:
        print("|\t학습하겠습니다")
        print("|\t{}부터 학습합니다".format(beforeTrainDataIdx+1))

        f = open('retrain.txt', mode='wt', encoding='utf-8')
        readData = cur.execute(
            'SELECT * FROM search_history WHERE id > %s ' % beforeTrainDataIdx
        )
        temp = ""
        for data in readData:
            row = str(data[1]) + "\t" + data[2] + "\t" + str(data[3]) + "\n"
            temp += row

        f.write(temp)
        print("|\t텍스트 파일 쓰기완료")
        f.close()

        beforeTrainDataIdx = recent_idx  # 현재 데이터까지 학습했음을 저장
        print("|\t{} idx 까지 학습완료".format(beforeTrainDataIdx))
        chk = True
        
        # 추가 데이터 트레이닝
        train_data = read_data('retrain.txt')
        train_docs = tokenize(train_data)
        print('|\tread_data, tokenize')

        X = lil_matrix((len(train_docs), len(word_indices)))
        Y = np.zeros(len(train_docs))

        print('|\tX, Y init')
        for idx in range(len(train_docs)):
            temp = [0]*len(word_indices)
            for morph in train_docs[idx]:
                word = morph.split('/')[0]
                if word_indices.get(word)!=None:
                    temp[word_indices[word]]=1
            X[idx]=temp
        print('|\tX one hot embedding ')
        for idx in range(len(train_data)):
            word = train_data[idx][2].split('\n')[0]
            Y[idx]=word
        print('|\tY label')
        clf.partial_fit(X, Y) # naive Bayes
        print('|\tnaive')
        clf2.partial_fit(X, Y) # Logistic
        print('|\tlogistic')
        clf3.partial_fit(X, Y) # SVM
        print('|\tSVM')

        fl = open('origin_model.clf', 'wb')
        pickle.dump(clf, fl)
        pickle.dump(clf2, fl)
        pickle.dump(clf3, fl)
        pickle.dump(clf4, fl)
        pickle.dump(word_indices, fl)
        fl.close()
    cur.close()
   
    return chk

# 전처리:토큰화
def preprocess(sentence):
    pos_tagger = Okt()
    Word_vector = []
    result = ['/'.join(t) for t in pos_tagger.pos(sentence, norm=True, stem=True)]
    Word_vector += [result]

    # one-hot 임베딩
    for idx in range(len(Word_vector)):
        temp = [0]*len(word_indices)
        for morph in Word_vector[idx]:
            word = morph.split('/')[0]
            if word_indices.get(word) != None:
                temp[word_indices[word]] = 1
        Word_vector[idx] = temp

    return Word_vector


def predict(Word_vector, test_clf):
    predict_result = test_clf.predict(Word_vector)[0]
    return predict_result

# # Req 2-2-3. 긍정 혹은 부정으로 분류

def classify(sentence, test_clf, model_name):
    global neg
    global pos
    
    weight = [0.3, 0.7]

    predict_result = int(predict(sentence, test_clf))    

    if model_name=="NB" or model_name=="DTC":
        neg += test_clf.predict_proba(sentence)[0][0]
        pos += test_clf.predict_proba(sentence)[0][1]

    else:
        neg += weight[1-predict_result]
        pos += weight[predict_result]

    return predict_result

def send_message(sentence):
    global neg, pos
    global output

    Word_vector = preprocess(sentence)
    predict_NB = classify(Word_vector, clf, "NB")
    predict_LR = classify(Word_vector, clf2, "LR")
    predict_SVM = classify(Word_vector, clf3, "SVM")
    predict_DTC = classify(Word_vector, clf4, "DTC")

    if neg > pos:
        output = 0
        print("|\t결과: < 부정 >")
    else:
        output = 1
        print("|\t결과: < 긍정 >")

    neg = 0
    pos = 0

    return output


if __name__ == '__main__':

    # Req 2-2-1. pickle로 저장된 model.clf 파일 불러오기
    pickle_obj = open('origin_model.clf', 'rb')
    clf = pickle.load(pickle_obj) # naive bayes
    clf2 = pickle.load(pickle_obj) # Logistic Regression
    clf3 = pickle.load(pickle_obj) # SVM
    clf4 = pickle.load(pickle_obj) # 의사결정트리
    word_indices = pickle.load(pickle_obj)

    # Req 2-2-2. 토큰화 및 one-hot 임베딩하는 전 처리
    pos_tagger = Okt()
    
    # 기능 수행 위한 초기 변수들
    neg = 0
    pos = 0
    output = -1  # 트레이닝결과값 0/1을 db에 쓰기위한 초기화 변수
    beforeTrainDataIdx = 0  # 까지 학습했다.를 저장하는 전역변수로 사용


    ''' 데모테스트 '''
     
    while True:
        print()
        print("============================================================")
        print("|\t메뉴를 선택해주세요")
        print("|\t1.문장입력")
        print("|\t2.Retraining")
        print("|\t3.종료")
        print("============================================================")
        select = int(input("|\t메뉴선택: "))
        if select == 3:
            break
        elif select == 1:
            sentence = input("|\t입력: ")
            send_message(sentence)
            save_text_to_db(sentence)
            print("|\t수정하시겠습니까?(y/n)", end=" ")
            ans = input()
            if ans == 'y':
                edit_data()
        elif select == 2:
            data_training()        
        else:
            print("|\t잘못입력했습니다.")
        print("============================================================")


    