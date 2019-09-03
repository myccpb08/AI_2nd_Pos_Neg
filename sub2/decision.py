import numpy as np
import pickle

from konlpy.tag import Okt
from scipy.sparse import lil_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import json
import os

pos_tagger = Okt()
# print(pos_tagger.nouns('한국어 분석을 시작합니다'))
# print(pos_tagger.morphs('한국어 분석을 시작합니다'))
# print(pos_tagger.pos('한국어 분석을 시작합니다'))

"""
Req 1-1-1. 데이터 읽기
read_data(): 데이터를 읽어서 저장하는 함수
"""

def read_data(filename):
    data = []
    with open(filename, 'r',encoding='UTF-8') as f:
        for line in f:
            temp = line.split('\t')
            if temp[1] != "document":
                data += [temp]
        return data

"""
Req 1-1-2. 토큰화 함수
tokenize(): 텍스트 데이터를 받아 KoNLPy의 okt 형태소 분석기로 토크나이징
"""

def tokenize(doc):
    total_pos = []
    for sentence in doc:
        check_sentence = sentence[1]
        result = ['/'.join(t) for t in pos_tagger.pos(check_sentence, norm=True, stem=True)]
        total_pos += [result]
            
    return total_pos


"""
데이터 전 처리
"""

# train, test 데이터 읽기
train_data = read_data('ratings_train.txt')
test_data = read_data('ratings_test.txt')

print("1. ___Data preprocessing complete____")

# Req 1-1-2. 문장 데이터 토큰화
# train_docs, test_docs : 토큰화된 트레이닝, 테스트  문장에 label 정보를 추가한 list
# 태깅 후 json 파일로 저장
# 태깅이 완료된 json 파일이 존재하면 토큰화를 반복하지 않음
if os.path.isfile('train_do.json'):
    with open('train_docs.json', 'r', encoding="utf-8") as f:
        train_docs = json.load(f)
    with open('test_docs.json', 'r', encoding="utf-8") as f:
        test_docs = json.load(f)
else:
    train_docs = tokenize(train_data)
    test_docs = tokenize(test_data)
    with open('train_docs.json', 'w', encoding="utf-8") as make_file:
        json.dump(train_docs, make_file, ensure_ascii=False, indent="\t")
    with open('test_docs.json', 'w', encoding="utf-8") as make_file:
        json.dump(test_docs, make_file, ensure_ascii=False, indent="\t")

print("2. ___Data Tokenization complete____")

# Req 1-1-3. word_indices 초기화

word_indices = {}
# Req 1-1-3. word_indices 채우기
idx = 0
for part in train_docs:
    for k in part:
        meaning = k.split('/')[0]
        if word_indices.get(meaning)==None:
            word_indices[meaning]=idx
            idx+=1

print("3. ___Word Indice Complete____")
#print(word_indices)
# print(word_indices)

# Req 1-1-4. sparse matrix 초기화
# X: train feature data
# X_test: test feature data
X = lil_matrix((len(train_docs), len(word_indices)))
X_test = lil_matrix((len(test_docs), len(word_indices)))

print("4. ___X, X_test sparse matrix Init____")

# 평점 label 데이터가 저장될 Y 행렬 초기화
# Y: train data label
# Y_test: test data label
Y = np.zeros(len(train_docs))
Y_test = np.zeros(len(test_docs))

print("5. ___Y, Y_test sparse matrix Init____")

# Req 1-1-5. one-hot 임베딩
# X,Y 벡터값 채우기

for idx in range(len(train_docs)):
    temp = [0]*len(word_indices)
    for verb in train_docs[idx]:
        part = verb.split('/')[0]
        temp[word_indices[part]]=1
    X[idx]=temp
print("6. ___X one-hot embedding Complete____")
for idx in range(len(test_docs)):
    temp = [0]*len(word_indices)
    for verb in test_docs[idx]:
        part = verb.split('/')[0]
        if word_indices.get(part)!=None:
            temp[word_indices[part]]=1
    X_test[idx]=temp
print("7. ___X_test one-hot embedding Complete____")
for idx in range(len(train_data)):
    part = train_data[idx][2].split('\n')[0]
    Y[idx]=part

for idx in range(len(test_data)):
    part = test_data[idx][2].split('\n')[0]
    Y_test[idx]=part
print("8. ___Y, Y_test processing Complete____")
# print(Y)

"""
트레이닝 파트
clf  <- Naive baysian mdoel
clf2 <- Logistic regresion model
"""

# Decision Tree
# clf3 <- Decision Tree
tree = DecisionTreeClassifier(max_depth=X.shape[0], random_state = 0)
tree.fit(X, Y)
print("의사결정트리")
print("훈련 세트 정확도: {:.3f}".format(tree.score(X, Y)))
print("훈련 세트 정확도: {:.3f}".format(tree.score(X_test, Y_test)))

# RandomForest
# clf4 <= RandomForest
forest = RandomForestClassifier(n_estimators= 1000, random_state=2)
forest.fit(X, Y)
print("랜덤포레스트")
print("훈련 세트 정확도: {:3f}".format(forest.score(X, Y)))
print("훈련 세트 정확도: {:3f}".format(forest.score(X_test, Y_test)))

# Req 1-2-1. Naive baysian mdoel 학습
clf = MultinomialNB().fit(X, Y)

# # Req 1-2-2. Logistic regresion mdoel 학습
clf2 = LogisticRegression(solver='lbfgs').fit(X,Y)


# # """
# # 테스트 파트
# # """
# # print(X_test[0])
# # print(Y_test[0])
# # # Req 1-3-1. 문장 데이터에 따른 예측된 분류값 출력
# # print("Naive bayesian classifier example result: {}, {}".format(test_data[4][1], clf.predict(X_test[4])[0]))
# # print("Logistic regression exampleresult: {}, {}".format(test_data[4][1], clf2.predict(X_test[4])[0]))
# # # Req 1-3-2. 정확도 출력
y_pred_temp = []
y_pred_temp2 = []
for data in X_test:
    y_pred_temp.append(clf.predict(data)[0])
    y_pred_temp2.append(clf2.predict(data)[0])
y_pred_NB = np.array(y_pred_temp)
y_pred_LR = np.array(y_pred_temp2)
# print(y_pred_NB)
# print(y_pred_LR)
# print(Y_test)
# print(accuracy_score(y_pred_NB, Y_test))
print("Naive bayesian classifier accuracy: {}".format(accuracy_score(Y_test, y_pred_NB)))
print("Logistic regression accuracy: {}".format(accuracy_score(Y_test, y_pred_LR)))

"""
데이터 저장 파트
"""

# Req 1-4. pickle로 학습된 모델 데이터 저장
fl = open('model.clf', 'wb')
pickle.dump(clf, fl)
pickle.dump(clf2, fl)
pickle.dump(word_indices, fl)
pickle.dump(tree, fl)
pickle.dump(forest, fl)
fl.close
