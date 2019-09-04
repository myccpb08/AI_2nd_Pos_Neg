import numpy as np
import pickle

from konlpy.tag import Okt
from scipy.sparse import lil_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import linear_model

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
train_data = read_data('naver_reple.txt')
test_data = read_data('ratings_test_test.txt')

print("1. ___Data preprocessing complete____")

# Req 1-1-2. 문장 데이터 토큰화
# train_docs, test_docs : 토큰화된 트레이닝, 테스트  문장에 label 정보를 추가한 list
# 태깅 후 json 파일로 저장
# 태깅이 완료된 json 파일이 존재하면 토큰화를 반복하지 않음

train_docs = tokenize(train_data)
test_docs = tokenize(test_data)
    

print("2. ___Data Tokenization complete____")

# Req 1-1-3. word_indices 초기화
pickle_obj = open('model3.clf', 'rb')
clf = pickle.load(pickle_obj)
word_indices = pickle.load(pickle_obj)

# word_indices = {}
# Req 1-1-3. word_indices 채우기
# idx = 0
# for part in train_docs:
#     for k in part:
#         meaning = k.split('/')[0]
#         if word_indices.get(meaning)==None:
#             word_indices[meaning]=idx
#             idx+=1

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
        if word_indices.get(part)!=None:
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

# NB = MultinomialNB()
# NB.fit(X[:100], Y[:100])
# fl = open('model3.clf', 'wb')
# pickle.dump(NB, fl)
# fl.close()
# # NB.partial_fit(X[100:], Y[100:])
# print(NB.score(X_test, Y_test))
# clf = MultinomialNB()
# clf.fit(X, Y)
# print(clf.score(X_test, Y_test))

print(clf.score(X_test, Y_test))
clf.partial_fit(X, Y)
print(clf.score(X_test, Y_test))
# NB = MultinomialNB()
# NB.fit(X, Y)
# print(NB.score(X_test, Y_test))



# print(clf.score(X_test, Y_test))
# clf.partial_fit(X[100:], Y[100:])
# print(clf.score(X_test, Y_test))

# print('logistic')
# clf1 = linear_model.SGDClassifier(loss='log', max_iter=1000, tol=1e-3, shuffle=False)
# clf1.fit(X, Y)
# print(clf1.score(X_test, Y_test)) 

# clf2 = linear_model.SGDClassifier(loss='log', max_iter=1000, tol=1e-3, shuffle=False)
# clf2.fit(X[:100], Y[:100])
# print(clf2.score(X_test, Y_test))
# clf2.partial_fit(X[100:], Y[100:])
# print(clf2.score(X_test, Y_test))

# print("logistic")
# clf2 = LogisticRegression(solver='lbfgs')
# clf2.fit(X, Y)
# print(clf2.score(X_test, Y_test))

# clfs = LogisticRegression(solver='lbfgs', warm_start = True)
# clfs.fit(X[:100], Y[:100])
# print(clfs.score(X_test, Y_test))
# clfs.fit(X[100:], Y[100:])
# print(clfs.score(X_test, Y_test))
