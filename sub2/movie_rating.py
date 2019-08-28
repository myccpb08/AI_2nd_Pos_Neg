import numpy as np
import pickle

from konlpy.tag import Okt
from scipy.sparse import lil_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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


# Req 1-1-2. 문장 데이터 토큰화
# train_docs, test_docs : 토큰화된 트레이닝, 테스트  문장에 label 정보를 추가한 list
train_docs = tokenize(train_data)
test_docs = tokenize(test_data)


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
#print(word_indices)
# print(word_indices)

# Req 1-1-4. sparse matrix 초기화
# X: train feature data
# X_test: test feature data
X = lil_matrix((len(train_docs), len(word_indices)))
X_test = lil_matrix((len(test_docs), len(word_indices)))

# 평점 label 데이터가 저장될 Y 행렬 초기화
# Y: train data label
# Y_test: test data label
Y = np.zeros(len(train_docs))
Y_test = np.zeros(len(test_docs))

# Req 1-1-5. one-hot 임베딩
# X,Y 벡터값 채우기

for idx in range(len(train_docs)):
    temp = [0]*len(word_indices)
    for verb in train_docs[idx]:
        part = verb.split('/')[0]
        temp[word_indices[part]]=1
    X[idx]=temp

for idx in range(len(test_docs)):
    temp = [0]*len(word_indices)
    for verb in test_docs[idx]:
        part = verb.split('/')[0]
        if word_indices.get(part)!=None:
            temp[word_indices[part]]=1
    X_test[idx]=temp

for idx in range(len(train_data)):
    part = train_data[idx][2].split('\n')[0]
    Y[idx]=part

for idx in range(len(test_data)):
    part = test_data[idx][2].split('\n')[0]
    Y_test[idx]=part

# print(Y)

"""
트레이닝 파트
clf  <- Naive baysian mdoel
clf2 <- Logistic regresion model
"""
# Req 1-2-1. Naive baysian mdoel 학습
clf = MultinomialNB().fit(X, Y)

# Req 1-2-2. Logistic regresion mdoel 학습
clf2 = LogisticRegression(solver='lbfgs').fit(X,Y)


# """
# 테스트 파트
# """
# print(X_test[0])
# print(Y_test[0])
# # Req 1-3-1. 문장 데이터에 따른 예측된 분류값 출력
print("Naive bayesian classifier example result: {}, {}".format(test_data[4][1], clf.predict(X_test[4])[0]))
print("Logistic regression exampleresult: {}, {}".format(test_data[4][1], clf2.predict(X_test[4])[0]))
# # Req 1-3-2. 정확도 출력
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

# """
# 데이터 저장 파트
# """

# Req 1-4. pickle로 학습된 모델 데이터 저장
fl = open('model.clf', 'wb')
pickle.dump(clf, fl)
pickle.dump(clf2, fl)
pickle.dump(word_indices, fl)
fl.close

# Naive bayes classifier algorithm part
# 아래의 코드는 심화 과정이기에 사용하지 않는다면 주석 처리하고 실행합니다.


"""
Naive_Bayes_Classifier 알고리즘 클래스입니다.
"""

class Naive_Bayes_Classifier(object):

    """
    Req 3-1-1.
    log_likelihoods_naivebayes():
    feature 데이터를 받아 label(class)값에 해당되는 likelihood 값들을
    naive한 방식으로 구하고 그 값의 log값을 리턴
    """

    def log_likelihoods_naivebayes(self, feature_vector, Class):
        log_likelihood = 0.0
        if Class == 0:
            for feature_index in range(len(feature_vector)):
                if feature_vector[feature_index] == 1: #feature present
                    log_likelihood += np.log(self.likelihoods_0[0][feature_index])

                elif feature_vector[feature_index] == 0: 
                    log_likelihood += np.log(1.0 - self.likelihoods_0[0][feature_index])
        elif Class == 1:
            for feature_index in range(len(feature_vector)):
                if feature_vector[feature_index] == 1:
                    log_likelihood += np.log(self.likelihoods_1[0][feature_index])

                elif feature_vector[feature_index] == 0:
                    log_likelihood += np.log(1.0 - self.likelihoods_1[0][feature_index])
                
        return log_likelihood

    """
    Req 3-1-2.
    class_posteriors():
    feature 데이터를 받아 label(class)값에 해당되는 posterior 값들을
    구하고 그 값의 log값을 리턴
    """
    
    def class_posteriors(self, feature_vector):
        log_likelihood_0 = self.log_likelihoods_naivebayes(feature_vector, Class = 0)
        log_likelihood_1 = self.log_likelihoods_naivebayes(feature_vector, Class = 1)

        log_posterior_0 = log_likelihood_0 + self.log_prior_0
        log_posterior_1 = log_likelihood_1 + self.log_prior_1

        return (log_posterior_0, log_posterior_1)

    """
    Req 3-1-3.
    classify():
    feature 데이터에 해당되는 posterir값들(class 개수)을 불러와 비교하여
    더 높은 확률을 갖는 class를 리턴
    """    

    def classify(self, feature_vector):
        log_posterior_0, log_posterior_1 = self.class_posteriors(feature_vector)
        return log_posterior_1 > log_posterior_0

    """
    Req 3-1-4.
    train():
    트레이닝 데이터를 받아 학습하는 함수
    학습 후, 각 class에 해당하는 prior값과 likelihood값을 업데이트

    알고리즘 구성
    1) 가중치 값인 beta_x_i, beta_c_i 초기화
    2) Y label 데이터 reshape
    3) 가중치 업데이트 과정 (iters번 반복) 
    3-1) prediction 함수를 사용하여 error 계산
    3-2) gadient_beta 함수를 사용하여 가중치 값 업데이트
    4) 최적화 된 가중치 값들 리턴
       self.beta_x, self.beta_c
    """

    def train(self, X, Y):
        # label 0에 해당되는 데이터의 개수 값(num_0) 초기화
        num_0 = 0 # 부정 댓글 수
        # label 1에 해당되는 데이터의 개수 값(num_1) 초기화
        num_1 = 0 # 긍정 댓글 수

        # Req 3-1-7. smoothing 조절
        # likelihood 확률이 0값을 갖는것을 피하기 위하여 smoothing 값 적용
        smoothing = 1

        # label 0에 해당되는 각 feature 성분의 개수값(num_token_0) 초기화 
        num_token_0 = np.zeros((1,X.shape[1])) # 부정 댓글에 등장하는 단어를 표시하기 위한 배열
        # label 1에 해당되는 각 feature 성분의 개수값(num_token_1) 초기화 
        num_token_1 = np.zeros((1,X.shape[1])) # 긍정 댓글에 등장하는 단어를 표시하기 위한 배열

        # label별 단어사전 만들기
        # 데이터의 num_0,num_1,num_token_0,num_token_1 값 계산     
        for i in range(X.shape[0]):
            if (Y[i] == 0):
                num_0 += 1
                num_token_0 += X[i][0].toarray()[0]
        
            if (Y[i] == 1):
                num_1 += 1
                num_token_1 += X[i][0].toarray()[0]

        # smoothing을 사용하여 각 클래스에 해당되는 likelihood값 계산        
        # 해당 형태소의 개수/부정댓글 벡터
        self.likelihoods_0 = (num_token_0 + smoothing) / ( np.sum(num_token_0) + 2*smoothing )
        self.likelihoods_1 = (num_token_1 + smoothing) / ( np.sum(num_token_0) + 2*smoothing )

        # 각 class의 prior를 계산
        prior_probability_0 = num_0/X.shape[0]
        prior_probability_1 = num_1/X.shape[0]

        # pior의 log값 계
        self.log_prior_0 = np.log(prior_probability_0)
        self.log_prior_1 = np.log(prior_probability_1)

        return None

    """
    Req 3-1-5.
    predict():
    테스트 데이터에 대해서 예측 label값을 출력해주는 함수
    """

    def predict(self, X_test):
        predictions = []
        X_test=X_test.toarray()
        if (len(X_test)==1):
            Y_pred = self.classify(X_test[0])
            predictions.append(Y_pred)
        else:
            for case in X_test:
                Y_pred = self.classify(case)
                predictions.append(Y_pred)
        return np.array(predictions)

    """
    Req 3-1-6.
    score():
    테스트를 데이터를 받아 예측된 데이터(predict 함수)와
    테스트 데이터의 label값을 비교하여 정확도를 계산
    """
    
    def score(self, X_test, Y_test):
        predictions = self.predict(X_test)
        mom = len(Y_test)
        cnt = 0
        for idx in range(mom):
            if int(predictions[idx])==int(Y_test[idx]):
                cnt +=1
        answer = cnt/mom*100
        return answer

# Req 3-2-1. model에 Naive_Bayes_Classifier 클래스를 사용하여 학습합니다.
# model = Naive_Bayes_Classifier()
# model.train(X, Y)

# # Req 3-2-2. 정확도 측정
# print("Naive_Bayes_Classifier accuracy: {}".format(model.score(X_test, Y_test)))

# # Logistic regression algorithm part
# # 아래의 코드는 심화 과정이기에 사용하지 않는다면 주석 처리하고 실행합니다.

"""
Logistic_Regression_Classifier 알고리즘 클래스입니다.
"""

class Logistic_Regression_Classifier(object):
    
    """
    Req 3-3-1.
    sigmoid():
    인풋값의 sigmoid 함수 값을 리턴
    """
    def sigmoid(self,z):
        hypothesis = 1/(1+np.exp(-z))
        return hypothesis

    """
    Req 3-3-2.
    prediction():
    X 데이터와 beta값들을 받아서 예측 확률P(class=1)을 계산.
    X 행렬의 크기와 beta의 행렬 크기를 맞추어 계산.
    ex) sigmoid(            X           x(행렬곱)       beta_x.T    +   beta_c)       
                (데이터 수, feature 수)             (feature 수, 1)
    """

    def prediction(self, beta_x, beta_c, X):
        # 예측 확률 P(class=1)을 계산하는 식을 만든다.
        pred_res = np.dot(X, beta_x) + beta_c
        return self.sigmoid(pred_res)

    """
    Req 3-3-3.
    gradient_beta():
    beta값에 해당되는 gradient값을 계산하고 learning rate를 곱하여 출력.
    """
    
    def gradient_beta(self, X, error, lr):
        # beta_x를 업데이트하는 규칙을 정의한다.
        beta_x_delta = -(lr/len(error)*(np.dot(X.T, error))) # (X.shape[1], 1)
        # beta_c를 업데이트하는 규칙을 정의한다.
        beta_c_delta = -(lr*np.mean(error))
        return beta_x_delta, beta_c_delta

    """
    Req 3-3-4.
    train():
    Logistic Regression 학습을 위한 함수.
    학습데이터를 받아서 최적의 sigmoid 함수으로 근사하는 가중치 값을 리턴.

    알고리즘 구성
    1) 가중치 값인 beta_x_i, beta_c_i 초기화
    2) Y label 데이터 reshape
    3) 가중치 업데이트 과정 (iters번 반복) 
    3-1) prediction 함수를 사용하여 error 계산
    3-2) gadient_beta 함수를 사용하여 가중치 값 업데이트
    4) 최적화 된 가중치 값들 리턴
       self.beta_x, self.beta_c
    """
    
    def train(self, X, Y):
        # Req 3-3-8. learning rate 조절
        # 학습률(learning rate)를 설정한다.(권장: 1e-3 ~ 1e-6)
        lr = 0.7
        # 반복 횟수(iteration)를 설정한다.(자연수)
        iters = 20000
        
        # beta_x, beta_c값을 업데이트 하기 위하여 beta_x_i, beta_c_i값을 초기화
        beta_x_i = np.zeros((X.shape[1], 1)) + 0.15
        beta_c_i = -3
    
        #행렬 계산을 위하여 Y데이터의 사이즈를 (len(Y),1)로 저장합니다.
        Y= Y.reshape(len(Y), 1)
        X = X.toarray()
        for i in range(iters):
            #실제 값 Y와 예측 값의 차이를 계산하여 error를 정의합니다.
            error = self.prediction(beta_x_i, beta_c_i, X) - Y 
            #gredient_beta함수를 통하여 델타값들을 업데이트 합니다.
            beta_x_delta, beta_c_delta = self.gradient_beta(X, error, lr)
            beta_x_i -= beta_x_delta
            beta_c_i -= beta_c_delta
            
        self.beta_x = beta_x_i
        self.beta_c = beta_c_i
        
        return None

    """
    Req 3-3-5.
    classify():
    확률값을 0.5 기준으로 큰 값은 1, 작은 값은 0으로 리턴
    """

    def classify(self, X_test):
        prob_Y = self.sigmoid(np.dot(X_test, self.beta_x) + self.beta_c)
        return prob_Y > 0.5 

    """
    Req 3-3-6.
    predict():
    테스트 데이터에 대해서 예측 label값을 출력해주는 함수
    """
    
    def predict(self, X_test):
        predictions = []
        X_test=X_test.toarray()
        if (len(X_test)==1):
            predictions.append(self.classify(X_test[0]))
        else:
            for case in X_test:
                predictions.append(self.classify(case))
        
        return np.array(predictions)


    """
    Req 3-3-7.
    score():
    테스트를 데이터를 받아 예측된 데이터(predict 함수)와
    테스트 데이터의 label값을 비교하여 정확도를 계산
    """
    
    def score(self, X_test, Y_test):
        predictions = self.predict(X_test)
        mom = len(Y_test)
        cnt = 0
        for idx in range(mom):
            if int(predictions[idx])==int(Y_test[idx]):
                cnt +=1
        answer = cnt/mom*100
        return answer

# Req 3-4-1. model2에 Logistic_Regression_Classifier 클래스를 사용하여 학습합니다.
model2 = Logistic_Regression_Classifier()
model2.train(X, Y)

# Req 3-4-2. 정확도 측정

print("Logistic_Regression_Classifier accuracy: {}".format(model2.score(X_test, Y_test)))