import numpy as np
import pickle

from konlpy.tag import Okt
from scipy.sparse import lil_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

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
            if line.startswith("i"):
                continue
            else:
                temp = line.split('\t')
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
    

# Req 1-1-4. sparse matrix 초기화
# X: train feature data
# X_test: test feature data
X = lil_matrix((len(train_docs), len(word_indices)))
X_test = lil_matrix((len(test_docs), len(word_indices)))


# 평점 label 데이터가 저장될 Y 행렬 초기화
# Y: train data label
# Y_test: test data label
Y = np.zeros((len(train_docs),1))
Y_test = np.zeros((len(test_docs),1))

# Req 1-1-5. one-hot 임베딩
# X,Y 벡터값 채우기

for idx in range(len(train_docs)): # idx = 댓글번호
    temp = [0]*len(word_indices) # temp = [0,0,0,.....]  : 0의 개수 = 사전단어수
    for verb in train_docs[idx]:  # idx 번재 댓글
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



"""
Naive_Bayes_Classifier 알고리즘 클래스입니다.
"""

class Naive_Bayes_Classifier(object):  # 분류작업하는 'Naive_Bayes_분류기' 라는 클래스

    """
    Req 3-1-1.
    log_likelihoods_naivebayes():
    feature 데이터를 받아 label(class)값에 해당되는 likelihood 값들을
    naive한 방식으로 구하고 그 값의 log값을 리턴
    """
    
    def log_likelihoods_naivebayes(self, feature_vector, Class):

        # 어떤 문장이 들어왔을 때, 분해해서 그 문장정보에 해당하는 feature_vector 를 생성
        # feature_vector : 단어 사전 크기를 갖는 벡터

        log_likelihood = 0.0  # 해당 class 에서 feature_vector 가 나타날 확률의 log 값
        
        if Class == 0:  # 부정댓글이면
            for feature_index in range(len(feature_vector)):
                if feature_vector[feature_index] == 1: #feature present : 부정댓글일 때, 해당 단어가 존재 = P(W1 | C1)
                    log_likelihood += np.log(self.likelihoods_0[0][feature_index])

                elif feature_vector[feature_index] == 0: #feature absent : 부정 댓글일 때, 해당 단어가 없음
                    log_likelihood += np.log(1-self.likelihoods_0[0][feature_index])

        elif Class == 1:
            for feature_index in range(len(feature_vector)):
                if feature_vector[feature_index] == 1:
                    log_likelihood += np.log(self.likelihoods_1[0][feature_index])
                elif feature_vector[feature_index] == 0:
                    log_likelihood += np.log(1-self.likelihoods_1[0][feature_index])

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

        (neg, pos) = self.class_posteriors(feature_vector)
        
        if max(neg, pos) == neg: #부정 확률이 더 크면
            group = '0'
        else:
            group = '1'

        return group


    """
    Req 3-1-4.
    train():
    트레이닝 데이터를 받아 학습하는 함수
    학습 후, 각 class에 해당하는 prior값과 likelihood값을 업데이트

    """

    def train(self, X, Y):
        # label 0에 해당되는 데이터의 개수 값(num_0) 초기화
        num_0 = 0
        # label 1에 해당되는 데이터의 개수 값(num_1) 초기화
        num_1 = 0

        # Req 3-1-7. smoothing 조절
        # likelihood 확률이 0값을 갖는것을 피하기 위하여 smoothing 값 적용

        smoothing = 0.35

        # label 0에 해당되는 각 feature 성분의 개수값(num_token_0) 초기화 
        num_token_0 = np.zeros((1,X.shape[1]))  # 부정단어 딕셔너리
        # label 1에 해당되는 각 feature 성분의 개수값(num_token_1) 초기화 
        num_token_1 = np.zeros((1,X.shape[1]))

        
        # 데이터의 num_0,num_1,num_token_0,num_token_1 값 계산 (부정과, 긍정에 해당하는 사전 완성)
        for i in range(X.shape[0]):  # 테스트할 문장 수
            if (Y[i] == 0): # 부정문장이면
                num_0 += 1 # 부정 문장 수 +1 하고
                num_token_0 += X[i][0].toarray()[0]
        
            if (Y[i] == 1):
                num_1 += 1 # 긍정문장 수 추가
                num_token_1 += X[i][0].toarray()[0]


        # smoothing을 사용하여 각 클래스에 해당되는 likelihood값 계산      

        
        # mom = sum(num_token_0) + np.count_nonzero(num_token_0)
        num_token_0 += smoothing
        num_token_0 /= (num_0 + 2*smoothing)

        num_token_1 += smoothing
        num_token_1 /= (num_1 + 2*smoothing)

        self.likelihoods_0 = num_token_0
        self.likelihoods_1 = num_token_1

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
            predictions.append(self.classify(X_test[0]))
        else:
            for case in X_test: # 각 문장의 feature vector
                # print(case)
                answer = self.classify(case)
                predictions.append(answer)
        
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
model = Naive_Bayes_Classifier()
model.train(X,Y)



# Req 3-2-2. 정확도 측정
print("Naive_Bayes_Classifier accuracy: {}%".format(round(model.score(X_test, Y_test),2)))