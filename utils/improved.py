from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import GradientBoostingClassifier 
from pandas import DataFrame
from collections import Counter
import nltk
from nltk.corpus import wordnet
import re 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.learning_curve import learning_curve



class Baseline(object):

    def __init__(self, language):
        self.language = language
        # from 'Multilingual and Cross-Lingual Complex Word Identification' (Yimam et. al, 2017)
        if language == 'english':
            self.avg_word_length = 5.3
        else:  # spanish
            self.avg_word_length = 6.2

        self.model = RandomForestClassifier()


    def word_frequences(self, trainset): 
        frame_train = DataFrame(trainset)
        column_train = frame_train['sentence']
        word = []
        for i in range(len(column_train)):
            word.extend(re.sub("[^\w']"," ",column_train[i]).split())
        word_frequence = Counter(word) 
        return word_frequence

    def char_frequence(self, trainset):
        frame_train = DataFrame(trainset)
        column_train = frame_train['sentence']
        char = []
        for i in range(len(column_train)):
            word = re.sub("[^\w']"," ",column_train[i]).split()
            for j in word:
                for k in j:
                    char.append(k)
        char_frequence = Counter(char) 
        return char_frequence
    
    def pos_dictionary(self, trainset):
        pos_list = []
        for sent in trainset:
            tagged = nltk.pos_tag(sent['target_word'])[0][1]
            pos_list.append(tagged)
        pos_list = list(set(pos_list))
        pos_dictionary = {}
        for i in range(len(pos_list)):
            pos_dictionary[pos_list[i]] = i        
        return pos_dictionary

    def bigram_counts_word(self,trainset):
        frame_train = DataFrame(trainset)
        column_train = frame_train['sentence'] 
        bigrams = []
        for i in range(len(column_train)):
            sent = re.sub("[^\w']"," ",column_train[i]).split()
            bigrams.extend(nltk.bigrams(sent, pad_left=True, pad_right=True))
        bigram_counts = Counter(bigrams)
        return bigram_counts


    def bigram_counts_char(self,trainset):
        frame_train = DataFrame(trainset)
        column_train = frame_train['sentence'] 
        bigrams = []
        for i in range(len(column_train)):
            sent = re.sub("[^\w']"," ",column_train[i]).split()
            character = []
            for j in sent:
                for k in j:
                    character.append(k)
            bigrams.extend(nltk.bigrams(character, pad_left=True, pad_right=True))
        bigram_counts = Counter(bigrams)
        return bigram_counts

    def lengh_trainset(self,trainset):
        lengh_trainset = len(DataFrame(trainset)['sentence'])
        return lengh_trainset

    def bigram_word(self,word,word_frequence,bigram_counts_word,lengh_trainset):
        unique_words = len(word_frequence) + 2
        smoothing=1.0
        word_list = re.sub("[^\w']"," ",word).split()
        x_bigrams = nltk.bigrams(word_list, pad_left=True, pad_right=True)
        prob_x = 1.0
        for bg in x_bigrams:
            if bg[0] == None:
                prob_bg = (bigram_counts_word[bg]+smoothing)/(lengh_trainset +smoothing*unique_words)
            else:
                prob_bg = (bigram_counts_word[bg]+smoothing)/(word_frequence[bg[0]]+smoothing*unique_words)
            prob_x = prob_x *prob_bg
        return prob_x

    def lengh_char(self, trainset):
        frame_train = DataFrame(trainset)
        column_train = frame_train['sentence']
        char = []
        for i in range(len(column_train)):
            word = re.sub("[^\w']"," ",column_train[i]).split()
            for j in word:
                for k in j:
                    char.append(k)
        lengh_char = len(char)
        return lengh_char

    def bigram_char(self,word,char_frequences,bigram_counts_char,lengh_char):

        unique_words = len(char_frequences) + 2
        smoothing=1.0
        word_list = re.sub("[^\w']"," ",word).split()
        char = []
        for i in word_list:
            for j in i:
                char.append(j)
        x_bigrams = nltk.bigrams(char, pad_left=True, pad_right=True)
        prob_x = 1.0
        for bg in x_bigrams:
            if bg[0] == None:
                prob_bg = (bigram_counts_char[bg]+smoothing)/(lengh_char +smoothing*unique_words)
            else:
                prob_bg = (bigram_counts_char[bg]+smoothing)/(char_frequences[bg[0]]+smoothing*unique_words)
            prob_x = prob_x *prob_bg
        return prob_x

    def extract_features(self, word, word_frequence,char_frequence,pos_dictionary,bigram_counts_word,lengh_trainset,bigram_counts_char,lengh_char):
        len_chars = len(word) / self.avg_word_length
        len_tokens = len(word.split(' '))

        word_list = re.sub("[^\w']"," ",word).split()
        fre = 0
        for i in range(len(word_list)):
            word1 = word_list[i]
            if word1 in word_frequence:
                fre = fre + word_frequence[word1]
        word_fre = fre / len(word_list)

        
        tagged = nltk.pos_tag(word)[0][1]
        pos_tag = pos_dictionary[tagged]
        
        synonyms=[]
        list_good=wordnet.synsets(word)
        for syn in list_good:
            for l in syn.lemmas():
                synonyms.append(l.name())
        
        lengh_synonyms = len(synonyms)
        
        char_count = 0
        for i in range(len_tokens):
            w = len(word.split(' ')[i])
            char_count = char_count + w
        char_count = char_count / len_tokens

        bigram_word = self.bigram_word(word,word_frequence,bigram_counts_word,lengh_trainset)

        bigram_char = self.bigram_char(word,char_frequence,bigram_counts_char,lengh_char)
        
        return [len_chars, len_tokens, word_fre, pos_tag, lengh_synonyms, char_count]

    
 
    def train(self, trainset,word_frequence,pos_dictionary,bigram_counts_word,lengh_trainset,char_frequence,lengh_char, bigram_counts_char):
        X = []
        y = []
        for sent in trainset:
            X.append(self.extract_features(sent['target_word'],word_frequence,char_frequence,pos_dictionary,bigram_counts_word,lengh_trainset,bigram_counts_char,lengh_char))
            y.append(sent['gold_label'])

        self.model.fit(X, y)

        plt.figure()
        title = "Learning Curves_RandomForestClassifier"
        plt.title(title)
        plt.xlabel("Training examples")
        plt.ylabel("F1 Score")
        estimator = self.model 
        train_sizes = np.linspace(.1, 1.0,5)
        train_sizes, train_scores,test_scores= learning_curve(estimator, X, y, n_jobs=1, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        # test_scores_mean = np.mean(test_scores, axis=1)
        # test_scores_std = np.std(test_scores, axis=1)
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1,
                        color="r")
        plt.plot(train_sizes, train_scores_mean, 'o-', 
                label="Training score")

        plt.show()

    def test(self, testset,word_frequence,pos_dictionary,bigram_counts_word,lengh_trainset,char_frequence,lengh_char, bigram_counts_char):
        X = []
        for sent in testset:
            X.append(self.extract_features(sent['target_word'],word_frequence,char_frequence,pos_dictionary,bigram_counts_word,lengh_trainset,bigram_counts_char,lengh_char))

        return self.model.predict(X)






