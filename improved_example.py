from utils.dataset import Dataset
from utils.improved import Baseline
from utils.scorer import report_score



def execute_demo(language):
    data = Dataset(language)

    print("{}: {} training - {} test".format(language, len(data.trainset), len(data.testset)))

    baseline = Baseline(language)

    word_frequence = baseline.word_frequences(data.trainset)

    char_frequence = baseline.char_frequence(data.trainset)

    lengh_trainset = baseline.lengh_trainset(data.trainset)
    
    bigram_counts_word = baseline.bigram_counts_word(data.trainset)

    pos_dictionary = baseline.pos_dictionary(data.trainset)

    lengh_char = baseline.lengh_char(data.trainset)

    bigram_counts_char = baseline.bigram_counts_char(data.trainset)



    baseline.train(data.trainset,word_frequence,pos_dictionary,bigram_counts_word,lengh_trainset,char_frequence,lengh_char, bigram_counts_char)

    predictions = baseline.test(data.testset,word_frequence,pos_dictionary,bigram_counts_word,lengh_trainset,char_frequence,lengh_char, bigram_counts_char)

    gold_labels = [sent['gold_label'] for sent in data.testset]




    report_score(gold_labels, predictions)




    
    

if __name__ == '__main__':
    execute_demo('english')
    execute_demo('spanish')


