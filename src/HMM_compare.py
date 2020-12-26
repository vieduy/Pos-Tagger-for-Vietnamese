import sys
import math
from decimal import *
import codecs
from pyvi import ViTokenizer, ViPosTagger
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
def parse_traindata(i):
    fin = sys.argv[i]
    wordtag_list = []
    b=0
    try:
        input_file = codecs.open(fin, mode = 'r', encoding="utf-8")
        lines = input_file.readlines()
        for line in lines:
            line = line.strip('\r\n')
            data = line.split(" ")
            print(data)
            b+=len(data)
            wordtag_list.append(data)
        input_file.close()
        return wordtag_list,b

    except IOError:
        print("Không thể đọc file")
        sys.exit()
def main():
    a=len(sys.argv)
    Train,b=parse_traindata(a-2)
    Gold,c=parse_traindata(a-1)
    correct=0
    a=[]
    for i,sen_tag_train in enumerate(Train):
        for j,sen_tag_gold in enumerate(Gold):
            if(i==j):
                for z,word_tag_train in enumerate(sen_tag_train):
                    for g,word_tag_gold in enumerate(sen_tag_gold):
                        if (sen_tag_train[z]==sen_tag_gold[g] and z==g):
                            correct+=1
    print("Độ chính xác:",correct/b)
if __name__ == '__main__':
    main()