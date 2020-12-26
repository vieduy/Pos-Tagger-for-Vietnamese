import sys
import math
from decimal import *
import codecs
from pyvi import ViTokenizer, ViPosTagger
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

tag_list = set()
tag_count = {}
word_set = set()


def parse_traindata():
    wordtag_list = []
    try:
        input_file = codecs.open('./src/dataset_nlp.txt', mode='r', encoding="utf-8")
        lines = input_file.readlines()
        for line in lines:
            line = line.strip('\r\n')
            data = line.split(" ")
            wordtag_list.append(data)
        input_file.close()
        return wordtag_list

    except IOError:
        print("Không thể đọc file")
        sys.exit()


# Thống kê tần số
def transition_count():
    # print "In Transition Model"
    global tag_list
    global word_set
    train_data = parse_traindata()
    transition_dict = {}
    global tag_count
    for value in train_data:
        previous = "start"
        for data in value:
            i = data[::-1]
            word = data[:-i.find("/") - 1]
            word_set.add(word.lower())
            data = data.split("/")
            tag = data[-1]
            tag_list.add(tag)
            if tag in tag_count:
                tag_count[tag] += 1
            else:
                tag_count[tag] = 1
            if (previous + "~tag~" + tag) in transition_dict:
                transition_dict[previous + "~tag~" + tag] += 1
                previous = tag
            else:
                transition_dict[previous + "~tag~" + tag] = 1
                previous = tag
    return transition_dict


def transition_probability():
    count_dict = transition_count()
    prob_dict = {}

    for key in count_dict:
        den = 0
        val = key.split("~tag~")[0]
        for key_2 in count_dict:
            if key_2.split("~tag~")[0] == val:
                den += count_dict[key_2]
        prob_dict[key] = Decimal(count_dict[key] + 1) / (den + len(tag_list))
    return prob_dict, count_dict


def transition_smoothing():
    a = {}
    transition_prob, count_dict = transition_probability()

    for i in count_dict.items():
        arr = i[::-1]
        if arr[1].split("~tag~")[0] in a:
            a[arr[1].split("~tag~")[0]] += arr[0]
        else:
            a[arr[1].split("~tag~")[0]] = arr[0]
    for j in tag_list:
        if j not in a.keys():
            a[j] = 0
    for tag in tag_list:
        if "start" + "~tag~" + tag not in transition_prob:
            transition_prob[("start" + "~tag~" + tag)] = Decimal(1) / Decimal(a["start"] + len(tag_list))
    for tag1 in tag_list:
        for tag2 in tag_list:
            if (tag1 + "~tag~" + tag2) not in transition_prob:
                transition_prob[(tag1 + "~tag~" + tag2)] = Decimal(1) / Decimal(a[tag1] + len(tag_list))
    return transition_prob


def emission_count():
    # print "In Emission Model"
    train_data = parse_traindata()
    count_word = {}
    for value in train_data:
        for data in value:
            i = data[::-1]
            word = data[:-i.find("/") - 1]
            tag = data.split("/")[-1]
            if word.lower() + "/" + tag in count_word:
                count_word[word.lower() + "/" + tag] += 1
            else:
                count_word[word.lower() + "/" + tag] = 1
    return count_word


def emission_probability():
    global tag_count
    word_count = emission_count()
    emission_prob_dict = {}
    for key in word_count:
        emission_prob_dict[key] = Decimal(word_count[key] + 1) / (tag_count[key.split("/")[-1]] + len(word_set))
    for word in word_set:
        for tag in tag_count.items():
            if word + "/" + tag[0] not in emission_prob_dict:
                emission_prob_dict[word + "/" + tag[0]] = Decimal(1) / (tag[1] + len(word_set))
    return emission_prob_dict


def viterbi_algorithm(sentence, tags, transition_prob, emission_prob, tag_count_emis, word_set):
    global tag_set
    word_list = sentence.split(" ")
    current_prob = {}
    for tag in tags:
        tp = Decimal(0)
        em = Decimal(0)
        if "start~tag~" + tag in transition_prob:
            tp = Decimal(transition_prob["start~tag~" + tag])
        if word_list[0].lower() in word_set:
            if (word_list[0].lower() + "/" + tag) in emission_prob:
                em = Decimal(emission_prob[word_list[0].lower() + "/" + tag])
                current_prob[tag] = tp * em
        else:
            em = Decimal(1) / (tag_count_emis[tag] + len(word_set))
            current_prob[tag] = tp
    if len(word_list) == 1:
        max_path = max(current_prob, key=current_prob.get)
        return max_path
    else:
        for i in range(1, len(word_list)):
            previous_prob = current_prob
            current_prob = {}
            locals()['dict{}'.format(i)] = {}
            previous_tag = ""
            for tag in tags:
                if word_list[i].lower() in word_set:
                    if word_list[i].lower() + "/" + tag in emission_prob:
                        em = Decimal(emission_prob[word_list[i].lower() + "/" + tag])
                        max_prob, previous_state = max((Decimal(previous_prob[previous_tag]) * Decimal(
                            transition_prob[previous_tag + "~tag~" + tag]) * em, previous_tag) for previous_tag in
                                                       previous_prob)
                        current_prob[tag] = max_prob
                        locals()['dict{}'.format(i)][previous_state + "~" + tag] = max_prob
                        previous_tag = previous_state
                else:
                    em = Decimal(1) / (tag_count_emis[tag] + len(word_set))
                    max_prob, previous_state = max((Decimal(previous_prob[previous_tag]) * Decimal(
                        transition_prob[previous_tag + "~tag~" + tag]) * em, previous_tag) for previous_tag in
                                                   previous_prob)
                    current_prob[tag] = max_prob
                    locals()['dict{}'.format(i)][previous_state + "~" + tag] = max_prob
                    previous_tag = previous_state
            if i == len(word_list) - 1:
                max_path = ""
                last_tag = max(current_prob, key=current_prob.get)
                max_path = max_path + last_tag
                for j in range(len(word_list) - 1, 0, -1):
                    for key in locals()['dict{}'.format(j)]:
                        data = key.split("~")
                        if data[-1] == previous_tag:
                            max_path = max_path + " " + data[0]
                            previous_tag = data[0]
                            break
                result = max_path.split()
                result.reverse()
                return " ".join(result)


def main(input_file):
    global tag_count, string_out
    tag_count_emis = {}

    # Xác suất chuyển trạng thái
    transition_model = transition_smoothing()

    # Xác suất emission
    emission_model = emission_probability()
    for prob in emission_model.items():
        i = prob[::-1]
        i = list(i)
        key_tag = i[1]
        val = key_tag.split("/")[-1]
        if val in tag_count_emis:
            tag_count_emis[val] += 1
        else:
            tag_count_emis[val] = 1
    # fin = sys.argv[2]
    # input_file = codecs.open(fin, mode='r', encoding="utf-8")
    input_file = [input_file]
    fout = codecs.open("HmmOutput.txt", mode='w', encoding="utf-8")
    for sentence in input_file:
        sentence = ViTokenizer.tokenize(sentence)
        lib_pos = ViPosTagger.postagging(sentence)
        path = viterbi_algorithm(sentence, tag_list, transition_model, emission_model, tag_count_emis, word_set)
        sentence = sentence.strip("\r\n")
        word = sentence.split(" ")
        tag = path.split(" ")
        string_out = ''
        lib_str = ''
        d = 0
        l = 0
        for j in range(0, len(word)):
            if tag[j] == lib_pos[1][j] and lib_pos[1][j] != 'F': d += 1
            if lib_pos[1][j] == 'F': l += 1
            if j == len(word) - 1:
                lib_str += lib_pos[0][j] + "/" + lib_pos[1][j] + u'\n'
                string_out += word[j] + "/" + tag[j] + u'\n'
                fout.write(word[j] + "/" + tag[j] + u'\n')
            else:
                lib_str += lib_pos[0][j] + "/" + lib_pos[1][j] + ' '
                string_out += word[j] + "/" + tag[j] + ' '
                fout.write(word[j] + "/" + tag[j] + " ")
    acc = round(d/(len(word) - l)*100)

    return string_out, lib_str, acc


if __name__ == '__main__':
    text = input()
    print(main(text))
