from collections import Counter
import math

corpus = ['this is the first document',
        'this is the second second document',
        'and the third one',
        'is this the first document']

def main():
    words_list = [item.split(' ') for item in corpus]
    cnt_list = [Counter(item) for item in words_list]
    ret = []
    for i, cnt in enumerate(cnt_list):
        temp = []
        for word in cnt.keys():
            temp.append((word, tf(word, cnt) * idf(word, cnt_list)))
        ret.append(temp)
    print(ret)

def tf(word, cnt):
    return cnt[word] / sum(cnt.values())

def idf(word, cnt_list):
    contain_num = sum([1 for cnt in cnt_list if word in cnt.keys()])
    return math.log((len(cnt_list) / (contain_num + 1)))

if __name__ == '__main__':
    main()