import re, nltk
'''
NLP Feature Engineering Functions

source : https://www.analyticsvidhya.com/blog/2021/04/a-guide-to-feature-engineering-in-nlp/
'''

#simply the length of the entire text
def count_chars(text):
    try:
        return len(text)
    except:
        return 0

#return number of words contain in the text
def count_words(text):
    try:
        return len(text.split())
    except:
        return 0

#return number of capital characters contained in the text
def count_capital_chars(text):
    try:
        count=0
        for i in text:
            if i.isupper():
                count+=1
        return count
    except:
        return 0

'''

Uppercase/capital-related function

all --> all letter in the word are capital
any --> check wether any letter in the word is capital
begin --> check wether the word starts in capital letter

'''
def contain_capital(word):
    try:
        return not(word.islower())
    except:
        return False

def begins_in_capital(word):
    try:
        return word[0].isupper()
    except:
        return False

capital_function = {
    'all' : str.isupper,
    'any' : contain_capital,
    'begin' : begins_in_capital
}

#return the number of words containing capital letter as described by the how argument
def count_capital_words(text,how='all'):
    try:
        return sum(map(capital_function[how],text.split()))
    except:
        return 0

#returns the number of punctuation contained in the text
def count_punctuations(text):
    try:
        punctuations='!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        d=dict()
        for i in punctuations:
            d[str(i)+' count']=text.count(i)
        return d
    except:
        return dict()

#return number of words in double quote
def count_words_in_double_quotes(text):
    try:
        x = re.findall(r'"([^"]*)"', text)
        count=0
        if x is None:
            return 0
        else:
            for i in x:
                t=i[1:-1]
                count+=count_words(t)
            return count
    except:
        return 0

#return number of words in single quote
def count_words_in_single_quotes(text):
    try:
        x = re.findall(r"'([^']*)'", text)
        count=0
        if x is None:
            return 0
        else:
            for i in x:
                t=i[1:-1]
                count+=count_words(t)
            return count
    except:
        return 0

#return number of sentences
def count_sent(text):
    try:
        return len(nltk.sent_tokenize(text))
    except:
        return 0

#return number of unique words
#example : "I love you the way you love me" returns 5 (you, love, and me appear more than once)
def count_unique_words(text):
    try:
        return len(set(text.split()))
    except:
        return 0

#return number of hashtags contained in the text
#  *this maybe helpful in handling social media status/post
def count_htags(text):
    try:
        x = re.findall(r'#\w+', text)
        return len(x)
    except:
        return 0 

#return number of mentions contained in the text
#  *this maybe helpful in handling social media status/post
def count_mentions(text):
    try:
        x = re.findall(r'@\w+', text)
        return len(x)
    except:
        return 0

'''
Stop words related functions

source : https://github.com/masdevid/ID-Stopwords/blob/master/id.stopwords.02.01.2016.txt

source also similar with nltk.corpus.stopwords.words('indonesian')
'''

filename = "id-stopwords.txt"

with open(filename) as file:
    lines = file.readlines()
    stopwords = [line.rstrip() for line in lines]

file.close()

#returns number of stopwords appear in the text
def count_stopwords(text):
    try:
        word_tokens = nltk.word_tokenize(text)
        appeared_stopwords = [w for w in word_tokens if w in stopwords]
        return len(appeared_stopwords)
    except:
        return 0

#returns the ration between the character counts and the word counts
def average_word_length(text):
    try:
        return count_chars(text)/count_words(text)
    except:
        return 0

#returns the ratio between the word counts and the sentence counts
def average_sentence_length(text):
    try:
        return count_words(text)/count_sent(text)
    except:
        return 0

#return the ration between the unique word counts and the word counts
def unique_vs_words(text):
    try:
        return count_unique_words(text)/count_words(text)
    except:
        return 0

#return the ratio between the stopwords count and the word counts
def stopwords_vs_words(text):
    try:
        return count_stopwords(text)/count_words(text)
    except:
        return 0


'''
Generate all feature

returning dataframe
'''

import pandas as pd

def generate_all(df):
    res = pd.DataFrame()
    res['count_chars'] = df.apply(count_chars)
    res['count_words'] = df.apply(count_words)
    res['count_capital_chars'] = df.apply(count_capital_chars)

    res['count_capital_words_all'] = df.apply(lambda x: count_capital_words(x,'all'))
    res['count_capital_words_any'] = df.apply(lambda x: count_capital_words(x,'any'))
    res['count_capital_words_begin'] = df.apply(lambda x: count_capital_words(x,'begin'))

    res['count_words_in_double_quotes'] = df.apply(count_words_in_double_quotes)
    res['count_words_in_single_quotes'] = df.apply(count_words_in_single_quotes)

    res['count_sent'] = df.apply(count_sent)
    res['count_unique_words'] = df.apply(count_unique_words)

    res['count_htags'] = df.apply(count_htags)
    res['count_mentions'] = df.apply(count_mentions)
    res['count_stopwords'] = df.apply(count_stopwords)

    res['average_word_length'] = df.apply(average_word_length)
    res['average_sentence_length'] = df.apply(average_sentence_length)
    res['unique_vs_words'] = df.apply(unique_vs_words)
    res['stopwords_vs_words'] = df.apply(stopwords_vs_words)

    return res