__author__ = 'rupali'

import re

'''
Method replaces emoticons from tweet with corresponding sentiment
'''
def removeEmoticons(tweets):
    with open('resources/emoticons_sentiments.csv', "r") as emoFile:
        rows = (line.split('2') for line in emoFile)
        for row in rows:
            cleanRow = re.escape(row[0].strip())
            tweets = re.sub(cleanRow, row[1].replace('\n', '').strip(), tweets)
    return tweets


'''
Method to replace strings with repeating characters with 3 repeating characters
eg. 'cooooooool' with 'coool'
'''
def removeRepeatChar(tweets):
    repl = r'\1\1\1'
    repeatRegexp = re.compile(r'(\w)(\1{2,})')
    repeatRegexp.sub(repl, tweets)
    return repeatRegexp.sub(repl, tweets)

'''
Method to replace URLs with ||U||
'''
def removeWebLinks(tweets):
    remLinks1 = re.compile(r"(http://[^ ]+)")
    remLinks2 = re.compile(r"(https://[^ ]+)")
    repl = "||U||"
    tweets = remLinks1.sub(repl, tweets)
    tweets = remLinks2.sub(repl, tweets)
    return tweets

'''
Method to replace targets with ||T||
'''
def removeTargets(tweets):
    remLinks1 = re.compile(r"(@[^ ]+)")
    remLinks2 = re.compile(r"(#[^ ]+)")
    repl = "||T||"
    tweets = remLinks1.sub(repl, tweets)
    tweets = remLinks2.sub(repl, tweets)
    return tweets



def main():
    #filename = 'resources/test_tweets.csv'
    filename = 'data/full_corpus_20'
    with open(filename, "r") as tweetFile:
        tweets = tweetFile.read().replace('\n', '|^^|')
        remLinks = removeWebLinks(tweets)
        remTargets = removeTargets(remLinks)
        replEmoticons = removeEmoticons(remTargets)
        cleanRepeatChar = removeRepeatChar(replEmoticons)

    with open('resources/test', 'w') as test:
        test.write(cleanRepeatChar.replace('|^^|', '\n'))

if __name__ == '__main__':
    main()
