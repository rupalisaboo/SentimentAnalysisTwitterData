"""
Read in the input csv file, get the tweet id and look for and parse the 
tweet and friends_data files, get the [tweet text] and [the concatenation of the friends description]
The friends descriptions are separated by [ ^|^ ].
It outputs one tab-separated-file with the format:
['company', 'sentiment', 'text', 'ID', 'friends_desc', 'user', 'date']
"""
import os, csv, json, codecs

def main():
    input_file = raw_input('Input csv file [./corpus.csv]: ')
    raw_dir = raw_input('raw data dir [./rawdata/]: ')
    friends_data_dir = raw_input('Friends data dir [./rawdata/friends_data]: ')
    output_file = raw_input('Full corpus [./full_corpus]: ')


    if not input_file:
        input_file = './corpus.csv'
    if not output_file:
        output_file = './full_corpus'
    if not friends_data_dir:
        friends_data_dir = './rawdata/friends_data/'
    elif not friends_data_dir.endswith('/'):
        friends_data_dir = friends_data_dir + '/'
    if not raw_dir:
        raw_dir = './rawdata/'
    elif not raw_dir.endswith('/'):
        raw_dir = raw_dir + '/'

    # ensure info data directory exists
    if not os.path.exists(raw_dir):
        raise RuntimeError('raw data dir %s not found' %raw_dir)
    if not os.path.exists(friends_data_dir):
        raise RuntimeError('raw data dir %s not found' %friends_data_dir)
    valid_tweets_data = []
    # read input corpus file
    with open(input_file, 'rb') as input_fh:
        csv_reader = csv.DictReader(input_fh, fieldnames=['company', 'sentiment', 'ID'])
        for row in csv_reader:
            data = {}
            data = row
            # find tweet in raw_dir and parse 
            tweet_id = row['ID'].replace('"','').strip()
            tweet_filename = '%s%s.json' %(raw_dir, tweet_id)
            friends_filename = '%s%s.json' %(friends_data_dir, tweet_id)
            if not os.path.exists(tweet_filename):
                print 'skipping %s: no tweet file.' %tweet_id
                continue
            if not os.path.exists(friends_filename):
                print 'skipping %s: no friends data file.' %tweet_id
                continue
            try:
                (data['date'], data['text'], data['user']) = parse_tweet_json(tweet_filename)
                data['friends_desc'] = get_friends_data(friends_filename)
            except RuntimeError as exc:
                print 'skipping %s: %s.' %(tweet_id, exc)
                continue
            valid_tweets_data.append(data)
    with open(output_file, 'wb') as output_fh:
        fields = ['company', 'sentiment', 'text', 'ID', 'friends_desc', 'user', 'date']
        for data in valid_tweets_data:
            row = [data[f] for f in fields]
            try:
                txt = u'\t'.join([unicode(r) for r in row]) + u'\n'
                #print type(txt)
                output_fh.write(txt.encode('utf8'))
            except TypeError as exc:
                #print row[4]
                raise exc
        """csv_writer = csv.DictWriter(output_fh, fieldnames=fields, delimiter='\t')
        csv_writer.writeheader()
        csv_writer.writerows(valid_tweets_data)
        csv_writer = csv.writer(output_fh, delimiter='\t')
        for data in valid_tweets_data:
            txt = [unicode(data[r]) for r in fields]
            csv_writer.writerow(txt)"""

def encode_fn(to_encode):
    if not isinstance(to_encode, basestring):
        return to_encode
    return to_encode.encode('utf8')

def parse_tweet(filename):
    # read tweet
    print
    'opening: ' + filename
    fp = open(filename, 'rb')

    # parse json
    try:
        tweet_json = json.load(fp)
    except ValueError:
        raise RuntimeError('error parsing json')

    # look for twitter api error msgs
    error_str = ['error', 'errors']
    for err in error_str:
        if err in tweet_json:
            raise RuntimeError('error in downloaded tweet: %s' %tweet_json[err])
    return tweet_json


def parse_tweet_json(filename):
    
    tweet_json = parse_tweet(filename)
    # extract creation date and tweet text
    return [tweet_json['created_at'], tweet_json['text'].replace('\n', '').replace('\t', ''), tweet_json['user']['id']]

def get_friends_data(filename):

    friends_desc = []
    friends_data_json = []
    with open(filename, 'rb') as friends_fh:
        jumbled_string = friends_fh.read()
        substr = '[{"follow_request_sent"'
        prev_index = -1
        #ret_value = 0
        while(True):
            ret_value = jumbled_string.find(substr, prev_index+1)
            if prev_index != -1:                
                # not first time .. complete previous block
                # the end ] of the previous block
                end = ret_value if (ret_value != -1) else len(jumbled_string)
                friends_data_json.append(jumbled_string[prev_index : end])
                #jumbled_string = jumbled_string[end:]
            if ret_value == -1:
                break
            prev_index = ret_value # the [
    for block in friends_data_json:
        friends = json.loads(block)
        for frnd in friends:
            if 'description' not in frnd:
                continue
            friends_desc.append(frnd['description'].replace('\n', '').replace('\t', ''))
    return u' ^|^ '.join(friends_desc) 

def split_full_corpus():
    full_corpus = raw_input('Full corpus csv file [./full_corpus.csv]: ')
    corpus = raw_input('Corpus (without tweet text and friends text) csv file [./tweet_corpus.csv]: ')
    tweet_text_filename = raw_input('Tweets Text corpus file[./tweet_text.csv]: ')
    friends_data_filename = raw_input('Friends Text corpus file [./friends_text.csv]: ')


    if not full_corpus:
        full_corpus = './full_corpus.csv'
    if not corpus:
        corpus = './tweet_corpus.csv'
    if not tweet_text_filename:
        tweet_text_filename = './tweet_text.csv'
    if not friends_data_filename:
        friends_data_filename = './friends_text.csv'

    # ensure full corpus exists
    if not os.path.exists(full_corpus):
        raise RuntimeError('Corpus %s not found' %full_corpus)
    corpus_data = []
    tweet_text_data = []
    friends_text_data = []
    with open(full_corpus, 'rb') as full_corpus_fh:
        #fields = ['company', 'sentiment', 'text', 'ID', 'friends_desc', 'user', 'date']
        for line in full_corpus_fh.readlines():
            fields = line.decode('utf8').strip().split(u'\t')
            #corpus
            temp = [fields[i] for i in [3, 0, 1, 5, 6]]
            corpus_data.append(u'\t'.join([unicode(r) for r in temp]) + u'\n')
            #tweet text
            temp = [fields[i] for i in [3, 2]]
            tweet_text_data.append(u'\t'.join([unicode(r) for r in temp]) + u'\n')
            #friends text
            temp = [fields[i] for i in [3, 4]]
            friends_text_data.append(u'\t'.join([unicode(r) for r in temp]) + u'\n')
    #write files
    write_file(corpus, corpus_data)
    write_file(tweet_text_filename, tweet_text_data)
    write_file(friends_data_filename, friends_text_data)

def write_file(filename, lines):
    with open(filename, 'wb') as file_fh:
        for line in lines:
            try:
                file_fh.write(line.encode('utf8'))
            except TypeError as exc:
                #print row[4]
                raise exc

if __name__ == '__main__':
    #main()
    split_full_corpus()
