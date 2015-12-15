#
# Sanders-Twitter Sentiment Corpus Install Script
# Version 0.1
#
# Pulls tweet data from Twitter because ToS prevents distributing it directly.
#
# Right now we use unauthenticated requests, which are rate-limited to 150/hr.
# We use 125/hr to stay safe.  
#
# We could more than double the download speed by using authentication with
# OAuth logins.  But for now, this is too much of a PITA to implement.  Just let
# the script run over a weekend and you'll have all the data.
#
#   - Niek Sanders
#     njs@sananalytics.com
#     October 20, 2011
#
#
# Excuse the ugly code.  I threw this together as quickly as possible and I
# don't normally code in Python.
#
import csv, getpass, json, os, time, sys
import oauth2 as oauth
import urllib2 as urllib

# See assignment1.html instructions or README for how to get these credentials

api_key = "jNjhXYVfvuKenyWwbccTgkcjX"
api_secret = "kO0YTvewzCDSf8YXFj2ICoee4HtZ7wkadES8zKdUgTpXaNbYmZ"
access_token_key = "289034756-XBqlBm6poMcSCR0yOdDVBzZ8K8O3aeR1uKVN0kam"
access_token_secret = "OimWPfFjbnD0F7AXBhcqP4Z0MyT0GCwkRMs9OJ58l2kQ2"

_debug = 0

oauth_token = oauth.Token(key=access_token_key, secret=access_token_secret)
oauth_consumer = oauth.Consumer(key=api_key, secret=api_secret)

signature_method_hmac_sha1 = oauth.SignatureMethod_HMAC_SHA1()

http_method = "GET"

http_handler = urllib.HTTPHandler(debuglevel=_debug)
https_handler = urllib.HTTPSHandler(debuglevel=_debug)


def twitterreq(url, method, parameters):
    req = oauth.Request.from_consumer_and_token(oauth_consumer,
                                                token=oauth_token,
                                                http_method=http_method,
                                                http_url=url,
                                                parameters=parameters)

    req.sign_request(signature_method_hmac_sha1, oauth_consumer, oauth_token)

    headers = req.to_header()

    if http_method == "POST":
        encoded_post_data = req.to_postdata()
    else:
        encoded_post_data = None
        url = req.to_url()

    opener = urllib.OpenerDirector()
    opener.add_handler(http_handler)
    opener.add_handler(https_handler)

    response = opener.open(url, encoded_post_data)

    return response


def get_user_params():
    user_params = {}

    # get user input params
    user_params['inList'] = raw_input('\nInput file [./corpus.csv]: ')
    user_params['outList'] = raw_input('Results file [./full-corpus.csv]: ')
    user_params['rawDir'] = raw_input('Raw data dir [./rawdata/]: ')

    # apply defaults
    if user_params['inList'] == '':
        user_params['inList'] = './corpus.csv'
    if user_params['outList'] == '':
        user_params['outList'] = './full-corpus.csv'
    if user_params['rawDir'] == '':
        user_params['rawDir'] = './rawdata/'

    return user_params


def dump_user_params(user_params):
    # dump user params for confirmation
    print
    'Input:    ' + user_params['inList']
    print
    'Output:   ' + user_params['outList']
    print
    'Raw data: ' + user_params['rawDir']
    return


def read_total_list(in_filename):
    # read total fetch list csv
    fp = open(in_filename, 'rb')
    reader = csv.reader(fp, delimiter=',', quotechar='"')

    total_list = []
    for row in reader:
        total_list.append(row)

    return total_list


def purge_already_fetched(fetch_list, raw_dir):
    # list of tweet ids that still need downloading
    rem_list = []

    # check each tweet to see if we have it
    for item in fetch_list:

        # check if json file exists
        tweet_file = raw_dir + item[2] + '.json'
        if os.path.exists(tweet_file):

            # attempt to parse json file
            try:
                parse_tweet_json(tweet_file)
                print
                '--> already downloaded #' + item[2]
            except RuntimeError:
                rem_list.append(item)
        else:
            rem_list.append(item)

    return rem_list


def get_time_left_str(cur_idx, fetch_list, download_pause):
    tweets_left = len(fetch_list) - cur_idx
    total_seconds = tweets_left * download_pause

    str_hr = int(total_seconds / 3600)
    str_min = int((total_seconds - str_hr * 3600) / 60)
    str_sec = total_seconds - str_hr * 3600 - str_min * 60

    return '%dh %dm %ds' % (str_hr, str_min, str_sec)


def download_tweets(fetch_list, raw_dir):
    # ensure raw data directory exists
    if not os.path.exists(raw_dir):
        os.mkdir(raw_dir)

    # stay within rate limits
    max_tweets_per_hr = 180
    download_pause_sec = 900 / max_tweets_per_hr

    # download tweets
    for idx in range(0, len(fetch_list)):
        # current item
        item = fetch_list[idx]

        # print status
        trem = get_time_left_str(idx, fetch_list, download_pause_sec)
        print
        '--> downloading tweet #%s (%d of %d) (%s left)' % \
        (item[2], idx + 1, len(fetch_list), trem)

        # pull data
        url = 'https://api.twitter.com/1.1/statuses/show.json?id=' + item[2]
        response = twitterreq(url, "GET", [])
        # urllib.urlretrieve( url, raw_dir + item[2] + '.json' )
        with open(raw_dir + item[2] + '.json', 'w') as fh:
            fh.write(response.read())

        # stay in Twitter API rate limits
        print item[2] + '.json' 
        print '    pausing %d sec to obey Twitter API rate limits' % (download_pause_sec)
        time.sleep( download_pause_sec )
    return

def build_user_info():

    raw_dir = raw_input('\nRaw tweets dir[./rawdata/]: ')
    user_info_dir_name = raw_input('Friends data directory name [friends_data]: ')

    # apply defaults
    if raw_dir == '':
        raw_dir = './rawdata/'
    if user_info_dir_name == '':
        user_info_dir_name= 'friends_data'

    # ensure info data directory exists
    if not os.path.exists(raw_dir):
        raise RuntimeError('raw data dir %s not found' %raw_dir)
    user_info_dir_full = '%s%s/' %(raw_dir, user_info_dir_name)
    if not os.path.exists(user_info_dir_full):
        os.mkdir(user_info_dir_full)
    file_list = os.listdir(raw_dir)

    #total = len(file_list)
    for i, tweet_file in enumerate(file_list):
        if tweet_file.endswith('.json'):
            try:
                tweet_json = parse_tweet('%s/%s' %(raw_dir, tweet_file))
                user_id = tweet_json['user']['id']
                friends_cnt = tweet_json['user']['friends_count']
                friends_data = download_user_info2(user_id)
                with open('%s%s' %(user_info_dir_full, tweet_file), 'w') as fh:
                    for fr_data in friends_data:
                        fh.write(json.dumps(fr_data))
                print '[%d]' %(i+1)
                # stay in Twitter API rate limits
                #print '[%d]  pausing %d sec to obey Twitter API rate limits' % (i+1, pause_sec * len(friends_data) / 2)
                #time.sleep( pause_sec * len(friends_data) / 2)
            except RuntimeError as exc:
                print '    skipping %s: %s' %(tweet_file, exc)
                continue

def download_user_info(user_id):

    #API limit - 
    # app auth: 30 calls / 15 min
    # user auth: 15 calls / 15 min
    #  after each call, sleep for (15*60/15) secs
    sleep_time = 900/15

    friends_data = []
    #How many pages you want to get - 1 pg = 200 friends(max)
    pages = 2
    #cursor for twitter pagination, default -1 for 1st page.
    cursor=-1
    for pg in range(pages):
        url = 'https://api.twitter.com/1.1/friends/list.json?user_id=%s&cursor=%s&count=200&skip_status=true&include_user_entities=false' %(user_id, cursor)
        response = twitterreq(url, "GET", [])
        response_str = response.read()
        response_json = json.loads(response_str)
        if 'error' in response_json or 'errors' in response_json:
            print response_str
            raise RuntimeError('error in downloaded friends data.')
        if 'next_cursor' not in response_json:
            raise RuntimeError('error in downloaded friends data: cursor missing')
        if 'users' not in response_json:
            raise RuntimeError('error in downloaded friends data: users missing')
        friends_data.append(response_json['users'])
        cursor = response_json['next_cursor']
        # stay in Twitter API rate limits
        print ' pausing %d sec to obey Twitter API rate limits' % (sleep_time)
        time.sleep(sleep_time)
        if cursor == 0:
            break
    return friends_data

def download_user_info2(user_id):

    #API limit - 
    # 15 calls / 15 min
    # after each friends/IDs call, sleep for (15*60/15) secs
    # no need sleeping for users/lookup call cos we call it 10 times per user_id since we take 1000 friends max
    # which comes to 150 calls / 15 min which is below the 180 limit.
    sleep_time = 900/15

    friends_data = []
    #get friend IDs
    url = 'https://api.twitter.com/1.1/friends/ids.json?user_id=%s&count=1000&stringify_ids=true' %(user_id)
    response = twitterreq(url, "GET", [])
    response_str = response.read()
    response_json = json.loads(response_str)
    if 'error' in response_json or 'errors' in response_json:
        print response_str
        raise RuntimeError('error in downloaded friends IDs data.')
    if 'ids' not in response_json:
        raise RuntimeError('error in downloaded friends data: IDs missing')
    friend_ids = response_json['ids']
    num_friends = len(friend_ids)
    # lookup 100 at a time
    for i in range(num_friends/100 + 1):
        start = i*100
        end = (i+1)*100
        if start >= num_friends:
            break
        if end > num_friends:
            end = num_friends
        #print friend_ids[i*100 : end]
        ids = ','.join(friend_ids[i*100 : end])
        # lookup
        url = 'https://api.twitter.com/1.1/users/lookup.json?user_id=%s&include_user_entities=false' %(ids)
        response = twitterreq(url, "GET", [])
        response_str = response.read()
        response_json = json.loads(response_str)
        if 'error' in response_json or 'errors' in response_json:
            print response_str
            raise RuntimeError('error in downloaded friends IDs data.')
        friends_data.append(response_json)
    # stay in Twitter API rate limits
    print ' pausing %d sec to obey Twitter API rate limits' % (sleep_time)
    time.sleep(sleep_time)
    return friends_data


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
    if 'error' in tweet_json or 'errors' in tweet_json:
        raise RuntimeError('error in downloaded tweet')
    return tweet_json


def parse_tweet_json(filename):
    
    tweet_json = parse_tweet(filename)
    # extract creation date and tweet text
    return [tweet_json['created_at'], tweet_json['text'], tweet_json['user']['id']]


def build_output_corpus(out_filename, raw_dir, total_list):
    # open csv output file
    fp = open(out_filename, 'wb')
    writer = csv.writer(fp, delimiter=',', quotechar='"', escapechar='\\',
                        quoting=csv.QUOTE_ALL)

    # write header row
    writer.writerow(['Topic', 'Sentiment', 'TweetId', 'TweetDate', 'TweetText'])

    # parse all downloaded tweets
    missing_count = 0
    for item in total_list:

        # ensure tweet exists
        if os.path.exists(raw_dir + item[2] + '.json'):

            try:
                # parse tweet
                parsed_tweet = parse_tweet_json(raw_dir + item[2] + '.json')
                full_row = item + parsed_tweet

                # character encoding for output
                for i in range(0, len(full_row)):
                    full_row[i] = full_row[i].encode("utf-8")

                # write csv row
                writer.writerow(full_row)

            except RuntimeError:
                print
                '--> bad data in tweet #' + item[2]
                missing_count += 1

        else:
            print
            '--> missing tweet #' + item[2]
            missing_count += 1

    # indicate success
    if missing_count == 0:
        print
        '\nSuccessfully downloaded corpus!'
        print
        'Output in: ' + out_filename + '\n'
    else:
        print
        '\nMissing %d of %d tweets!' % (missing_count, len(total_list))
        print
        'Partial output in: ' + out_filename + '\n'

    return


def main():
    # get user parameters
    user_params = get_user_params()
    dump_user_params(user_params)

    # get fetch list
    total_list = read_total_list(user_params['inList'])
    fetch_list = purge_already_fetched(total_list, user_params['rawDir'])

    with open('fetch_list', 'w+') as fch:
        for f in fetch_list:
            fch.write(str(f)+'\n')
    sys.exit(1)
    # start fetching data from twitter
    download_tweets(fetch_list, user_params['rawDir'])

    # second pass for any failed downloads
    print
    '\nStarting second pass to retry any failed downloads';
    fetch_list = purge_already_fetched(total_list, user_params['rawDir'])
    download_tweets(fetch_list, user_params['rawDir'])

    # build output corpus
    #build_output_corpus(user_params['outList'], user_params['rawDir'], total_list)

    return


if __name__ == '__main__':
    #build_user_info()
    main()
#data = download_user_info('376287327')
