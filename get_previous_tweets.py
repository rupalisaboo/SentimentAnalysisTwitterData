"""
Read in the input csv file, get the tweet id and look for and parse the
tweet and friends_data files, get the [tweet text] and [the concatenation of the friends description]
The friends descriptions are separated by [ ^|^ ].
It outputs one tab-separated-file with the format:
['company', 'sentiment', 'text', 'ID', 'friends_desc', 'user', 'date']
"""

import os, csv, json, codecs
import csv, getpass, json, os, time, sys
import oauth2 as oauth
import urllib2 as urllib

api_key = "jNjhXYVfvuKenyWwbccTgkcjX"
api_secret = "kO0YTvewzCDSf8YXFj2ICoee4HtZ7wkadES8zKdUgTpXaNbYmZ"
access_token_key = "289034756-XBqlBm6poMcSCR0yOdDVBzZ8K8O3aeR1uKVN0kam"
access_token_secret = "OimWPfFjbnD0F7AXBhcqP4Z0MyT0GCwkRMs9OJ58l2kQ2"
oauth_token = oauth.Token(key=access_token_key, secret=access_token_secret)
oauth_consumer = oauth.Consumer(key=api_key, secret=api_secret)

signature_method_hmac_sha1 = oauth.SignatureMethod_HMAC_SHA1()

http_method = "GET"
_debug = 0
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


def download_tweets(tweet_id, tweet_time):
    # stay within rate limits
    max_tweets_per_hr = 180
    download_pause_sec = 900 / max_tweets_per_hr
    # pull data
    url = 'https://api.twitter.com/1.1/statuses/show.json?id=' + tweet_id
    response = twitterreq(url, "GET", [])
    # urllib.urlretrieve( url, raw_dir + tweet_id + '.json' )
    raw_dir = 'user_id/'
    '''
    with open(raw_dir + tweet_id + '.json', 'w') as fh:
        fh.write(response.read())
    '''
    # stay in Twitter API rate limits
    print('Downloaded: '+tweet_id + '.json')
    response_str = response.read()
    response_json = json.loads(response_str)
    if 'error' in response_json or 'errors' in response_json:
        print('Error in downloaded data')
        print('pausing %d sec to obey Twitter API rate limits' % (download_pause_sec))
        time.sleep(download_pause_sec)
        return 0
    elif 'id_str' not in response_json:
        print('Error in downloaded data, id_str missing')
        print('pausing %d sec to obey Twitter API rate limits' % (download_pause_sec))
        time.sleep(download_pause_sec)
        return 0
    else:
        user_id = response_json['user']['id_str']
        last_tweet_url = 'https://api.twitter.com/1.1/statuses/user_timeline.json?user_id=' + user_id + '&count=2' + '&max_id=' + tweet_id
        response_last_tweet = twitterreq(last_tweet_url, "GET", [])
        response_last_tweet_str = response_last_tweet.read()
        response_last_tweet_json = json.loads(response_last_tweet_str)
        if 'error' in response_last_tweet_json or 'errors' in response_last_tweet_json:
            print('Error in downloaded data')
            print('pausing %d sec to obey Twitter API rate limits' % (download_pause_sec))
            time.sleep(download_pause_sec)
            return 0
        else:
            print(response_last_tweet_json)
            with open(raw_dir + tweet_id + '.json', 'w') as fh:
                if (response_last_tweet_json):
                    if (response_last_tweet_json[1]):
                        fh.write(response_last_tweet_json[1]['text'])
                else:
                    print('Error in downloaded data, text missing')
                    print('pausing %d sec to obey Twitter API rate limits' % (download_pause_sec))
                    time.sleep(download_pause_sec)
                    return 0
            print('pausing %d sec to obey Twitter API rate limits' % (download_pause_sec))
            time.sleep(download_pause_sec)


def main():
    input_file = 'resources/test_tweets_new.csv'
    positive_label_count = 0
    negative_label_count = 0
    max_positive = 2500
    max_negative = 2500
    # read input corpus file
    with open(input_file, 'r') as input_fh:
        rows = (line.split(',') for line in input_fh)
        for row in rows:
            tweet_id = row[1].replace('"', '').strip()
            time = row[2].replace('"', '').strip()
            label = row[0].replace('"', '').strip()
            if label=='2':
                continue
            if (label=='0') & (negative_label_count==max_negative):
                continue
            if (label=='4') & (positive_label_count==max_positive):
                continue

            count = download_tweets(tweet_id, time)
            if label=='0':
                negative_label_count += count
            elif label=='4':
                positive_label_count += count




if __name__ == '__main__':
    main()