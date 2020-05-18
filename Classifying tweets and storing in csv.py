import re 
import tweepy 
from tweepy import OAuthHandler 
from textblob import TextBlob 
import csv
class TwitterClient(object): 
    ''' 
    Generic Twitter Class for sentiment analysis. 
    '''
    def __init__(self): 
        ''' 
        Class constructor or initialization method. 
        '''
        # keys and tokens from the Twitter Console 
        consumer_key = 'hy9piO6ZE0yXUR1mg7Cbj1X0F'
        consumer_secret = 'aEx4bD2ZrRpyp1LGaIfEZ90Phz4ZEgZeDRfrHWLuN5vs1pvSvO'
        access_token = '874876545843818496-VlLg69MbOkl5b7tlbeh5xfzu7IEUMRq'
        access_token_secret = 'ePpJxvLez6wkYgCFUTVV6u3bfWmQgvAaS1DoLaOtdGkZN'
  
        # attempt authentication 
        try: 
            # create OAuthHandler object 
            self.auth = OAuthHandler(consumer_key, consumer_secret) 
            # set access token and secret 
            self.auth.set_access_token(access_token, access_token_secret) 
            # create tweepy API object to fetch tweets 
            self.api = tweepy.API(self.auth) 
        except: 
            print("Error: Authentication Failed") 
  
    def clean_tweet(self, tweet): 
        ''' 
        Utility function to clean tweet text by removing links, special characters 
        using simple regular expression. 
        '''
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) |(\w+:\/\/\S+)", " ", tweet).split()) 
  
    def get_tweet_sentiment(self, tweet): 
        ''' 
        Utility function to classify sentiment of passed tweet 
        using textblob's sentiment method 
        '''
        # create TextBlob object of passed tweet text 
        analysis = TextBlob(self.clean_tweet(tweet)) 
        # set sentiment 
        if analysis.sentiment.polarity > 0: 
            return 'positive'
        else:
            return 'negative'
  
    def get_tweets(self, query, count ): 
        ''' 
        Main function to fetch tweets and parse them. 
        '''
        # empty list to store parsed tweets 
        tweets = [] 
  
        try: 
            # call twitter api to fetch tweets 
            fetched_tweets = self.api.search(q = query, count = count) 
  
            # parsing tweets one by one 
            for tweet in fetched_tweets: 
                # empty dictionary to store required params of a tweet 
                parsed_tweet = {} 
  
                # saving text of tweet 
                parsed_tweet['text'] = tweet.text 
                # saving sentiment of tweet 
                parsed_tweet['sentiment'] = self.get_tweet_sentiment(tweet.text) 
  
                # appending parsed tweet to tweets list 
                if tweet.retweet_count > 0: 
                    # if tweet has retweets, ensure that it is appended only once 
                    if parsed_tweet not in tweets: 
                        tweets.append(parsed_tweet) 
                else: 
                    tweets.append(parsed_tweet) 
  
            # return parsed tweets 
            return tweets 
  
        except tweepy.TweepError as e: 
            # print error (if any) 
            print("Error : " + str(e))
import os
api = TwitterClient() 
    # calling function to get tweets 
tweets = api.get_tweets(query = 'realDonaldTrump', count = 1000) 
  
    # picking positive tweets from tweets 
ptweets = [tweet for tweet in tweets if tweet['sentiment'] == 'positive'] 
    # percentage of positive tweets 
print("Positive tweets percentage: {} %".format(100*len(ptweets)/len(tweets))) 
    # picking negative tweets from tweets 
ntweets = [tweet for tweet in tweets if tweet['sentiment'] == 'negative'] 
    # percentage of negative tweets 
print("Negative tweets percentage: {} %".format(100*len(ntweets)/len(tweets))) 
    # percentage of neutral tweets 
   # print("Neutral tweets percentage: {} %" .format(100*len(tweets - ntweets - ptweets)/len(tweets))) 
  
    # printing first 5 positive tweets 
with open('E:/final project 4th yr/Predicting-Cyberbulling-on-Twitter-master/test1.csv', 'a', encoding='utf-8') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(['Tweet','Text Label'])
print("\n\nPositive tweets:") 
for tweet in ptweets: 
    print(tweet['text'])
    row=[tweet['text'],'Non-Bullying']
    with open('E:/final project 4th yr/Predicting-Cyberbulling-on-Twitter-master/test1.csv', 'a', encoding='utf-8') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)
   
  
print("\n\nNegative tweets:") 
for tweet in ntweets: 
    print(tweet['text'])
    row=[tweet['text'],'Bullying']
    with open('E:/final project 4th yr/Predicting-Cyberbulling-on-Twitter-master/test1.csv', 'a', encoding='utf-8') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)
