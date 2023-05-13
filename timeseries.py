#%%
import datetime
from transformers import pipeline
import os
import pandas as pd
import tweepy
from textblob import TextBlob
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
#%%
classifier = pipeline("text-classification",model='j-hartmann/emotion-english-distilroberta-base', return_all_scores=True)

def check_emotion(text):
    prediction = classifier(text)[0]
    max_emo = max(prediction, key=lambda x: x['score'])['label']
    return max_emo

#%%
access_token = '950034457494343681-zfCgrSvn73y11HfqUmaU56Sj67ME9gy'
access_token_secret = 'OIHbLVz3buuofmu4toUNYPG9CCSCJdr21tj4wcHxAiDXw'
consumer_key = 'eUTGLbynkp25KvQdNPgZvNJIy'
consumer_secret = 'qqQDNOX2IZyjVvzF3q1Esh2Y8sXibfH4eYJ5gPUFlgPsePfQQ3'
# consumer_key = 'QDSOD0qvG5AE7skDsega7HkeY'
# consumer_secret = '9ybUtOloNk9C7m3OckSDj39r356ysT4W8Jbfsq9eY1WYFtfVKm'
# access_token = '1259453588524408834-4zQyoBtECTnYru2OjyjyuFWCx8FSod'
# access_token_secret = 'CJQkT8GTvdqCEEDNxWDTeTJGSlwVpBxEK8PNGDxKUDPnf'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)


#%%
def get_date(date):
    return (datetime.datetime.now() - datetime.timedelta(days=date)).strftime("%Y-%m-%d")

#%%
query = '#COVID19 OR coronavirus OR covid OR COVID-19'
max_tweets = 500
tweets = []
positive_tweets = 0
negative_tweets = 0
neutral_tweets = 0

for i in range(0,10):
    end_date = get_date(i)
    cur_tweet=[]
    for tweet in tweepy.Cursor(api.search_tweets,
                            q=query,
                            lang="en",
                            until=end_date,
                            tweet_mode='extended').items(max_tweets):
        cur_tweet.append({"text":tweet.full_text, "created_at": tweet.created_at})
    tweets.append(cur_tweet)
tweets
#%%
def preprocess_tweet(tweet):
    tweet = re.sub(r"http\S+|www\S+|https\S+", "", tweet, flags=re.MULTILINE)
    tweet = re.sub(r'\@\w+|\#', "", tweet)
    tweet = re.sub(r'\d+', '', tweet)
    tweet = re.sub(r'[^\w\s]', '', tweet)
    tweet = tweet.strip()
    tokens = tweet.lower().split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    cleaned_tweet = " ".join(tokens)
    return cleaned_tweet
#%%
emo_time = []
for i in range(0,10):
    emotion_dict = {
    "anger": 0, 
    "disgust": 0, 
    "fear": 0, 
    "joy": 0, 
    "neutral": 0, 
    "sadness": 0, 
    "surprise": 0
}
    print(i)
    for tweet in tweets[i]:
        tweet_text = preprocess_tweet(tweet["text"])
        blob = TextBlob(tweet_text)
        polarity, subjectivity = blob.sentiment
        curr_emotion = check_emotion(tweet_text)
        emotion_dict[curr_emotion] += 1
        tweet['emotion'] = curr_emotion
        tweet['polarity'] = polarity
        tweet['subjectivity'] = subjectivity

    labels = ["Anger", "Disgust", "Fear", "Joy", "Neutral", "Sadness", "Surprise"]
    sizes = list(emotion_dict.values())
    emo_time.insert(0, emotion_dict)
emo_time
emo_df = pd.DataFrame(emo_time)
index_lst = []
for i in range(0,10):
    index_lst.append(get_date(9-i))
emo_df.index = index_lst
f3 = plt.figure()
emo_df.plot(figsize=(10,6), title='Emotions over Last 10 days')
plt.xlabel('Time Period')
plt.ylabel('Emotion Score')
plt.show()