from flask import Flask, redirect, render_template, url_for
import os
import pickle
# bert
from transformers import pipeline
import tweepy
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('vader_lexicon')
from nltk.stem import WordNetLemmatizer
# lex
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import re
import csv
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/last')
def last():
    if os.path.exists("./static/emotion.png"):
        return render_template('analysis.html')
    else:
        redirect(url_for('home'))

@app.route('/recent')
def recent():
    with open('saved_dictionary.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
    tweets = loaded_dict
    print("---LEX---")
    lex(tweets['searched_tweets'])
    print("---BERT---")
    bert(tweets['tweets'])
    if os.path.exists("./static/emotion.png"):
        return render_template('analysis.html')
    else:
        redirect(url_for('home'))

def deleteImg(path):
    if os.path.exists(path):
        os.remove(path)

def getTweets():
    access_token, access_token_secret, consumer_key, consumer_secret = keys['2'].values()

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)


    query = '#COVID19 OR coronavirus OR covid OR COVID-19 OR pandemic'
    max_tweets = 1000
    tweets = []
    searched_tweets = []
    for tweet in tweepy.Cursor(api.search_tweets,
                           q=query,
                           lang="en",
                           tweet_mode='extended').items(max_tweets):
        searched_tweets.append(tweet.full_text)
        tweets.append({"text":tweet.full_text, "created_at": tweet.created_at})

    final = {"tweets": tweets, "searched_tweets": searched_tweets}
    return final

def bert(tweets):
    classifier = pipeline("text-classification",model='j-hartmann/emotion-english-distilroberta-base', return_all_scores=True)
    # classifier = pipeline("text-classification",model='digitalepidemiologylab/covid-twitter-bert-v2', return_all_scores=True)

    def check_emotion(text):
        prediction = classifier(text)[0]
        max_emo = max(prediction, key=lambda x: x['score'])['label']
        return max_emo

    emotion_dict = {
        "anger": 0,
        "disgust": 0,
        "fear": 0,
        "joy": 0,
        "neutral": 0,
        "sadness": 0,
        "surprise": 0
    }

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

    for idx, tweet in enumerate(tweets):
        tweet_text = preprocess_tweet(tweet["text"])
        blob = TextBlob(tweet_text)
        polarity, subjectivity = blob.sentiment
        curr_emotion = check_emotion(tweet_text)
        emotion_dict[curr_emotion] += 1
        tweet['emotion'] = curr_emotion
        tweet['polarity'] = polarity
        tweet['subjectivity'] = subjectivity


    f2 = plt.figure()
    labels = ["Anger", "Disgust", "Fear", "Joy", "Neutral", "Sadness", "Surprise"]
    sizes = list(emotion_dict.values())
    plt.bar(labels, sizes,  color='#ff7b00')

    plt.title('Emotion Analysis Results')
    plt.xlabel('Emotion')
    plt.ylabel('Number of Tweets')
    plt.savefig('./static/emotion.png', bbox_inches='tight')
    f2.clear()

def lex(searched_tweets):
    def preprocess_tweet(tweet):
        tweet = re.sub(r"http\S+", "", tweet)
        tweet = re.sub(r"@\S+", "", tweet)
        tweet = re.sub(r"[^\w\s]", "", tweet)
        tokens = word_tokenize(tweet.lower())
        stop_words = set(stopwords.words('english') + list(string.punctuation))
        tokens = [token for token in tokens if token not in stop_words and len(token) > 2]

        return tokens

    tokens = [preprocess_tweet(tweet) for tweet in searched_tweets]

    dictionary = Dictionary(tokens)
    corpus = [dictionary.doc2bow(tokens) for tokens in tokens]
    corpus
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=5, passes=10)
    lda_model
    filename_1 = './static/topics.csv'
    with open(filename_1, mode='w', encoding='utf-8') as file:
        writer = csv.writer(file)
        for idx, topic in lda_model.print_topics(-1):
            words = topic.split('+')
            words = [word.split('*')[1].replace('"', '').strip() for word in words]
            writer.writerow([f'Topic {idx}'] + words)

    print(f"Saved results ")
    def get_sentiment_polarity(tweet):
        blob = TextBlob(tweet)
        return blob.sentiment.polarity
    def get_sentiment_subjectivity(tweet):
        blob = TextBlob(tweet)
        return blob.sentiment.subjectivity
    filename = "./static/tweet_sentiments.csv"

    with open(filename, "w", encoding='utf-8') as file:
        sentiment_scores = []
        writer = csv.writer(file)
        writer.writerow(["Tweet", "Polarity", "Subjectivity"])

        for tweet in searched_tweets:
            tokens = preprocess_tweet(tweet)
            sentiment_polarity = get_sentiment_polarity(tweet)
            sentiment_subjectivity = get_sentiment_subjectivity(tweet)
            sentiment_scores.append({'tweet': tweet, 'polarity': sentiment_polarity, 'subjectivity': sentiment_subjectivity})
            writer.writerow([tweet, sentiment_polarity, sentiment_subjectivity])

    print(f"Saved results ")
    with open(filename, "r", encoding='utf-8') as file:
        reader = csv.DictReader(file)
        sentiment_scores = [{'tweet': row['Tweet'], 'polarity': float(row['Polarity']), 'subjectivity': float(row['Subjectivity'])} for row in reader]
    polarity_scores = [score['polarity'] for score in sentiment_scores]
    f1 = plt.figure()
    plt.hist(polarity_scores, bins=20)
    plt.title('Sentiment Polarity Histogram')
    plt.xlabel('Polarity Score')
    plt.ylabel('Frequency')
    plt.savefig('./static/lex_sph.png', bbox_inches='tight')
    f1.clear()
    f2 = plt.figure()
    subjectivity_scores = [score['subjectivity'] for score in sentiment_scores]
    plt.hist(subjectivity_scores, bins=20)
    plt.title('Sentiment Subjectivity Histogram')
    plt.xlabel('Subjectivity Score')
    plt.ylabel('Frequency')
    plt.savefig('./static/lex_ssh.png', bbox_inches='tight')
    f2.clear()
    with open(filename, "r", encoding='utf-8') as file:
        reader = csv.DictReader(file)
        sentiment_scores = [{'tweet': row['Tweet'], 'polarity': float(row['Polarity']), 'subjectivity': float(row['Subjectivity'])} for row in reader]
    polarity_scores = [score['polarity'] for score in sentiment_scores]
    subjectivity_scores = [score['subjectivity'] for score in sentiment_scores]
    f3 = plt.figure()
    plt.scatter(polarity_scores, subjectivity_scores)
    plt.title('Sentiment Polarity vs Subjectivity')
    plt.xlabel('Polarity Score')
    plt.ylabel('Subjectivity Score')
    plt.savefig('./static/lex_spvss.png', bbox_inches='tight')
    f3.clear()
    tweet_array = []
    with open('./static/tweet_sentiments.csv', mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        sentiment_scores = [{'tweet': row['Tweet'], 'polarity': float(row['Polarity']), 'subjectivity': float(row['Subjectivity'])} for row in reader]
        for sentiment in sentiment_scores:
            tweet_array.append(sentiment['tweet'])
    # print(tweet_array)
    positive_tweets = []
    neutral_tweets = []
    negative_tweets = []

    with open('./static/tweet_sentiments.csv', mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        sentiment_scores = [{'tweet': row['Tweet'], 'polarity': float(row['Polarity']), 'subjectivity': float(row['Subjectivity'])} for row in reader]
        for sentiment in sentiment_scores:
            tweet_array.append(sentiment['tweet'])

    for tweet in tweet_array:
        blob = TextBlob(tweet)
        polarity_score = blob.sentiment.polarity

        if polarity_score > 0:
            positive_tweets.append(tweet)
        elif polarity_score == 0:
            neutral_tweets.append(tweet)
        else:
            negative_tweets.append(tweet)
    print()
    print()
    print("-----------")
    print(f"Number of Positive Tweets: {len(positive_tweets)}")
    print(f"Number of Neutral Tweets: {len(neutral_tweets)}")
    print(f"Number of Negative Tweets: {len(negative_tweets)}")
    print("-----------")
    print()
    print()
    sid = SentimentIntensityAnalyzer()
    emotion_categories = {
        'fear': {'keywords': ["covid", "pandemic", "virus", "lockdown", "quarantine", "isolation", "symptoms", "death", "uncertainty"], 'count': 0},
        'happiness': {'keywords': ["vaccine", "hope", "recovery", "community", "support", "gratitude", "positivity", "innovation"], 'count': 0},
        'sorrow': {'keywords': ["loss", "grief", "mourning", "death", "isolation", "distress", "trauma", "loneliness"], 'count': 0},
        'neutral': {'keywords': [], 'count': 0}
    }
    tweet_sentiments = []
    with open('./static/tweet_sentiments.csv', 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            tweet_sentiments.append({'tweet': row['Tweet'], 'polarity': float(
                row['Polarity']), 'subjectivity': float(row['Subjectivity'])})

    for tweet in tweet_sentiments:
        sentiment_scores = sid.polarity_scores(tweet['tweet'])
        dominant_emotion = None
        for emotion in emotion_categories.keys():
            keywords = emotion_categories[emotion]['keywords']
            if any(keyword in tweet['tweet'].lower() for keyword in keywords):
                dominant_emotion = emotion
                break
        if dominant_emotion is None:
            if sentiment_scores['compound'] >= 0.05:
                dominant_emotion = 'happiness'
            elif sentiment_scores['compound'] <= -0.05:
                dominant_emotion = 'sorrow'
            else:
                dominant_emotion = 'neutral'
        emotion_categories[dominant_emotion]['count'] += 1

    # for emotion in emotion_categories.keys():
        # print(emotion.capitalize() + ': ' +
            # str(emotion_categories[emotion]['count']))

    emotion_counts = [emotion_categories[emotion]['count']
                    for emotion in emotion_categories.keys()]
    emotion_labels = [emotion.capitalize()
                    for emotion in emotion_categories.keys()]
    f4 = plt.figure()
    plt.pie(emotion_counts, labels=emotion_labels, autopct='%1.1f%%')
    plt.title('Distribution of Covid 19 Tweets across Emotions:')
    plt.savefig('./static/lex_emotions.png', bbox_inches='tight')
    f4.clear()

keys = {
    "1": {
        "access_token":'1259453588524408834-4zQyoBtECTnYru2OjyjyuFWCx8FSod',
        "access_token_secret":'CJQkT8GTvdqCEEDNxWDTeTJGSlwVpBxEK8PNGDxKUDPnf',
        "consumer_key":'QDSOD0qvG5AE7skDsega7HkeY',
        "consumer_secret":'9ybUtOloNk9C7m3OckSDj39r356ysT4W8Jbfsq9eY1WYFtfVKm',
    },
    "2":{
        "access_token":'950034457494343681-zfCgrSvn73y11HfqUmaU56Sj67ME9gy',
        "access_token_secret":'OIHbLVz3buuofmu4toUNYPG9CCSCJdr21tj4wcHxAiDXw',
        "consumer_key":'eUTGLbynkp25KvQdNPgZvNJIy',
        "consumer_secret":'qqQDNOX2IZyjVvzF3q1Esh2Y8sXibfH4eYJ5gPUFlgPsePfQQ3'
    }
}


@app.route('/analysis')
def analysis():
    deleteImg("./static/emotion.png")
    deleteImg("./static/sentiments.png")
    deleteImg("./static/lex_emotions.png")
    deleteImg("./static/lex_sph.png")
    deleteImg("./static/lex_ssh.png")
    deleteImg("./static/lex_spvss.png")

    tweets = getTweets()
    print("---LEX---")
    lex(tweets['searched_tweets'])
    print("---BERT---")
    bert(tweets['tweets'])
    if os.path.exists("./static/emotion.png"):
        with open('saved_dictionary.pkl', 'wb') as f:
            pickle.dump(tweets, f)
        return render_template('analysis.html')
    else:
        redirect(url_for('home'))

if __name__ == '__main__':
    app.run()






