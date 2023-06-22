import numpy as np
from fastapi import FastAPI, Form
import pandas as pd
from starlette.responses import HTMLResponse
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import re
import pickle
import joblib
# !pip install textblob
# !pip install wordcloud
from textblob import TextBlob
from wordcloud import WordCloud
# !pip install googletrans==3.1.0a0
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import pickle
import joblib
from googleapiclient.discovery import build
from googletrans import Translator

app = FastAPI()

@app.get('/predict', response_class=HTMLResponse)
def take_inp():
    return '''
        <form method="post">
        <input maxlength="100" name="text" type="text" value="Text Emotion to be tested" />
        <input type="submit" />'''

loaded_model = joblib.load("model.sav")
cv = pickle.load(open("cv.pickle", "rb"))

my_stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 's', 't', 'can', 'will', 'just', 'don', 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain']

def remove_mystopwords(sentence):
    tokens = sentence.split(" ")
    tokens_filtered= [word for word in tokens if not word in my_stopwords]
    return (" ").join(tokens_filtered)

youtube = build('youtube', 'v3', developerKey="AIzaSyDnF66Jy21_9TzE-LvJfLK4q9oijV6iM8U")
box = [['Name', 'Comment', 'Time', 'Likes', 'Reply Count']]


def scrape_comments_with_replies(videoId):
     data = youtube.commentThreads().list(part='snippet', videoId=videoId, maxResults='100', textFormat="plainText").execute()
     for i in data["items"]:
          name = i["snippet"]['topLevelComment']["snippet"]["authorDisplayName"]
          comment = i["snippet"]['topLevelComment']["snippet"]["textDisplay"]
          published_at = i["snippet"]['topLevelComment']["snippet"]['publishedAt']
          likes = i["snippet"]['topLevelComment']["snippet"]['likeCount']
          replies = i["snippet"]['totalReplyCount']
          
          box.append([name, comment, published_at, likes, replies])
          
          totalReplyCount = i["snippet"]['totalReplyCount']
     
          if totalReplyCount > 0:
          
               parent = i["snippet"]['topLevelComment']["id"]
               
               data2 = youtube.comments().list(part='snippet', maxResults='100', parentId=parent,
                                             textFormat="plainText").execute()
          
               for i in data2["items"]:
                    name = i["snippet"]["authorDisplayName"]
                    comment = i["snippet"]["textDisplay"]
                    published_at = i["snippet"]['publishedAt']
                    likes = i["snippet"]['likeCount']
                    replies = ""

                    box.append([name, comment, published_at, likes, replies])

     while ("nextPageToken" in data):
     
          data = youtube.commentThreads().list(part='snippet', videoId=videoId, pageToken=data["nextPageToken"],
                                        maxResults='100', textFormat="plainText").execute()
                                        
          for i in data["items"]:
               name = i["snippet"]['topLevelComment']["snippet"]["authorDisplayName"]
               comment = i["snippet"]['topLevelComment']["snippet"]["textDisplay"]
               published_at = i["snippet"]['topLevelComment']["snippet"]['publishedAt']
               likes = i["snippet"]['topLevelComment']["snippet"]['likeCount']
               replies = i["snippet"]['totalReplyCount']

               box.append([name, comment, published_at, likes, replies])

               totalReplyCount = i["snippet"]['totalReplyCount']

               if totalReplyCount > 0:
                    
                    parent = i["snippet"]['topLevelComment']["id"]

                    data2 = youtube.comments().list(part='snippet', maxResults='100', parentId=parent,
                                                  textFormat="plainText").execute()

                    for i in data2["items"]:
                         name = i["snippet"]["authorDisplayName"]
                         comment = i["snippet"]["textDisplay"]
                         published_at = i["snippet"]['publishedAt']
                         likes = i["snippet"]['likeCount']
                         replies = ''

                         box.append([name, comment, published_at, likes, replies])

     df = pd.DataFrame({'Name': [i[0] for i in box], 'Comment': [i[1] for i in box], 'Time': [i[2] for i in box],
                         'Likes': [i[3] for i in box], 'Reply Count': [i[4] for i in box]})

     sql_vids = pd.DataFrame([])

     sql_vids = sql_vids.append(df, ignore_index = True)

     return sql_vids

translator = Translator()

def Translate(text):
    translated_text = translator.translate(text)

    return translated_text.text


def cleanTxt(text):
    text = re.sub(r'[^\w]', ' ', str(text))
    return text

def predict_sentiment(text):
    text = text.lower()
    text = remove_mystopwords(text)
    cmt = cv.transform([text])
    prediction = loaded_model.predict(cmt)[0]
    # print(text, prediction)
    return prediction

@app.post('/predict')
def predict(text:str = Form(...)):

     videolink=text
     videoId=""
     identifiers =["?v=","&v=","v%3D","/v/","/vi/","/embed/","youtu.be/","/e/"]

     idx=-1
     for i in identifiers:
          idx=videolink.find(i)
          if idx!=-1:
               videoId=videolink[idx+len(i):idx+len(i)+11]
               break


     data = pd.DataFrame()
     data = scrape_comments_with_replies(videoId)


     data['Comment'] = data['Comment'].apply(cleanTxt)
     # data['Comment'] = data['Comment'].apply(Translate)

     negative = 0
     positive = 0
     neutral = 0
     for comment in data['Comment']:
          if predict_sentiment(comment) == 1:
               positive += 1
          elif predict_sentiment(comment) == -1:
               negative += 1
          else:
               neutral += 1
     allWords = ' '.join( [cmts for cmts in data['Comment']])
     wordCloud = WordCloud(width = 500, height = 300, random_state = 21, max_font_size = 119).generate(allWords)
     wordCloud.to_file('N.png')
   

     return { #return the dictionary for endpoint
         "Video Link": text,
         "Positive Comments": positive,
         "Negative Comments": negative,
         "Neutral Comments": neutral,
        #  "Probability": probability
    }