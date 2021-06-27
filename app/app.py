import streamlit as st
import os
import pandas as pd
import numpy as np
import ast
import seaborn as sns
import matplotlib.pyplot as plt
import elasticsearch
from elasticsearch import Elasticsearch
from elasticsearch import helpers
from datetime import datetime
from dateutil import rrule
import home, data, visualisation, analysis

def get_coordinates(coordinates):
    latitudes = []
    longitudes = []
    for i in coordinates:
        longitudes.append(i[0])
        latitudes.append(i[1])
    long = sum(longitudes)/len(longitudes)
    lat = sum(latitudes)/len(latitudes)
    return [lat, long]

def getDate(date_str):
    datetime_object = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
    datetime_string = datetime.strftime(datetime_object, '%Y-%m-%d')
    return datetime_string

def tweet_generator(df, index_name):
    df_iterator = df.iterrows()
    for index, document in df_iterator:
        yield {
                "_index": index_name,
                "_type": "_doc",
                "_id" : f"{document['tweet_id']}",
                "_source": document.to_dict(),
            }

@st.cache(hash_funcs={elasticsearch.client.Elasticsearch: id})
def init_es():
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, '../elasticsearch-7.10.2/bin/elasticsearch.bat')
    os.popen(filename)
    print("Elasticsearch Initialised")
    return Elasticsearch(http_compress=True)

@st.cache(hash_funcs={elasticsearch.client.Elasticsearch: id})
def create_index(es, index_name, tweet_df):
  if es.indices.exists(index=index_name):
    es.indices.delete(index_name)
    print("Old index deleted")
  body={
      'settings': {
        'number_of_shards': 2,
        'number_of_replicas': 0,
        'index': {
          'sort.field': 'time',
          'sort.order': 'asc'
        },

        # custom analyzer
        'analysis': {
          'analyzer': {
            'tweet_analyzer': {
              'type': 'custom',
              'tokenizer': 'standard',
              'filter': ['lowercase', 'english_stop', 'porter_stem']
            }
          },
          'filter': {
            'english_stop': { 
              'type': 'stop',
              'stopwords': '_english_'
            }
          }
        }
      },
      'mappings': {
        'properties': {
          'tweet_text': {
            'type': 'text',
            'fielddata': True,
            'analyzer': 'tweet_analyzer',
            'search_analyzer': 'tweet_analyzer'
          },
          'time': {
            'type': 'date',
            'format': 'yyyy-MM-dd HH:mm:ss'
          }
        }
      }
    }
  es.indices.create(index=index_name, body=body)
  print("Inverted Index '" + index_name + "' created")
  helpers.bulk(es, tweet_generator(tweet_df, index_name))
  print("Data added to inverted index")

@st.cache(hash_funcs={list: lambda _: None}, allow_output_mutation=True)
def preprocess(tweet_df):
  filtered_df = tweet_df[["tweet_id", "time", "text", "place"]]
  filtered_df = filtered_df.dropna()
  filtered_df = filtered_df.rename(columns = {"text": "tweet_text"})
  filtered_df['longitude'] = filtered_df['place'].apply(lambda x: get_coordinates(ast.literal_eval(x)['bounding_box']['coordinates'][0])[1])
  filtered_df['latitude'] = filtered_df['place'].apply(lambda x: get_coordinates(ast.literal_eval(x)['bounding_box']['coordinates'][0])[0])
  filtered_df['coordinates'] = filtered_df['place'].apply(lambda x: get_coordinates(ast.literal_eval(x)['bounding_box']['coordinates'][0]))
  filtered_df['date'] = filtered_df['time'].apply(lambda x: getDate(x))
  filtered_df['datetime'] = filtered_df['time'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
  return filtered_df

#Initializations
es = init_es()

apps = []
apps.append({"title": "Home", "function": home.app})
apps.append({"title": "Data", "function": data.app})
apps.append({"title": "Visualisation", "function": visualisation.app})
apps.append({"title": "Keyword Search", "function": analysis.app})

#Side Bar
nav = st.sidebar.selectbox('Navigation', apps, format_func=lambda app: app['title'])
default_file = '../data/df_tweet_taal.csv'
uploaded_file = st.sidebar.file_uploader("Choose a file", help="Select a Twitter tweet dataset in CSV format")
if uploaded_file is None:
    uploaded_file = default_file

df = pd.read_csv(uploaded_file)
filtered_df = preprocess(df)
#Inverted index
index_name = uploaded_file.name.split(".")[0] + "-index" if uploaded_file != default_file else "tweet-index"
create_index(es, index_name, filtered_df)

st.title("Deep Learning Supported Spatial Keyword Search")

nav['function'](df, filtered_df, es, index_name)
