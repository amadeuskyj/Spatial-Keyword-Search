import streamlit as st
import os
import pandas as pd
import numpy as np
import ast
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import altair as alt
import analysis

from datetime import datetime
from dateutil import rrule

def app(df, filtered_df, es, index_name):
    st.title('Visualisation')

    #Basic Scatter plot
    st.write("Scatterplot:")
    fig, ax = plt.subplots()
    #sns.scatterplot(data = filtered_df, x="longitude", y="latitude")
    #st.pyplot(fig)
    fig = px.scatter(filtered_df, x="longitude", y="latitude", hover_data=["tweet_text"], width=800, height=600)
    st.plotly_chart(fig)

    #Scatter plot on map
    st.write("Map Plot:")
    px.set_mapbox_access_token("pk.eyJ1IjoiYW1hZGV1c2t5aiIsImEiOiJja290bTdtMmUwOXh5Mm9xbXc0dzBpbGR4In0.ZQYWX00qp75XObfeRbk5gg")
    fig = px.scatter_mapbox(filtered_df,
                            lat="latitude",
                            lon="longitude",
                            hover_data=["tweet_text"],
                            height=600,
                            width=800,
                            zoom=analysis.get_zoom(filtered_df['latitude'], filtered_df['longitude'], 800/700),
                            color_discrete_sequence=["#aa0000"]
                            )
    st.plotly_chart(fig)

    #Bar chart of number of tweets
    st.write("Bar Chart:")
    tweet_groupby = filtered_df.groupby([filtered_df['date']])['tweet_id'].count()
    flat_tweet_groupby = tweet_groupby.reset_index()
    start_date = datetime.strptime(filtered_df['date'].min(), '%Y-%m-%d')
    end_date = datetime.strptime(filtered_df['date'].max(), '%Y-%m-%d')
    dummy_index = []
    for dtime in rrule.rrule(rrule.DAILY, dtstart=start_date, interval = 1, until=end_date):
        dummy_index.append(datetime.strftime(dtime, '%Y-%m-%d'))
    for i in dummy_index:
        if i not in tweet_groupby.index:
            flat_tweet_groupby = flat_tweet_groupby.append(
                                {'date': i, 'tweet_id': 0}, ignore_index=True)
    tweet_groupby = flat_tweet_groupby.sort_values(['date']).set_index(['date'])
    #st.bar_chart(tweet_groupby)
    fig = px.bar(tweet_groupby, x=tweet_groupby.index, y='tweet_id')
    st.plotly_chart(fig, use_container_width=True)

    #Common words bar chart
    st.write("Top-k Most Common Words:")
    num_words = st.slider("Pick number of words", 1, 20, 10)
    body = {
        "aggs": {
            "common_words": {
                "terms": {
                    "field": "tweet_text",
                    "size": num_words
                }
            }
        }
    }
    res = es.search(index=index_name, body=body)
    buckets = res['aggregations']['common_words']['buckets']
    agg_df = pd.DataFrame(buckets)
    agg_df = agg_df.set_index('key').sort_values(['doc_count'], ascending=False)
    fig = px.bar(agg_df, x=agg_df.index, y='doc_count')
    st.plotly_chart(fig)