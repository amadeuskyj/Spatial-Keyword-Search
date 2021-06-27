import streamlit as st
import os
import pandas as pd
import numpy as np
import ast
import time as t
import math
import seaborn as sns
import matplotlib.pyplot as plt
import elasticsearch
from elasticsearch import Elasticsearch
from elasticsearch import helpers
from datetime import datetime, timedelta, date, time
from dateutil import rrule
from DBSCAN import DBSCAN
from numpy import unique, where
import plotly.express as px
import plotly.graph_objects as go

eps = 0.15
min_pts = 9

@st.cache(hash_funcs={elasticsearch.client.Elasticsearch: id})
def make_query(es, index_name, keyword, start, end):
    doc_count = 0
    docs = []
    query = {
        "size": 100,
        "query": {
            "bool": {
                "must": [
                    {"match": {
                        "tweet_text": keyword
                    }},
                    {"range": {
                        "time": {
                            "gte": start,
                            "lt": end
                        }
                    }}
                ]
            }
        }
    }
    # make a search() request to get all docs in the index
    resp = es.search(
        index=index_name,
        body=query,
        scroll='2s'  # length of time to keep search context
    )

    for doc in resp['hits']['hits']:
        print("\n", doc['_id'], doc['_source']['tweet_text'],
            doc['_source']['time'], doc['_score'])
        doc_count += 1
        print("DOC COUNT:", doc_count)
        docs.append(doc)

    # keep track of pass scroll _id
    old_scroll_id = resp['_scroll_id']

    while len(resp['hits']['hits']):
        resp = es.scroll(scroll_id=old_scroll_id, scroll='2s')
        if old_scroll_id != resp['_scroll_id']:
            print("New scroll id: " + resp['_scroll_id'])
        old_scroll_id = resp['_scroll_id']

        print("\nResponse for index:", index_name)
        print("Scroll ID:", resp['_scroll_id'])
        print('Total Hits:', resp["hits"]["total"]["value"])
    
        # iterate over the document hits for each 'scroll'
        for doc in resp['hits']['hits']:
            print("\n", doc['_id'], doc['_source']
              ['tweet_text'], doc['_source']['time'])
            doc_count += 1
            print("DOC COUNT:", doc_count)
            docs.append(doc)
    print("\nTOTAL DOC COUNT:", doc_count)

    results = []
    for hit in docs:
        results.append(hit['_source'])
    result_df = pd.DataFrame(results)
    try:   
        result_df['datetime'] = result_df['time'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    except KeyError:
        print("Make query found no documents...")
    return result_df

@st.cache
def run_clustering(df):
    if len(df.index) == 0:
        return [-1]
    model = DBSCAN(eps, min_pts)
    data = pd.DataFrame(df[["longitude", "latitude"]])
    start = t.time()
    labels = model.predict(data)
    end = t.time()
    print("DBSCAN executed in " + str(end-start) + " seconds")
    return labels

def date_to_string(dt):
    return datetime.strftime(dt, '%Y-%m-%d %H:%M:%S')

def string_to_date(s):
    return datetime.strptime(s, '%Y-%m-%d %H:%M:%S')

def text_format(text, num_char):
    text_arr = text.split()
    result = ""
    counter = 0
    while len(text_arr) > 0:
        word = text_arr.pop(0)
        if counter + len(word) + 1 > num_char:
            counter = 0
            result += "<br>" + word + " "
        else:
            counter += len(word) + 1
            result += word + " "
    return result

def get_size(size):
    return (size/1000) * 10 + 10

def get_center(ls):
    max = ls.max()
    min = ls.min()
    return (max+min)/2

def get_color(size):
    size = math.log2(size)
    size_percentage = min(size/10, 1)
    size = int(size_percentage*510)
    if size<=255:
        green = 255
        red = size
    else:
        red = 255
        green = 255 - (size - red)
    return "rgb(" + str(red) + ", " + str(green) + ", 0)"

def get_zoom(lats, lons, width_to_height: float=2.0):
    # longitudinal range by zoom level (20 to 1)
    # in degrees, if centered at equator
    lon_zoom_range = np.array([
        0.0007, 0.0014, 0.003, 0.006, 0.012, 0.024, 0.048, 0.096,
        0.192, 0.3712, 0.768, 1.536, 3.072, 6.144, 11.8784, 23.7568,
        47.5136, 98.304, 190.0544, 360.0
    ])
    maxlat = lats.max()
    maxlon = lons.max()
    minlat = lats.min()
    minlon = lons.min()
    margin = 2
    height = (maxlat - minlat) * margin * width_to_height
    width = (maxlon - minlon) * margin
    lon_zoom = np.interp(width , lon_zoom_range, range(20, 0, -1))
    lat_zoom = np.interp(height, lon_zoom_range, range(20, 0, -1))
    zoom = round(min(lon_zoom, lat_zoom), 2)
    return zoom

def individual_plot(mapbox_access_token, clusters, labels, result_df, start_ts, end_ts):
    width = 800
    height = 600
    fig = go.Figure()
    for i, cluster in enumerate(clusters):
        row_ix = where(labels == cluster[0])
        fig.add_trace(go.Scattermapbox(
            lat = result_df.loc[row_ix, 'latitude'],
            lon = result_df.loc[row_ix, 'longitude'],
            mode = "markers",
            text = result_df.loc[row_ix, 'tweet_text'],
            name = "Cluster " + str(i + 1),
            hovertemplate = result_df.loc[row_ix, "tweet_text"].apply(lambda x: text_format(x, 100))
        ))
    fig.update_layout(
        title = "Timeframe: " + str(start_ts) + " to " + str(end_ts),
        hovermode = 'closest',
        showlegend = False,
        width = width,
        height = height,
        mapbox = dict(
            accesstoken = mapbox_access_token,
            center = dict(
                lat = get_center(result_df['latitude']),
                lon = get_center(result_df['longitude'])
            ),
            zoom = get_zoom(result_df['latitude'], result_df['longitude'], width/height)
        )
    )
    return fig

def cluster_plot(mapbox_access_token, clusters, labels, result_df, start_ts, end_ts, prev_clusters, prev_labels, prev_result_df):
    width = 800
    height = 600
    fig = go.Figure()
    if len(prev_clusters) != 0:
        for i, cluster in enumerate(prev_clusters):
            row_ix = where(prev_labels == cluster[0])
            fig.add_trace(go.Scattermapbox(
                lat = [prev_result_df.loc[row_ix, 'latitude'].mean()],
                lon = [prev_result_df.loc[row_ix, 'longitude'].mean()],
                mode = "markers",
                hoverinfo = "none",
                marker = go.scattermapbox.Marker(
                    size = get_size(cluster[1]) + 3,
                    opacity = 0.5,
                    color = 'rgb(255, 255, 255)'
                )
            ))
            fig.add_trace(go.Scattermapbox(
                lat = [prev_result_df.loc[row_ix, 'latitude'].mean()],
                lon = [prev_result_df.loc[row_ix, 'longitude'].mean()],
                text = prev_result_df.loc[row_ix[0][0], 'tweet_text'],
                mode = "markers",
                hovertemplate = "Latitude: %{lat: .3f}<br>Longitude: %{lon: .3f}<br>Size: " + str(cluster[1]) + "<br>" + text_format(prev_result_df.loc[row_ix[0][0], "tweet_text"], 100),
                name = "Cluster " + str(i + 1),
                marker = go.scattermapbox.Marker(
                    size = get_size(cluster[1]),
                    opacity = 0.3,
                    color = 'rgb(50, 50, 50)'
                )
            ))
    for i, cluster in enumerate(clusters):
        row_ix = where(labels == cluster[0])
        fig.add_trace(go.Scattermapbox(
            lat = [result_df.loc[row_ix, 'latitude'].mean()],
            lon = [result_df.loc[row_ix, 'longitude'].mean()],
            mode = "markers",
            hoverinfo = "none",
            marker = go.scattermapbox.Marker(
                size = get_size(cluster[1]) + 3,
                opacity = 0.8,
                color = 'rgb(255, 255, 255)'
            )
        ))
        fig.add_trace(go.Scattermapbox(
            lat = [result_df.loc[row_ix, 'latitude'].mean()],
            lon = [result_df.loc[row_ix, 'longitude'].mean()],
            text = result_df.loc[row_ix[0][0], 'tweet_text'],
            mode = "markers",
            hovertemplate = "Latitude: %{lat: .3f}<br>Longitude: %{lon: .3f}<br>Size: " + str(cluster[1]) + "<br>" + text_format(result_df.loc[row_ix[0][0], "tweet_text"], 100),
            name = "Cluster " + str(i + 1),
            marker = go.scattermapbox.Marker(
                size = get_size(cluster[1]),
                opacity = 0.8,
                color = get_color(cluster[1])
            )
        ))
    fig.update_layout(
        title = "Timeframe: " + str(start_ts) + " to " + str(end_ts),
        hovermode = 'closest',
        showlegend = False,
        width = width,
        height = height,
        mapbox = dict(
            accesstoken = mapbox_access_token,
            center = dict(
                lat = get_center(result_df['latitude']),
                lon = get_center(result_df['longitude'])
            ),
            zoom = get_zoom(result_df['latitude'], result_df['longitude'], width/height)
        )
    )
    return fig
    

def app(df, filtered_df, es, index_name):
    st.title('Keyword Search')
    search = st.sidebar.text_input("Enter Keyword Search:", help="Enter the word/phrase that you want to query the set of tweets with")
    k = st.sidebar.slider("Select a k value:", 1, 10, 3, help="k value will be used to return the top-k locations pertaining to the keyword search")
    num_intervals = st.sidebar.slider("Select number of intervals:", 1, 10, 3, help="Split the total timeframe into XX intervals of equal length")
    individual_pts = st.sidebar.checkbox("View Individual Points", help="View each tweet individually on the map instead of clusters")
    if not individual_pts:
        compare = st.sidebar.checkbox("Compare with Previous Interval", help="Display clusters of previous intervals for comparison")

    if search == "":
        st.markdown("Please make a **spatial keyword search** at the `sidebar` :sunglasses:")
        return

    query_df = make_query(es, index_name, search, None, None)
    if len(query_df) == 0:
        st.write("Sorry, no tweets found! :persevere: :heavy_multiplication_x:")
        return
    start = query_df['datetime'].min()
    end = query_df['datetime'].max()
    st.write("Start:", start)
    st.write("End:", end)
    results = []
    cluster_list = []
    label_list = []
    timestamps = []
    figs = []
    mapbox_access_token = "pk.eyJ1IjoiYW1hZGV1c2t5aiIsImEiOiJja290bTdtMmUwOXh5Mm9xbXc0dzBpbGR4In0.ZQYWX00qp75XObfeRbk5gg"

    #Select timeframe
    left_column, right_column = st.beta_columns(2)
    start_date = left_column.date_input("Start Date of Timeframe", start, start, end)
    start_time = left_column.time_input('Start Time of Timeframe', start)
    end_date = right_column.date_input("End Date of Timeframe", end, start, end)
    end_time = right_column.time_input('End Time of Timeframe', end)

    start_datetime = datetime.combine(start_date, start_time)
    end_datetime = datetime.combine(end_date, end_time)
    time_diff = end_datetime-start_datetime
    timestamps.append(start_datetime)

    for i in range(num_intervals):
        #in case adding timedelta with nanoseconds, we want to remove nanoseconds
        timestamps.append(string_to_date(str(timestamps[-1] + time_diff/num_intervals).split('.')[0]))
    
    for i in range(len(timestamps)-1):
        results.append(make_query(es, index_name, search, date_to_string(timestamps[i]), date_to_string(timestamps[i+1])))
        labels = run_clustering(results[i])
        clusters = list(unique(labels))
        #Remove the outlier cluster
        clusters.remove(-1)
        cluster_sizes = [len(where(labels==cluster)[0]) for cluster in clusters]
        sorted_clusters = [(int(cluster), int(size)) for size, cluster in sorted(zip(cluster_sizes, clusters))]
        sorted_clusters.reverse()
        #extract only top k clusters
        clusters = np.array(sorted_clusters[:k])
        cluster_list.append(clusters)
        label_list.append(labels)
        if len(clusters) == 0:
            figs.append(None)
        elif individual_pts:
            fig = individual_plot(mapbox_access_token, clusters, labels, results[i], timestamps[i], timestamps[i+1])
            figs.append(fig)
        else:
            if compare:
                if i == 0:
                    fig = cluster_plot(mapbox_access_token, cluster_list[i], label_list[i], results[i], timestamps[i], timestamps[i+1], [], [], [])
                else:
                    fig = cluster_plot(mapbox_access_token, cluster_list[i], label_list[i], results[i], timestamps[i], timestamps[i+1], cluster_list[i-1], label_list[i-1], results[i-1])
            else:
                fig = cluster_plot(mapbox_access_token, cluster_list[i], label_list[i], results[i], timestamps[i], timestamps[i+1], [], [], [])
            figs.append(fig)


    #Select interval
    if (num_intervals != 1):
        interval_num = st.slider("Select interval:", 1, num_intervals, 1)
        st.write("Timeframe:", timestamps[interval_num-1], "to", timestamps[interval_num])
        fig_to_display = figs[interval_num-1]
        if fig_to_display == None:
            st.write("Sorry, there are no clusters here! :pensive: :heavy_multiplication_x:")
        else:
            st.plotly_chart(fig_to_display)
    else:
        st.write("Timeframe:", timestamps[0], "to", timestamps[1])
        st.plotly_chart(figs[0])
    
