import streamlit as st

def app(df, filtered_df, es, index_name):
    st.title('Home Page')

    st.write('This is the `home page` of this application.')
    st.write('Use the sidebar in order to navigate to other pages')
    st.write('Please follow the order of instructions below:')
    st.write('1. Please select a file for a dataset (Tweets), else the default (Taal Volcano dataset) will be used')
    st.write('2. Look at the contents of the dataset at the `Data` page')
    st.write('3. Explore the dataset through visualisation at the `Visualisation` page')
    st.write('4. Analyse the spread of an event at the `Keyword Search` page')
