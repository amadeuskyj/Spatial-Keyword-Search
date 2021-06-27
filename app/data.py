import streamlit as st

def app(df, filtered_df, es, index_name):
    st.title('Data Page')
    st.write('View the `raw data` and the `data after cleaning` below')

    st.write("Raw Data:")
    st.write(df)
    st.write("Filtered Data:")
    st.write(filtered_df)