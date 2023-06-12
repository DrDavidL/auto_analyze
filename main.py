import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport
import streamlit as st
from streamlit_pandas_profiling import st_profile_report
import streamlit.components.v1 as components
import numpy as np
import plotly.figure_factory as ff
import matplotlib.pyplot as plt

st.title("Autoanalyzer")
with st.expander('About Autoanalyzer'):
    st.write("Author: David Liebovitz, MD, Northwestern University")
    st.write("Last updated 6/11/23")
st.info("""Be sure your data is first in a 'tidy' format. Use the demo dataset below to check out the top 5 rows for an example. (*See https://tidyr.tidyverse.org/ for more information.*)
Be sure to check out the **automated analysis** option for a full report on your data.
Additional information on the demo dataset: https://hbiostat.org/data/repo/diabetes.html""")



demo_or_custom = st.selectbox("Choose a demo or upload your CSV file. NO PHI - use only anonymized data", ("Demo", "CSV Upload"))

@st.cache_resource
def make_profile(df):
    return ProfileReport(df, title="Profiling Report")
    
# Function to plot bar chart
def plot_categorical(df, col_name):
    # Get frequency of categories
    freq = df[col_name].value_counts()

    # Create bar chart
    plt.figure(figsize=(10, 6))  # set the size of the plot
    plt.bar(freq.index, freq.values)

    # Add title and labels
    plt.title(f'Frequency of Categories fpipenv --python 3.9or {col_name}')
    plt.xlabel('Category')
    plt.ylabel('Frequency')

    return plt

def plot_numeric(df, col_name):
    plt.figure(figsize=(10, 6))  # set the size of the plot
    plt.hist(df[col_name], bins=30, alpha=0.5, color='blue', edgecolor='black')

    # Add title and labels
    plt.title(f'Distribution for {col_name}')
    plt.xlabel(col_name)
    plt.ylabel('Frequency')

    return plt

def process_dataframe(df):
    # Iterating over each column
    for col in df.columns:
        # Checking if the column is of object type (categorical)
        if df[col].dtype == 'object':
            # Getting unique values in the column
            unique_values = df[col].unique()

            # If the column has exactly 2 unique values
            if len(unique_values) == 2:
                # Counting the occurrences of each value
                value_counts = df[col].value_counts()

                # Getting the most and least frequent values
                most_frequent = value_counts.idxmax()
                least_frequent = value_counts.idxmin()

                # Replacing the values and converting to integer
                df[col] = df[col].replace({most_frequent: 1, least_frequent: 0}).astype(int)
                
    return df


if demo_or_custom == 'Demo':
    col1, col2 = st.columns(2)
    df = pd.read_csv("data/predictdm.csv")
    # num_df = process_dataframe(df)
    with col1:
        header = st.checkbox("Show header (top 5 rows of data)", key = "show header")
        summary = st.checkbox("Show summary of numerical data", key = "show data")
        barchart = st.checkbox("Show bar chart (categorical data)", key = "show barchart")
        # summary_cat = st.checkbox("Show summary (categorical data)", key = "show summary cat")
    with col2:
        
        histogram = st.checkbox("Show histogram (numerical data)", key = "show histogram")
        full_analysis = st.checkbox("Show automated analysis", key = "show analysis")
    
    
    if summary:
        st.info("Summary of data")
        st.write(df.describe())
    if header:
        st.info("Header of data")
        st.write(df.head())
    if full_analysis:
        st.info("Full analysis of data")
        profile = make_profile(df)
        # profile = ProfileReport(df, title="Profiling Report")
        st_profile_report(profile)
    if histogram: 
        st.info("Histogram of data")
        options =[]
        columns = list(df.columns)
        for col in columns:
            if df[col].dtype == np.float64 or df[col].dtype == np.int64:
                options.append(col)
        selected_col = st.selectbox("Choose a column", options)
        if selected_col:
            plt = plot_numeric(df, selected_col)
            st.pyplot(plt)
        # hist_data = [df[selected_col]]
        # group_labels = [selected_col]
        # fig = ff.create_distplot(hist_data, group_labels)
        # st.plotly_chart(fig, use_container_width=True)
    
    if barchart: 
        st.info("Barchart for categorical data")
        cat_options =[]
        columns = list(df.columns)
        for col in columns:
            if df[col].dtype != np.float64 and df[col].dtype != np.int64:
                cat_options.append(col)
        cat_selected_col = st.selectbox("Choose a column", cat_options)
        if cat_selected_col:
            plt = plot_categorical(df, cat_selected_col)
            st.pyplot(plt)

st.sidebar.subheader("Upload your data") 
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    
if demo_or_custom == 'CSV Upload':
    if uploaded_file:            
        df = pd.read_csv(uploaded_file)
        # df
        # profile = make_profile(df)
        # # profile = ProfileReport(df, title="Profiling Report")
        # st_profile_report(profile)
        col1, col2 = st.columns(2)
        preprocess = st.checkbox("Preprocess data - optional - convert bivariate variables into 1 or 0 based on frequency.", key = "Preprocess")
        if preprocess:
            df = process_dataframe(df)
        with col1:
            header = st.checkbox("Show header (top 5 rows of data)", key = "show header")
            summary = st.checkbox("Show summary of numerical data", key = "show data")
            barchart = st.checkbox("Show bar chart (categorical data)", key = "show barchart")
            # summary_cat = st.checkbox("Show summary (categorical data)", key = "show summary cat")
        with col2:
            
            histogram = st.checkbox("Show histogram (numerical data)", key = "show histogram")
            full_analysis = st.checkbox("Show automated analysis", key = "show analysis")
        
        
        if summary:
            st.info("Summary of data")
            st.write(df.describe())
        if header:
            st.info("Header of data")
            st.write(df.head())
        if full_analysis:
            st.info("Full analysis of data")
            profile = make_profile(df)
            # profile = ProfileReport(df, title="Profiling Report")
            st_profile_report(profile)
        if histogram: 
            st.info("Histogram of data")
            options =[]
            columns = list(df.columns)
            for col in columns:
                if df[col].dtype == np.float64 or df[col].dtype == np.int64:
                    options.append(col)
            selected_col = st.selectbox("Choose a column", options)
            if selected_col:
                plt = plot_numeric(df, selected_col)
                st.pyplot(plt)
            
            # hist_data = [df[selected_col]]
            # group_labels = [selected_col]
            # fig = ff.create_distplot(hist_data, group_labels)
            # st.plotly_chart(fig, use_container_width=True)
        
        if barchart: 
            st.info("Barchart for categorical data")
            cat_options =[]
            columns = list(df.columns)
            for col in columns:
                if df[col].dtype != np.float64 and df[col].dtype != np.int64:
                    cat_options.append(col)
            cat_selected_col = st.selectbox("Choose a column", cat_options)
            if cat_selected_col:
                plt = plot_categorical(df, cat_selected_col)
                st.pyplot(plt)



