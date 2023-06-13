import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport
import streamlit as st
from streamlit_pandas_profiling import st_profile_report
import streamlit.components.v1 as components
import numpy as np
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.imputation import mice

# Function to replace missing values
@st.cache_data
def replace_missing_values(df, method):
    # Differentiate numerical and categorical columns
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if method == 'drop':
        df = df.dropna()
    elif method == 'zero':
        df[num_cols] = df[num_cols].fillna(0)
    elif method == 'mean':
        df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    elif method == 'median':
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    elif method == 'mode':
        df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])
    elif method == 'mice':
        imp = mice.MICEData(df[num_cols])  # only apply to numerical columns
        df[num_cols] = imp.data
    return df

@st.cache_data  # This function will be cached
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def analyze_dataframe(df):
    # Analyzing missing values
    missing_values = df.isnull().sum()

    # Analyzing outliers using the Z-score
    # (you might want to use a different method for identifying outliers)
    z_scores = np.abs((df - df.mean()) / df.std())
    outliers = (z_scores > 3).sum()

    # Analyzing data types
    data_types = df.dtypes

    # Analyzing skewness for numeric columns
    skewness = df.select_dtypes(include=[np.number]).apply(lambda x: x.skew())

    # Analyzing cardinality in categorical columns
    cardinality = df.select_dtypes(include=['object', 'category']).nunique()

    return missing_values, outliers, data_types, skewness, cardinality


st.title("Autoanalyzer")
with st.expander('About Autoanalyzer'):
    st.write("Author: David Liebovitz, MD, Northwestern University")
    st.write("Last updated 6/11/23")
st.info("""Be sure your data is first in a 'tidy' format. Use the demo dataset below to check out the top 5 rows for an example. (*See https://tidyr.tidyverse.org/ for more information.*)
Be sure to check out the **automated analysis** option for a full report on your data.
Additional information on the demo dataset: https://hbiostat.org/data/repo/diabetes.html""")

st.sidebar.subheader("Upload your data") 
st.subheader("Step 1: Upload your data or choose our demo dataset")
demo_or_custom = st.selectbox("Choose a demo or upload your CSV file. NO PHI - use only anonymized data", ("Select here!", "Demo", "CSV Upload"))
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if demo_or_custom == 'Demo':
    file_path = "data/predictdm.csv"
    df = load_data(file_path)

else:
    if uploaded_file:            
        df = load_data(uploaded_file)
    # num_df = process_dataframe(df)



with st.expander("Data Preprocessing Tools - *Use analysis tools **first** to check if needed.*"):
    preprocess = st.checkbox("Assign bivariate categories into 1 or 0 based on frequency (0 most frequent) if needed for correlations, e.g.", key = "Preprocess")
    st.info("Select a method to impute missing values in your dataset. Built in checks to apply only to applicable data types.")
    method = st.selectbox("Choose a method to replace missing values", ("Select here!", "drop", "zero", "mean", "median", "mode", "mice"))
    if st.button('Apply the Method to Replace Missing Values'):
            df = replace_missing_values(df, method)
st.subheader("Step 2: Tools for Analysis")
col1, col2 = st.columns(2)
with col1:
    check_preprocess = st.checkbox("Check if you need to preprocess data", key = "Preprocess needed")
    header = st.checkbox("Show header (top 5 rows of data)", key = "show header")
    summary = st.checkbox("Show summary of numerical data", key = "show data")
    summary_cat = st.checkbox("Show summary (categorical data)", key = "show summary cat")
with col2:
    barchart = st.checkbox("Show bar chart (categorical data)", key = "show barchart")
    histogram = st.checkbox("Show histogram (numerical data)", key = "show histogram")
    piechart = st.checkbox("Show pie chart (categorical data)", key = "show piechart")
    show_corr = st.checkbox("Show correlation heatmap", key = "show corr")
full_analysis = st.checkbox("**Automated Analysis** (*Check **Alerts** with key findings.*)", key = "show analysis")



# Function to plot pie chart
def plot_pie(df, col_name):
    plt.figure(figsize=(10, 8))  # set the size of the plot
    df[col_name].value_counts().plot(kind='pie', autopct='%1.1f%%')

    # Add title
    plt.title(f'Distribution for {col_name}')

    return plt


# Function to summarize categorical data
def summarize_categorical(df):
    # Select only categorical columns
    cat_df = df.select_dtypes(include=['object', 'category'])

    # If there are no categorical columns, return None
    if cat_df.empty:
        st.write("The DataFrame does not contain any categorical columns.")
        return None

    # Summarize categorical data
    summary = pd.DataFrame()

    for col in cat_df.columns:
        # Number of unique values
        unique_count = df[col].nunique()

        # Most frequent category and its frequency
        most_frequent = df[col].mode()[0]
        freq_most_frequent = df[col].value_counts().iloc[0]

        summary = summary.append(pd.Series({
            'column': col,
            'unique_count': unique_count,
            'most_frequent': most_frequent,
            'frequency_most_frequent': freq_most_frequent,
        }), ignore_index=True)

    summary.set_index('column', inplace=True)

    return summary

# Function to plot correlation heatmap
def plot_corr(df):
    corr = df.corr()  # Compute pairwise correlation of columns
    plt.figure(figsize=(12, 10))  # set the size of the plot
    sns.heatmap(corr, annot=True, cmap='coolwarm', cbar=True)
    plt.title('Correlation Heatmap')
    return plt

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
@st.cache_data
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
                df[col] = df[col].replace({most_frequent: 0, least_frequent: 1}).astype(int)
                
    return df



if demo_or_custom == 'Demo' or uploaded_file:
    
    if preprocess:
        df = process_dataframe(df)
    
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

    if show_corr:
        st.info("Correlation heatmap")
        plt = plot_corr(df)
        st.pyplot(plt)

    if summary_cat:
        st.info("Summary of categorical data")
        summary = summarize_categorical(df)
        st.write(summary)
        
    if piechart:
        st.info("Pie chart for categorical data")
        cat_options =[]
        columns = list(df.columns)
        for col in columns:
            if df[col].dtype != np.float64 and df[col].dtype != np.int64:
                cat_options.append(col)
        cat_selected_col = st.selectbox("Choose a column", cat_options)
        if cat_selected_col:
            plt = plot_pie(df, cat_selected_col)
            st.pyplot(plt)
            
    if check_preprocess:
        st.info("Check if you need to preprocess data")
        missing_values, outliers, data_types, skewness, cardinality = analyze_dataframe(df)
        st.write("Missing values")
        st.write(missing_values)
        st.write("Outliers")
        st.write(outliers)
        st.write("Data types")
        st.write(data_types)
        st.write("Skewness")
        st.write(skewness)
        st.write("Cardinality")
        st.write(cardinality)
        
    
# if demo_or_custom == 'CSV Upload':
#     if uploaded_file:            
#         df = load_data(uploaded_file)
#         # df
#         # profile = make_profile(df)
#         # # profile = ProfileReport(df, title="Profiling Report")
#         # st_profile_report(profile)

        
#         if preprocess:
#             df = process_dataframe(df)

        
#         if summary:
#             st.info("Summary of data")
#             st.write(df.describe())
#         if header:
#             st.info("Header of data")
#             st.write(df.head())
#         if full_analysis:
#             st.info("Full analysis of data")
#             profile = make_profile(df)
#             # profile = ProfileReport(df, title="Profiling Report")
#             st_profile_report(profile)
#         if histogram: 
#             st.info("Histogram of data")
#             options =[]
#             columns = list(df.columns)
#             for col in columns:
#                 if df[col].dtype == np.float64 or df[col].dtype == np.int64:
#                     options.append(col)
#             selected_col = st.selectbox("Choose a column", options)
#             if selected_col:
#                 plt = plot_numeric(df, selected_col)
#                 st.pyplot(plt)
            
#             # hist_data = [df[selected_col]]
#             # group_labels = [selected_col]
#             # fig = ff.create_distplot(hist_data, group_labels)
#             # st.plotly_chart(fig, use_container_width=True)
        
#         if barchart: 
#             st.info("Barchart for categorical data")
#             cat_options =[]
#             columns = list(df.columns)
#             for col in columns:
#                 if df[col].dtype != np.float64 and df[col].dtype != np.int64:
#                     cat_options.append(col)
#             cat_selected_col = st.selectbox("Choose a column", cat_options)
#             if cat_selected_col:
#                 plt = plot_categorical(df, cat_selected_col)
#                 st.pyplot(plt)



