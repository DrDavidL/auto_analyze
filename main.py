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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    return fig

def plot_roc_curve(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label='ROC curve')
    ax.plot([0, 1], [0, 1], 'k--', label='Random guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.xlim([-0.02, 1])
    plt.ylim([0, 1.02])
    plt.legend(loc="lower right")
    return fig

def preprocess(df, target_col):
    included_cols = []
    excluded_cols = []

    for col in df.columns:
        if col != target_col:  # Exclude target column from preprocessing
            if df[col].dtype == 'object':
                if len(df[col].unique()) == 2:  # Bivariate case
                    most_freq = df[col].value_counts().idxmax()
                    least_freq = df[col].value_counts().idxmin()
                    df[col] = df[col].map({most_freq: 0, least_freq: 1})
                    included_cols.append(col)
                else:  # Multivariate case
                    excluded_cols.append(col)
            elif df[col].dtype in ['int64', 'float64']:  # Numerical case
                if df[col].isnull().values.any():
                    mean_imputer = SimpleImputer(strategy='mean')
                    df[col] = mean_imputer.fit_transform(df[[col]])
                    print(f"Imputed missing values in {col} with mean.")
                included_cols.append(col)

    print(f"Included Columns: {included_cols}")
    print(f"Excluded Columns: {excluded_cols}")
    
    return df[included_cols], included_cols, excluded_cols


def create_violinplot(df, numeric_col, categorical_col):
    if numeric_col and categorical_col:
        fig, ax = plt.subplots()

        # Plot the violin plot
        sns.violinplot(x=categorical_col, y=numeric_col, data=df, ax=ax)

        st.pyplot(fig)


def create_scatterplot(df, scatter_x, scatter_y):
    if scatter_x and scatter_y:
        fig, ax = plt.subplots()

        # Plot the scatter plot
        sns.regplot(x=scatter_x, y=scatter_y, data=df, ax=ax)

        # Calculate the slope and intercept of the regression line
        slope, intercept = np.polyfit(df[scatter_x], df[scatter_y], 1)

        # Add the slope and intercept as a text annotation on the plot
        ax.text(0.05, 0.95, f'y={slope:.2f}x+{intercept:.2f}', transform=ax.transAxes)

        st.pyplot(fig)


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

st.title("Autoanalyzer")
with st.expander('About Autoanalyzer'):
    st.write("Author: David Liebovitz, MD, Northwestern University")
    st.write("Last updated 6/11/23")
    
tab1, tab2 = st.tabs(["Data Exploration", "Machine Learning"])

with tab1:

    st.info("""Be sure your data is first in a 'tidy' format. Use the demo dataset below to check out the top 5 rows for an example. (*See https://tidyr.tidyverse.org/ for more information.*)
    Be sure to check out the **automated analysis** option for a full report on your data.
    Additional information on the demo dataset: https://hbiostat.org/data/repo/diabetes.html""")

    # st.sidebar.subheader("Upload your data") 
    st.subheader("Step 1: Upload your data or choose our demo dataset")
    demo_or_custom = st.selectbox("Choose a demo or upload your CSV file. NO PHI - use only anonymized data", ("Select here!", "Demo", "CSV Upload"))
    if demo_or_custom == "CSV Upload":
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file:
            df = load_data(uploaded_file)

    if demo_or_custom == 'Demo':
        file_path = "data/predictdm.csv"
        df = load_data(file_path)



    with st.expander("Data Preprocessing Tools - *Use analysis tools **first** to check if needed.*"):
        pre_process = st.checkbox("Assign bivariate categories into 1 or 0 based on frequency (0 most frequent) if needed for correlations, e.g.", key = "Preprocess")
        st.info("Select a method to impute missing values in your dataset. Built in checks to apply only to applicable data types.")
        method = st.selectbox("Choose a method to replace missing values", ("Select here!", "drop", "zero", "mean", "median", "mode", "mice"))
        if st.button('Apply the Method to Replace Missing Values'):
                df = replace_missing_values(df, method)
    st.subheader("Step 2: Tools for Analysis")
    col1, col2 = st.columns(2)
    with col1:
        check_preprocess = st.checkbox("Check if you need to preprocess data", key = "Preprocess needed")
        header = st.checkbox("Show header (top 5 rows of data)", key = "show header")
        summary = st.checkbox("Show summary for numerical data", key = "show data")
        summary_cat = st.checkbox("Show summary for categorical data", key = "show summary cat")
        show_scatter  = st.checkbox("Show scatterplot", key = "show scatter")
    with col2:
        barchart = st.checkbox("Show bar chart (categorical data)", key = "show barchart")
        histogram = st.checkbox("Show histogram (numerical data)", key = "show histogram")
        piechart = st.checkbox("Show pie chart (categorical data)", key = "show piechart")
        show_corr = st.checkbox("Show correlation heatmap", key = "show corr")
        violin_plot = st.checkbox("Show violin plot", key = "show violin")
    full_analysis = st.checkbox("*(Takes 1-2 minutes*) **Automated Analysis** (*Check **Alerts** with key findings.*)", key = "show analysis")
    view_full_df = st.checkbox("The CSV file", key = "view full df")


    try:
        x = df
    except NameError:

        st.warning("Please upload a CSV file or choose a demo dataset")
    else:
        
        if pre_process:
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
            
        if show_scatter:
            st.info("Scatterplot")

            # Filter numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols.sort()  # sort the list of columns alphabetically
            
                # Filter categorical columns
            categorical_cols = df.select_dtypes(include=[object]).columns.tolist()
            categorical_cols.sort()  # sort the list of columns alphabetically
            # Dropdown to select columns to visualize
            col1, col2 = st.columns(2)
            with col1:
                scatter_x = st.selectbox('Select column for x axis:', numeric_cols)
            with col2:
                scatter_y = st.selectbox('Select column for y axis:', numeric_cols)
                
            # Use st.beta_expander to hide or expand filtering options
            with st.expander('Filter Options'):
                # Filter for the remaining numerical column
                remaining_cols = [col for col in numeric_cols if col != scatter_x and col != scatter_y]
                if remaining_cols:
                    filter_col = st.selectbox('Select a numerical column to filter data:', remaining_cols)
                    if filter_col:
                        min_val, max_val = float(df[filter_col].min()), float(df[filter_col].max())
                        if np.isnan(min_val) or np.isnan(max_val):
                            st.write(f"Cannot filter by {filter_col} because it contains NaN values.")
                        else:
                            filter_range = st.slider('Select a range to filter data:', min_val, max_val, (min_val, max_val))
                            df = df[(df[filter_col] >= filter_range[0]) & (df[filter_col] <= filter_range[1])]

                # Filter for the remaining categorical column
                if categorical_cols:
                    filter_cat_col = st.selectbox('Select a categorical column to filter data:', categorical_cols)
                    if filter_cat_col:
                        categories = df[filter_cat_col].unique().tolist()
                        selected_categories = st.multiselect('Select categories to include in the data:', categories, default=categories)
                        df = df[df[filter_cat_col].isin(selected_categories)]
            # Check if DataFrame is empty before creating scatterplot
            if df.empty:
                st.write("The current filter settings result in an empty dataset. Please adjust the filter settings.")
            else:
                    create_scatterplot(df, scatter_x, scatter_y)
                    
        if violin_plot:
            # Filter numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols.sort()  # sort the list of columns

            # Filter categorical columns
            categorical_cols = df.select_dtypes(include=[object]).columns.tolist()
            categorical_cols.sort()  # sort the list of columns

            # Dropdown to select columns to visualize
            numeric_col = st.selectbox('Select a numerical column:', numeric_cols)
            categorical_col = st.selectbox('Select a categorical column:', categorical_cols)

            create_violinplot(df, numeric_col, categorical_col)
            
        if view_full_df:
            st.write(df)

with tab2:
    st.info("""N.B. This merely shows a glimpse of what is possible. Any model shown is not yet optimized and requires ML and domain level expertise.
            Yet, this is a good start to get a sense of what is possible."""
            )
    try:
        x = df
    except NameError:
        st.warning("First upload a CSV file or choose a demo dataset from the **Data Exploration** tab")
    else:

        # Filter categorical columns and numerical bivariate columns
        categorical_cols = df.select_dtypes(include=[object]).columns.tolist()

        # Add bivariate numerical columns
        numerical_bivariate_cols = [col for col in df.select_dtypes(include=['int64', 'float64']).columns 
                                    if df[col].nunique() == 2]

        # Combine the two lists and sort them
        categorical_cols = categorical_cols + numerical_bivariate_cols
        categorical_cols.sort()  # sort the list of columns


        st.write("""
        # Choose the Target Column
        """)
        target_col = st.selectbox('Select a categorical column as the target:', categorical_cols)

        st.write("""
        # Choose the Value to Predict
        """)
        categories_to_predict = st.multiselect('Select one or more categories:', df[target_col].unique().tolist())

        # Preprocess the data and exclude the target column from preprocessing
        df_processed, included_cols, excluded_cols = preprocess(df.drop(columns=[target_col]), target_col)
        df_processed[target_col] = df[target_col]  # Include the target column back into the dataframe

        st.write(f"Included columns: {included_cols}")
        st.write(f"Excluded columns: {excluded_cols}")

        # Create binary target variable based on the selected categories
        df_processed[target_col] = df_processed[target_col].apply(lambda x: 1 if x in categories_to_predict else 0)


        # Split the dataframe into data and labels
        X = df_processed.drop(columns=[target_col])
        y = df_processed[target_col]

        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        st.write("""
        # Choose the Machine Learning Model
        """)
        model_option = st.selectbox(
            "Which machine learning model would you like to use?",
            ("Logistic Regression", "Decision Tree", "Random Forest")
        )

        if st.button("Predict"):
            if model_option == "Logistic Regression":
                model = LogisticRegression()
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                accuracy = accuracy_score(y_test, predictions)
                st.write(f"Accuracy: {accuracy}")
                st.write(plot_confusion_matrix(y_test, predictions))
                y_scores = model.predict_proba(X_test)[:, 1]
                st.write(plot_roc_curve(y_test, y_scores))
                # After training the logistic regression model, assuming the model's name is "model"

                coeff = model.coef_[0]
                features = X_train.columns

                equation = "Logit(P) = " + str(model.intercept_[0])

                for c, feature in zip(coeff, features):
                    equation += " + " + str(c) + " * " + feature

                st.write("The equation of the logistic regression model is:")
                st.write(equation)


            elif model_option == "Decision Tree":
                model = DecisionTreeClassifier()
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                accuracy = accuracy_score(y_test, predictions)
                st.write(f"Accuracy: {accuracy}")
                st.write(plot_confusion_matrix(y_test, predictions))
                y_scores = model.predict_proba(X_test)[:, 1]
                st.write(plot_roc_curve(y_test, y_scores))

            elif model_option == "Random Forest":
                model = RandomForestClassifier()
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                accuracy = accuracy_score(y_test, predictions)
                st.write(f"Accuracy: {accuracy}")
                st.write(plot_confusion_matrix(y_test, predictions))
                y_scores = model.predict_proba(X_test)[:, 1]
                st.write(plot_roc_curve(y_test, y_scores))

                
        