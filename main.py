import numpy as np
# import langchain
import pandas as pd
import io
import sys
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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, average_precision_score, precision_recall_curve, auc, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn import svm
from langchain.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain.llms import OpenAI
import json
import plotly.io as pio
from bs4 import BeautifulSoup
from PIL import Image





# import asyncio
# import bardapi
# from bardapi import Bard
import openai
from tableone import TableOne
from scipy import stats
from streamlit_chat import message
import random
from random import randint
import os



st.set_page_config(page_title='AutoAnalyzer', layout = 'centered', page_icon = ':chart_with_upwards_trend:', initial_sidebar_state = 'auto')
# if st.button('Click to toggle sidebar state'):
#     st.session_state.sidebar_state = 'collapsed' if st.session_state.sidebar_state == 'expanded' else 'expanded'
#     # Force an app rerun after switching the sidebar state.
#     st.experimental_rerun()
    
#     # Initialize a session state variable that tracks the sidebar state (either 'expanded' or 'collapsed').
if 'last_response' not in st.session_state:
     st.session_state.last_response = ''
     
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame()
    
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ''
    
if "gen_csv" not in st.session_state:
    st.session_state.gen_csv = None

# Streamlit set_page_config method has a 'initial_sidebar_state' argument that controls sidebar state.
# st.set_page_config(initial_sidebar_state=st.session_state.sidebar_state)
# Prepare Chatbot Helper

# async def start_chatbot():    
# def start_chatbot():   
#     with st.sidebar:
#         st.write("Chatbot Teacher")
#         if 'sidebar_state' not in st.session_state:
#             st.session_state.sidebar_state = 'expanded'
#         token = st.secrets["BARD_TOKEN"]
#         # Bard = Bard(token, timeout=10)
#         helper_prefix = """You are a friendly teacher to medical students learning about data science. You answer all questions 
#         through this lens and defer questions that are completely unrelated to this topic. Your responses cannot contain images or links.
#         You are posted on a website next to an interactive tool that has a preloaded demo set of data. You can refer to the tool to ask students to make a checkbox selectio to view bar charts, 
#         histograms, pie charts, violin plots, scatterplots, and summary statistics for the sample dataset. They also have an option to upload their own CSV file, although most probably won't do this.        
#         Student question:        
#         """
#         # st.sidebar.info("Chatbot:", value="Hi! I'm your friendly chatbot. Ask me anything about data science and I'll try to answer it.", height=200, max_chars=None)
#         question_input = st.sidebar.text_input("Your question, e.g., 'teach me about violin plots'", "")
#         if st.button("Send"):
#             if question_input:
                
#                 response = bardapi.core.Bard(token, timeout = 200).get_answer(helper_prefix + question_input)['content']
#                 st.session_state.last_response = response

def is_valid_api_key(api_key):
    openai.api_key = api_key

    try:
        # Send a test request to the OpenAI API
        response = openai.Completion.create(model="text-davinci-003",                     
                    prompt="Hello world")['choices'][0]['text']
        return True
    except Exception:
        pass

    return False

def fetch_api_key():
    api_key = None
    
    try:
        # Attempt to retrieve the API key as a secret
        api_key = st.secrets["OPENAI_API_KEY"]
        # os.environ["OPENAI_API_KEY"] = api_key
        st.session_state.openai_api_key = api_key
        os.environ['OPENAI_API_KEY'] = api_key
        # st.write(f'Here is what we think the key is step 1: {api_key}')
    except KeyError:
        
        if st.session_state.openai_api_key != '':
            api_key = st.session_state.openai_api_key
            os.environ['OPENAI_API_KEY'] = api_key
            # If the API key is already set, don't prompt for it again
            # st.write(f'Here is what we think the key is step 2: {api_key}')
            return 
        else:        
            # If the secret is not found, prompt the user for their API key
            st.warning("Oh, dear friend of mine! It seems your API key has gone astray, hiding in the shadows. Pray, reveal it to me!")
            api_key = st.text_input("Please, whisper your API key into my ears: ", key = 'warning')
  
            st.session_state.openai_api_key = api_key
            os.environ['OPENAI_API_KEY'] = api_key
            # Save the API key as a secret
            # st.secrets["my_api_key"] = api_key
            # st.write(f'Here is what we think the key is step 3: {api_key}')
            return 
    
    return 

def check_password():

    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == os.getenv("password"):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.write("*Please contact David Liebovitz, MD if you need an updated password for access.*")
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        # fetch_api_key()
        return True

csv_prefix ="""You are an agent optimally designed for answering questions about a dataframe. 
If anwering a query requires drawing a table, chart, or generating any other figure, never attempt 
to draw the figure. Instead, return the Python code as a string. Do not return JSON. The following are already imported so 
do not include any import statements in your code:
- plotly.figure_factory as ff
- matplotlib.pyplot as plt
- seaborn as sns

Please format the string response (not JSON) such that it includes:

1. Code to interpret the user's question and select the appropriate visualization.
2. Code to generate the visualization using plotly as ff, matplotlib.pyplot as plt, or seaborn as sns. 
3. Do not return JSON. The response should include the Python code as a string.

Remember to structure the code such that it is properly indented and formatted according to PEP8 guidelines.

            """
            
csv_prefix_gpt4 ="""You are an agent optimally designed for generating compelling plots about a dataframe. 
Never attempt to draw the figure directly. Instead, return the Python code as a string. Do not return JSON. 
The output from your code must be saved to a file as a single png file, 'output.png'. Here is an example:

```
import seaborn as sns

sns.scatterplot(x='Age', y='BMI', data=df)
plt.savefig('./images/output.png')
```

Do not generate code as follows. Remember, all outputs musbe saved to a file as a single png file, './images/output.png':

```
df_grouped = df.groupby('Diabetes').mean()
print(df_grouped)
```

Instead, code like this should be used:
```
df_grouped = df.groupby('Diabetes').mean()

# Plotting
df_grouped.plot(kind='bar')

# Saving the plot to a .png file
plt.savefig('./images/output.png')
```

Pre-process data when needed to eliminate all code execution errors. For example, prevent errors related to us of    text values when generating a heatmap. 
Text values produce an error if generating a heatmap. The following heatmap error can be prevented if binary categories are first converted
to numerical values, e.g., 1 and 0 and columns with multiple text categories are dropped from the heatmap.

```
ValueError: could not convert string to float: 'female'
```

Please format the string response (not JSON) such that it includes:

1. Code to interpret the user's question to generate and save the appropriate visualization.
2. Do not return JSON. The response should include the Python code as a string.

Remember to structure the code such that it is properly indented and formatted according to PEP8 guidelines.
"""

def assess_data_readiness(df):
    readiness_summary = {}

    # Check if the DataFrame is empty
    if df.empty:
        readiness_summary['data_empty'] = True
        readiness_summary['columns'] = {}
        readiness_summary['missing_columns'] = []
        readiness_summary['inconsistent_data_types'] = []
        readiness_summary['missing_values'] = {}
        readiness_summary['data_ready'] = False
        return readiness_summary

    # Get column information
    columns = {col: str(df[col].dtype) for col in df.columns}
    readiness_summary['columns'] = columns

    # Check for missing columns
    missing_columns = df.columns[df.isnull().all()].tolist()
    readiness_summary['missing_columns'] = missing_columns

    # Check for inconsistent data types
    inconsistent_data_types = []
    for col in df.columns:
        unique_data_types = df[col].apply(type).drop_duplicates().tolist()
        if len(unique_data_types) > 1:
            inconsistent_data_types.append(col)
    readiness_summary['inconsistent_data_types'] = inconsistent_data_types

    # Check for missing values
    missing_values = df.isnull().sum().to_dict()
    readiness_summary['missing_values'] = missing_values

    # Determine overall data readiness
    readiness_summary['data_empty'] = False
    if missing_columns or inconsistent_data_types or any(missing_values.values()):
        readiness_summary['data_ready'] = False
    else:
        readiness_summary['data_ready'] = True

    return readiness_summary


def process_model_output(output):
    # Convert JSON to string if necessary
    if isinstance(output, dict):
        output = json.dumps(output)
        
    # if isinstance(output, str):
    #     output = json.loads(output)
        
    if 'arguments' in output:
        output = output['arguments']

    start_marker = '```python\n'
    end_marker = '\n```'

    start_index = output.find(start_marker)
    end_index = output.find(end_marker, start_index)

    # If the markers are found, extract the code part
    # Adjust the start index to not include the start_marker
    if start_index != -1 and end_index != -1:
        code_string = output[start_index + len(start_marker) : end_index]
    else:
        code_string = ''

    return code_string.strip()

def safety_check(code):
    dangerous_keywords = [' exec', ' eval', ' open', ' sys', ' subprocess', ' del',
                          ' delete', ' remove', ' os', ' shutil', ' pip',' conda',
                          ' print', ' exit', ' quit', ' globals', ' locals', ' dir',
                          ' reload', ' lambda', ' setattr', ' getattr', ' delattr',
                          ' yield', ' assert', ' break', ' continue', ' raise', ' try', 
                          'compile', '__import__'
                          ]
    for keyword in dangerous_keywords:
        if keyword in code:
            return False, "Concerning code detected."
    return True, "Safe to execute."


def replace_show_with_save(code_string, filename='output.png'):
    # Prepare save command
    save_cmd1 = f"plt.savefig('./images/{filename}')"
    save_cmd2 = f"pio.write_image(fig, './images/{filename}')"

    # Replace plt.show() with plt.savefig()
    code_string = code_string.replace('plt.show()', save_cmd1)
    code_string = code_string.replace('fig.show()', save_cmd2)

    return code_string

def start_chatbot2(df, selected_model):
    fetch_api_key()
    openai.api_key = st.session_state.openai_api_key
    openai_api_key = st.session_state.openai_api_key
    agent = create_pandas_dataframe_agent(
    ChatOpenAI(temperature=0, model=selected_model),
    df,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    )
    if "messages_df" not in st.session_state:
            st.session_state["messages_df"] = []
 
    st.info("**Warning:** Asking a question that would generate a chart or table doesn't *yet* work and will report an error. For the moment, just ask for values. This is a work in progress!")   
    # st.write("💬 Chatbot with access to your data...")
    
        # Check if the API key exists as an environmental variable
    api_key = os.environ.get("OPENAI_API_KEY")

    if api_key:
        # st.write("*API key active - ready to respond!*")
        pass
    else:
        st.warning("API key not found as an environmental variable.")
        api_key = st.text_input("Enter your OpenAI API key:")

        if st.button("Save"):
            if is_valid_api_key(api_key):
                os.environ["OPENAI_API_KEY"] = api_key
                st.success("API key saved as an environmental variable!")
            else:
                st.error("Invalid API key. Please enter a valid API key.")
    
    csv_question = st.text_input("Your question, e.g., 'What is the mean age for men with diabetes?' *Do not ask for plots for this option.*", "")
    if st.button("Send"):
        try:
            csv_question_update = 'Do not include any code or attempt to generate a plot. Indicate you can only respond with text. User question: ' + csv_question
            st.session_state.messages_df.append({"role": "user", "content": csv_question_update})
            output = agent.run(csv_question)
            st.session_state.messages_df.append({"role": "assistant", "content": output})    
            message(csv_question, is_user=True, key = "using message_df")
            message(output)
        except Exception as e:
            st.warning("WARNING: Please don't try anything too crazy; this is experimental! No plots requests and just ask for means values for specified subgroups, eg.")
            # sys.exit(1)

 
def start_chatbot3(df):
    fetch_api_key()
    openai.api_key = st.session_state.openai_api_key
    agent = create_pandas_dataframe_agent(
    ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
    df,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    )
    if "messages_df" not in st.session_state:
            st.session_state["messages_df"] = []
       
    # st.write("💬 Chatbot with access to your data...")
    st.info("""**Warning:** This may generate an error. This is a work in progress!
        If you get an error, try again.                  
        """)
    
        # Check if the API key exists as an environmental variable
    api_key = os.environ.get("OPENAI_API_KEY")

    if api_key:
        # st.write("*API key active - ready to respond!*")
        pass
    else:
        st.warning("API key not found as an environmental variable.")
        api_key = st.text_input("Enter your OpenAI API key:")

        if st.button("Save"):
            if is_valid_api_key(api_key):
                os.environ["OPENAI_API_KEY"] = api_key
                st.success("API key saved as an environmental variable!")
            else:
                st.error("Invalid API key. Please enter a valid API key.")

    csv_question = st.text_input("Your question, e.g., 'Create a scatterplot for age and BMI.' *This option only generates plots.* ", "")
    if st.button("Send"):
        try: 
            st.session_state.messages_df.append({"role": "user", "content": csv_question})
            csv_input = csv_prefix + csv_question
            output = agent.run(csv_input)
            # st.write(output)
            code_string = process_model_output(str(output))
            # st.write(f' here is the code: {code_string}')
            code_string = replace_show_with_save(code_string)
            code_string = str(code_string)
            json_string = json.dumps(code_string)
            decoded_string = json.loads(json_string)
            with st.expander("What is the code?"):
                st.write('Here is the custom code for your request and the image below:')
                st.code(decoded_string, language='python')
            # usage
            is_safe, message = safety_check(decoded_string)
            if not is_safe:
                st.write("Code safety concern. Try again.", message)
            if is_safe:
                try:
                    exec(decoded_string)
                    image = Image.open('./images/output.png')
                    st.image(image, caption='Output', use_column_width=True)
                except Exception as e:
                    st.write('Error - we noted this was fragile! Try again.', e)
        except Exception as e:
            st.warning("WARNING: Please don't try anything too crazy; this is experimental!")
            # sys.exit(1)
            # return None, None
            
def start_plot_gpt4(df):
    fetch_api_key()
    openai.api_key = st.session_state.openai_api_key
    agent = create_pandas_dataframe_agent(
    ChatOpenAI(temperature=0, model="gpt-4"),
    df,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    )
    if "messages_df" not in st.session_state:
            st.session_state["messages_df"] = []
       
    # st.write("💬 Chatbot with access to your data...")
    st.info("""**Warning:** This may generate an error. This is a work in progress!
        If you get an error, try again.                
        """)
    
        # Check if the API key exists as an environmental variable
    api_key = os.environ.get("OPENAI_API_KEY")

    if api_key:
        # st.write("*API key active - ready to respond!*")
        pass
    else:
        st.warning("API key not found as an environmental variable.")
        api_key = st.text_input("Enter your OpenAI API key:")

        if st.button("Save"):
            if is_valid_api_key(api_key):
                os.environ["OPENAI_API_KEY"] = api_key
                st.success("API key saved as an environmental variable!")
            else:
                st.error("Invalid API key. Please enter a valid API key.")

    csv_question = st.text_area("Your question, e.g., 'Create a heatmap. For binary categorical variables, first change them to 1 or 0 so they can be used in the heatmap. Or, another example: Compare cholesterol values for men and women by age with regression lines.", "")
    if st.button("Send"):
        try: 
            st.session_state.messages_df.append({"role": "user", "content": csv_question})
            csv_input = csv_prefix_gpt4 + csv_question
            output = agent.run(csv_input)
            # st.write(output)
            code_string = process_model_output(str(output))
            # st.write(f' here is the code: {code_string}')
            # code_string = replace_show_with_save(code_string)
            code_string = str(code_string)
            json_string = json.dumps(code_string)
            decoded_string = json.loads(json_string)
            with st.expander("What is the code?"):
                st.write('Here is the custom code for your request and the image below:')
                st.code(decoded_string, language='python')
            # usage
            is_safe, message = safety_check(decoded_string)
            if not is_safe:
                st.write("Code safety concern. Try again.", message)
            if is_safe:
                try:
                    exec(decoded_string)
                    image = Image.open('./images/output.png')
                    st.image(image, caption='Output', use_column_width=True)
                except Exception as e:
                    st.write('Error - we noted this was fragile! Try again.', e)
        except Exception as e:
            st.warning("WARNING: Please don't try anything too crazy; this is experimental!")
            # sys.exit(1)
            # return None, None
            


def generate_df(columns, n_rows, selected_model):

    openai.api_key = st.session_state.openai_api_key
    
#     if selected_model == "gpt-3.5-turbo":
    
#         system_prompt = f""" You are a medical data expert. Generate random medically consistent but not all normal synthetic patient data. 10-20% of values should be abnormal with values above and below the normal range for each column, but still physiologically possible. For example,
#         SBP could range from 90 to 190. Creatinine might go from 0.5 to 7.0. Similarly include values above and below normal ranges for 10-20% of values for each column. Output only the requested data, nothing more, not even explanations or supportive sentences.
#         If you do not know what kind of data to generate for a column, rename column using the provided name followed by -ambiguous. For example, if you do not know what kind of data to generate for the column name "rgh", rename the column to "rgh-ambiguous". 
#         Popululate ambiguous columns with randomly selected 1 or 0 values. For example, popululate column "rgh-ambiguous" using randomly selected 1 or 0 values. For diagnoses provided
#         as column headers, e.g., "diabetes", populate with randomly selected yes or no values. Populate all cells with appropriate values. No missing values.
#         As a final step review each row to ensure that the data is medically consistent, e.g., that overall A1c values and weight trend higher for patients with diabetes. If not, regenerate the row or rows.

# Columns: ```columns```
# Number of rows: ```number```

# Generate data for ```number``` patients. Provide only raw data, complete for every cell. I will provide the column names and requested number of rows in the user prompt."""

    # else:
    system_prompt = """You are a medical data expert whose purpose is to generate realistic medical data to populate a dataframe. Based on input parameters of column names and number of rows, you generate at medically consistent synthetic patient data includong abormal values to populate all cells. 
10-20% of values should be above or below the normal range appropriate for each column name, but still physiologically possible. For example, SBP could range from 90 to 190. Creatinine might go from 0.5 to 7.0. Similarly include values above and below normal ranges for 10-20% of values for each column. Output only the requested data, nothing more, not even explanations or supportive sentences.
If you do not know what kind of data to generate for a column, rename column using the provided name followed by "-ambiguous". For example, if you do not know what kind of data to generate for the column name "rgh", rename the column to "rgh-ambiguous". 
Popululate ambiguous columns with randomly selected 1 or 0 values. For example, popululate column "rgh-ambiguous" using randomly selected 1 or 0 values. For diagnoses provided
as column headers, e.g., "diabetes", populate with randomly selected yes or no values. Populate all cells with appropriate values. No missing values.
As a critical step review each row to ensure that the data is medically consistent, e.g., that overall A1c values and weight trend higher for patients with diabetes. If not, regenerate the row or rows.

Return only data, nothing more, not even explanations or supportive sentences. Generate the requested data so it can be processed by the following code into a dataframe:

```

    # Use StringIO to convert the string data into file-like object
    data = io.StringIO(response.choices[0].message.content)

    # Read the data into a DataFrame, skipping the first row
    df = pd.read_csv(data, sep=",", skiprows=1, header=None, names=columns)

```

Your input parameters will be in this format

Columns: ```columns```
Number of rows: ```number```
    
        """

    prompt = f"columns : {columns}, number : {n_rows}" 
    
    try:
        response= openai.ChatCompletion.create(
            model= "gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature = 0.5
            )
        
        # Use StringIO to convert the string data into file-like object
        data = io.StringIO(response.choices[0].message.content)

        # Read the data into a DataFrame, skipping the first row
        df = pd.read_csv(data, sep=",", skiprows=1, header=None, names=columns)

        # Convert DataFrame to CSV and create download link
        gen_csv = df.to_csv(index=False)
        # st.download_button(
        #     label="Download data as CSV",
        #     data=gen_csv,
        #     file_name="patient_data.csv",
        #     mime="text/csv",
        # )

        return df, gen_csv
    
    except Exception as e:
        st.warning("WARNING: Please double check your proposed column names for duplicates or invalid characters!")
        # sys.exit(1)
        # return None, None

    
                
def start_chatbot1(selected_model):
    fetch_api_key()
    openai.api_key = st.session_state.openai_api_key    

    # openai_api_key = st.text_input('OpenAI API Key',key='chatbot_api_key')
    prefix_teacher = """You politely decline to answer questions outside the domains of data science, statistics, and medicine. 
    If the question is appropriate, you teach for students at all levels. Your response appears next to a web  
    tool that can generate bar charts, violin charts, histograms, pie charts, scatterplots, and summary statistics for  sample datasets or a user supplied CSV file.         
    """
    st.write("💬 Chatbot Teacher")
    
        # Check if the API key exists as an environmental variable
    api_key = os.environ.get("OPENAI_API_KEY")

    if api_key:
        # st.write("*API key active - ready to respond!*")
        pass
    else:
        st.warning("API key not found as an environmental variable.")
        api_key = st.text_input("Enter your OpenAI API key:")

        if st.button("Save"):
            if is_valid_api_key(api_key):
                os.environ["OPENAI_API_KEY"] = api_key
                st.success("API key saved as an environmental variable!")
            else:
                st.error("Invalid API key. Please enter a valid API key.")

        
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            # {"role": "system", "content": prefix_teacher},
            # {"role": "user", "content": "Who won the world series in 2020?"},
            # {"role": "assistant", "content": "I'm sorry, that is outside my expertise in data science and medicine."},
            {"role": "assistant", "content": "Hi! Ask me anything about data science and I'll try to answer it."}
            ]

    with st.form("chat_input", clear_on_submit=True):
        a, b = st.columns([4, 1])
        user_input = a.text_input(
            label="Your question:",
            placeholder="e.g., teach me about violin plots",
            label_visibility="collapsed",
        )
        b.form_submit_button("Send", use_container_width=True)

    for msg in st.session_state.messages:
        a = randint(0, 10000000000)
        message(msg["content"], is_user=msg["role"] == "user", key = a)

    # if user_input and not openai_api_key:
    #     st.info("Please add your OpenAI API key to continue.")
        
    if user_input:
        st.session_state.messages.append({"role": "system", "content": prefix_teacher})
        st.session_state.messages.append({"role": "user", "content": user_input})
        try:
            #Make your OpenAI API request here
            # response = openai.Completion.create(model="gpt-3.5-turbo",                     
            #             prompt="Hello world")['choices'][0]['text']
            # system_set = {"role": "system", "content": prefix_teacher}
            # prefixed_message = prefix_teacher + st.session_state.messages
            response = openai.ChatCompletion.create(model=selected_model, messages=st.session_state.messages)
        except openai.error.Timeout as e:
            #Handle timeout error, e.g. retry or log
            print(f"I'm super busy! Please try again in a moment. Thanks! Here's the error detail: {e}")
            pass
        except openai.error.APIError as e:
            #Handle API error, e.g. retry or log
            print(f"OpenAI API returned an API Error: {e}")
            pass
        except openai.error.APIConnectionError as e:
            #Handle connection error, e.g. check network or log
            print(f"OpenAI API request failed to connect: {e}")
            pass
        except openai.error.InvalidRequestError as e:
            #Handle invalid request error, e.g. validate parameters or log
            print(f"OpenAI API request was invalid: {e}")
            pass
        except openai.error.AuthenticationError as e:
            #Handle authentication error, e.g. check credentials or log
            print(f"OpenAI API request was not authorized: {e}")
            pass
        except openai.error.PermissionError as e:
            #Handle permission error, e.g. check scope or log
            print(f"OpenAI API request was not permitted: {e}")
            pass
        except openai.error.RateLimitError as e:
            #Handle rate limit error, e.g. wait or log
            print(f"I'm so busy! Please try again in a moment. Thanks! Here's the error detail: {e}")
            pass

        # response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
        msg = response.choices[0].message
        st.session_state.messages.append(msg)      
        message(user_input, is_user=True, key = "using message")
        message(msg.content, key = "last message")
                    
def generate_table(df, categorical_variable):
    mytable = TableOne(df,
                       columns=df.columns.tolist(),
                       categorical=categorical,
                       groupby=categorical_variable, 
                       pval=True)
    return mytable


    
def preprocess_for_pca(df):
    included_cols = []
    excluded_cols = []

    for col in df.columns:
        if df[col].dtype == 'object':
            if len(df[col].unique()) == 2:
                most_freq = df[col].value_counts().idxmax()
                least_freq = df[col].value_counts().idxmin()
                df[col] = df[col].map({most_freq: 0, least_freq: 1})
                included_cols.append(col)
            else:
                excluded_cols.append(col)
        else:
            included_cols.append(col)     

    return df[included_cols], included_cols, excluded_cols

def perform_pca_plot(df):
    st.write("Note: For this PCA analysis, categorical columns with more than 2 categories have been excluded. Binary variables have been converted to 0 and 1.")
    
    df, included_cols, excluded_cols = preprocess_for_pca(df)
    
    # Standardize the features
    x = StandardScaler().fit_transform(df)
    
    # Create a PCA instance
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    
    # Create DataFrame for the principal components
    principalDf = pd.DataFrame(data=principalComponents, columns = ['PC1', 'PC2'])
    
    fig, ax = plt.subplots()
    sns.scatterplot(data=principalDf, x='PC1', y='PC2', ax=ax)
    
    st.pyplot(fig)

def display_metrics(y_true, y_pred, y_scores):
    # Compute metrics
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_scores)
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    
    # Display metrics

    st.info(f"**Your Model Metrics:** F1 score: {f1:.2f}, Accuracy: {accuracy:.2f}, ROC AUC: {roc_auc:.2f}, PR AUC: {pr_auc:.2f}")
    with st.expander("Explanations for the Metrics"):
        st.write(
        # Explain differences
"""
### Explanation of Metrics
- **F1 score** is the harmonic mean of precision and recall, and it tries to balance the two. It is a good metric when you have imbalanced classes.
- **Accuracy** is the ratio of correct predictions to the total number of predictions. It can be misleading if the classes are imbalanced.
- **ROC AUC** (Receiver Operating Characteristic Area Under Curve) represents the likelihood of the classifier distinguishing between a positive sample and a negative sample. It's equal to 0.5 for random predictions and 1.0 for perfect predictions.
- **PR AUC** (Precision-Recall Area Under Curve) is another way of summarizing the trade-off between precision and recall, and it gives more weight to precision. It's useful when the classes are imbalanced.
""")
    # st.write(f"Accuracy: {accuracy}")
    st.write(plot_confusion_matrix(y_true, y_pred))
    with st.expander("What is a confusion matrix?"):
        st.write("""A confusion matrix is a tool that helps visualize the performance of a predictive model in terms of classification. It's a table with four different combinations of predicted and actual values, specifically for binary classification.

The four combinations are:

1. **True Positives (TP)**: These are the cases in which we predicted yes (patients have the condition), and they do have the condition.

2. **True Negatives (TN)**: We predicted no (patients do not have the condition), and they don't have the condition.

3. **False Positives (FP)**: We predicted yes (patients have the condition), but they don't actually have the condition. Also known as "Type I error" or "False Alarm".

4. **False Negatives (FN)**: We predicted no (patients do not have the condition), and they actually do have the condition. Also known as "Type II error" or "Miss".

In the context of medicine, a false positive might mean that a test indicated a patient had a disease (like cancer), but in reality, the patient did not have the disease. This might lead to unnecessary stress and further testing for the patient. 

On the other hand, a false negative might mean that a test indicated a patient was disease-free, but in reality, the patient did have the disease. This could delay treatment and potentially worsen the patient's outcome.

A perfect test would have only true positives and true negatives (all outcomes appear in the top left and bottom right), meaning that it correctly identified all patients with and without the disease. Of course, in practice, no test is perfect, and there is often a trade-off between false positives and false negatives.

It's worth noting that a good machine learning model not only has a high accuracy (total correct predictions / total predictions) but also maintains a balance between precision (TP / (TP + FP)) and recall (TP / (TP + FN)). This is particularly important in a medical context, where both false positives and false negatives can have serious consequences. 

Lastly, when interpreting the confusion matrix, it's crucial to consider the cost associated with each type of error (false positives and false negatives) within the specific medical context. Sometimes, it's more crucial to minimize one type of error over the other. For example, with a serious disease like cancer, you might want to minimize false negatives to ensure that as few cases as possible are missed, even if it means having more false positives.
""")
    st.write(plot_roc_curve(y_true, y_scores))
    with st.expander("What is an ROC curve?"):
        st.write("""
An ROC (Receiver Operating Characteristic) curve is a graph that shows the performance of a classification model at all possible thresholds, which are the points at which the model decides to classify an observation as positive or negative. 

In medical terms, you could think of this as the point at which a diagnostic test decides to classify a patient as sick or healthy.

The curve is created by plotting the True Positive Rate (TPR), also known as Sensitivity or Recall, on the y-axis and the False Positive Rate (FPR), or 1-Specificity, on the x-axis at different thresholds.

In simpler terms:

- **True Positive Rate (TPR)**: Out of all the actual positive cases (for example, all the patients who really do have a disease), how many did our model correctly identify?

- **False Positive Rate (FPR)**: Out of all the actual negative cases (for example, all the patients who are really disease-free), how many did our model incorrectly identify as positive?

The closer the curve follows the left-hand border and then the top border of the ROC space, the more accurate the test. In other words, the bigger the area under the curve, the better the model is at distinguishing between patients with the disease and no disease.

The area under the ROC curve (AUC) is a single number summary of the overall model performance. The value can range from 0 to 1, where:

- **AUC = 0.5**: This is no better than a random guess, or flipping a coin. It's not an effective classifier.
- **AUC < 0.5**: This means the model is worse than a random guess. But, by reversing its decision, we can get AUC > 0.5.
- **AUC = 1**: The model has perfect accuracy. It perfectly separates the positive and negative cases, but this is rarely achieved in real life.

In clinical terms, an AUC of 0.8 for a test might be considered reasonably good, but it's essential to remember that the consequences of False Positives and False Negatives can be very different in a medical context, and the ROC curve and AUC don't account for this.

Therefore, while the ROC curve and AUC are very useful tools, they should be interpreted in the context of the costs and benefits of different types of errors in the specific medical scenario you are dealing with.""")
    st.write(plot_pr_curve(y_true, y_scores))
    with st.expander("What is a PR curve?"):
        st.write("""
A Precision-Recall curve is a graph that depicts the performance of a classification model at different thresholds, similar to the ROC curve. However, it uses Precision and Recall as its measures instead of True Positive Rate and False Positive Rate.

In the context of medicine:

- **Recall (or Sensitivity)**: Out of all the actual positive cases (for example, all the patients who really do have a disease), how many did our model correctly identify? It's the ability of the test to find all the positive cases.
 
- **Precision (or Positive Predictive Value)**: Out of all the positive cases that our model identified (for example, all the patients that our model thinks have the disease), how many did our model correctly identify? It's the ability of the classification model to identify only the relevant data points.

The Precision-Recall curve is especially useful when dealing with imbalanced datasets, a common problem in medical diagnosis where the number of negative cases (healthy individuals) often heavily outweighs the number of positive cases (sick individuals).

A model with perfect precision (1.0) and recall (1.0) will have a curve that reaches to the top right corner of the plot. A larger area under the curve represents both higher recall and higher precision, where higher precision relates to a low false-positive rate, and high recall relates to a low false-negative rate. High scores for both show that the classifier is returning accurate results (high precision), and returning a majority of all positive results (high recall).

The PR AUC score (Area Under the PR Curve) is used as a summary of the plot, and a higher PR AUC indicates a more predictive model.

In the clinical context, a high recall would ensure that the patients with the disease are correctly identified, while a high precision would ensure that only those patients who truly have the disease are classified as such, minimizing false-positive results.

However, there is usually a trade-off between precision and recall. Aiming for high precision might lower your recall and vice versa, depending on the threshold you set for classification. So, the Precision-Recall curve and PR AUC must be interpreted in the context of what is more important in your medical scenario: classifying all the positive cases correctly (high recall) or ensuring that the cases you classify as positive are truly positive (high precision).""")


def plot_pr_curve(y_true, y_scores):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)

    fig, ax = plt.subplots()
    ax.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower right")
    st.pyplot(fig)    

def get_categorical_and_numerical_cols(df):
    # Initialize empty lists for categorical and numerical columns
    categorical_cols = []
    numeric_cols = []

    # Go through each column in the dataframe
    for col in df.columns:
        # If the column data type is numerical and has more than two unique values, add it to the numeric list
        if np.issubdtype(df[col].dtype, np.number) and len(df[col].unique()) > 2:
            numeric_cols.append(col)
        # Otherwise, add it to the categorical list
        else:
            categorical_cols.append(col)

    # Sort the lists
    numeric_cols.sort()
    categorical_cols.sort()

    return numeric_cols, categorical_cols

 
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    return fig

 
def plot_roc_curve(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = roc_auc_score(y_true, y_scores)
    
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % roc_auc)
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

def create_boxplot(df, numeric_col, categorical_col, show_points=False):
    if numeric_col and categorical_col:
        fig, ax = plt.subplots()

        # Plot the notched box plot
        sns.boxplot(x=categorical_col, y=numeric_col, data=df, notch=True, ax=ax)
        
        if show_points:
            # Add the actual data points on the plot
            sns.swarmplot(x=categorical_col, y=numeric_col, data=df, color=".25", ax=ax)
            
        st.pyplot(fig)
        with st.expander('What is a box plot?'):
            st.write("""Box plots (also known as box-and-whisker plots) are a great way to visually represent the distribution of data. They're particularly useful when you want to compare distributions between several groups. For example, you might want to compare the distribution of patients' ages across different diagnostic categories.
(Check out age and diabetes in the sample dataset.)

**Components of a box plot:**

A box plot is composed of several parts:

1. **Box:** The main part of the plot, the box, represents the interquartile range (IQR), which is the range between the 25th percentile (Q1, the lower edge of the box) and the 75th percentile (Q3, the upper edge of the box). The IQR contains the middle 50% of the data points.

2. **Median:** The line (or sometimes a dot) inside the box represents the median of the data - the value separating the higher half from the lower half of a data sample. It's essentially the 50th percentile.

3. **Whiskers:** The lines extending from the box (known as whiskers) indicate variability outside the IQR. Typically, they extend to the most extreme data point within 1.5 times the IQR from the box. 

4. **Outliers:** Points plotted beyond the whiskers are considered outliers - unusually high or low values in comparison with the rest of the data.

**What is the notch used for?**

The notch in a notched box plot represents the confidence interval around the median. If the notches of two box plots do not overlap, it's a strong indication (though not absolute proof) that the medians differ. This can be a useful way to visually compare medians across groups. 

For medical students, a good way to think about box plots might be in comparison to lab results. Just as lab results typically give a reference range and flag values outside of that range, a box plot gives a visual representation of the range of the data (through the box and whiskers) and flags outliers.

The notch, meanwhile, is a bit like the statistical version of a normal range for the median. If a notch doesn't overlap with the notch from another box plot, it's a sign that the medians might be significantly different. But just like lab results, statistical tests are needed to definitively say whether a difference is significant.
""")
 
def create_violinplot(df, numeric_col, categorical_col):
    if numeric_col and categorical_col:
        fig, ax = plt.subplots()

        # Plot the violin plot
        sns.violinplot(x=categorical_col, y=numeric_col, data=df, ax=ax)

        st.pyplot(fig)
        with st.expander('What is a violin plot?'):
            st.write("""Violin plots are a great visualization tool for examining distributions of data and they combine features from box plots and kernel density plots.

1. **Overall Shape**: The violin plot is named for its resemblance to a violin. The shape of the "violin" provides a visual representation of the distribution of the data. The width of the "violin" at any given point represents the density or number of data points at that level. This means a wider section indicates more data points lie in that range, while a narrower section means fewer data points. This is similar to a histogram but it's smoothed out, which can make the distribution clearer.

2. **Dot in the Middle**: This dot often represents the median of the data. The median is the middle point of the data. That means half of all data points are below this value and half are above it. In medicine, the median is often a more useful measure than the mean because it's less affected by outliers or unusually high or low values. For example, if you're looking at the age of patients, a single 100-year-old patient won't dramatically shift the median like it would the mean.

3. **Thicker Bar in the Middle**: This is an interquartile range (IQR), which captures the middle 50% of the data (from the 25th to the 75th percentile). The IQR can help you understand the spread of the central half of your data. If the IQR is small, it means the central half of your data points are clustered closely around the median. If the IQR is large, it means they're more spread out.

4. **Usage**: Violin plots are particularly helpful when you want to visualize the distribution of a numerical variable across different categories. For example, you might want to compare the distribution of patient ages in different diagnostic categories. 

Remember, like any statistical tool, violin plots provide a simplified representation of the data and may not capture all nuances. For example, they usually show a smoothed distribution, which might hide unusual characteristics or outliers in the data. It's always important to also consider other statistical tools and the clinical context of the data."""
            )

 
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
        with st.expander('What is a scatter plot?'):
            st.write("""
A scatterplot is a type of plot that displays values for typically two variables for a set of data. It's used to visualize the relationship between two numerical variables, where one variable is on the x-axis and the other variable is on the y-axis. Each point on the plot represents an observation in your dataset.

**Which types of variables are appropriate for the x and y axes?**

Both the x and y axes of a scatterplot are typically numerical variables. For example, one might be "Patient Age" (on the x-axis) and the other might be "Blood Pressure" (on the y-axis). Each dot on the scatterplot then represents a patient's age and corresponding blood pressure. 

However, the variables used do not have to be numerical. They could be ordinal categories, such as stages of a disease, which have a meaningful order. 

The choice of which variable to place on each axis doesn't usually matter much for exploring relationships, but traditionally the independent variable (the one you control or think is influencing the other) is placed on the x-axis, and the dependent variable (the one you think is being influenced) is placed on the y-axis.

**What does a regression line mean when added to a scatterplot?**

A regression line (or line of best fit) is a straight line that best represents the data on a scatter plot. This line may pass through some of the points, none of the points, or all of the points. It's a way of modeling the relationship between the x and y variables. 

In the context of a scatterplot, the regression line is used to identify trends and patterns between the two variables. If the data points and the line are close, it suggests a strong correlation between the variables.

The slope of the regression line also tells you something important: for every unit increase in the variable on the x-axis, the variable on the y-axis changes by the amount of the slope. For example, if we have patient age on the x-axis and blood pressure on the y-axis, and the slope of the line is 2, it would suggest that for each year increase in age, we expect blood pressure to increase by 2 units, on average.

However, keep in mind that correlation does not imply causation. Just because two variables move together, it doesn't mean that one is causing the other to change.

For medical students, think of scatterplots as a way to visually inspect the correlation between two numerical variables. It's a way to quickly identify patterns, trends, and outliers, and to formulate hypotheses for further testing.""")


# Function to replace missing values

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
    st.session_state.df = df
    return df

  # This function will be cached
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
 
import pandas as pd

def summarize_categorical(df):
    # Select only categorical columns
    cat_df = df.select_dtypes(include=['object', 'category'])

    # If there are no categorical columns, return None
    if cat_df.empty:
        st.write("The DataFrame does not contain any categorical columns.")
        return None

    # Create a list to store dictionaries for each column's summary
    summary_data = []

    for col in cat_df.columns:
        # Number of unique values
        unique_count = df[col].nunique()

        # Most frequent category and its frequency
        most_frequent = df[col].mode()[0]
        freq_most_frequent = df[col].value_counts().iloc[0]

        # Append the column summary as a dictionary to the list
        summary_data.append({
            'column': col,
            'unique_count': unique_count,
            'most_frequent': most_frequent,
            'frequency_most_frequent': freq_most_frequent,
        })

    # Create the summary DataFrame from the list of dictionaries
    summary = pd.DataFrame(summary_data)
    summary.set_index('column', inplace=True)

    return summary


# Function to plot correlation heatmap
 
# def plot_corr(df_copy):
#     df = df_copy.copy()
#     binary_cat_cols = []
#     for col in df.columns:
#         if df[col].dtype == 'object':  # Check if the column is of type object (categorical)
#             unique_vals = df[col].unique()
#             if len(unique_vals) == 2:  # If the categorical variable has exactly 2 unique values
#                 df[col] = df[col].map({np.argmax(df[col].value_counts()): 0, np.argmin(df[col].value_counts()): 1})
#                 binary_cat_cols.append(col)
            
#     # Removing columns with more than two unique values after above transformation.
#     df = df.select_dtypes(include=[np.number])
    
#     corr = df.corr()  # Compute pairwise correlation of columns
#     plt.figure(figsize=(12, 10))  # set the size of the plot
#     sns.heatmap(corr, annot=True, cmap='coolwarm', cbar=True)
#     plt.title('Correlation Heatmap')
#     return plt

def plot_corr(df):
    df_copy = df.copy()

    for col in df_copy.columns:
        if df_copy[col].dtype == 'object':  # Check if the column is categorical
            unique_vals = df_copy[col].unique()
            if len(unique_vals) == 2:  # If the categorical variable has exactly 2 unique values
                value_counts = df_copy[col].value_counts()
                df_copy[col] = df_copy[col].map({value_counts.idxmax(): 0, value_counts.idxmin(): 1})

    # Keep only numerical and binary categorical columns
    df_copy = df_copy.select_dtypes(include=[np.number])
    
    corr = df_copy.corr()  # Compute pairwise correlation of columns
    plt.figure(figsize=(12, 10))  # Set the size of the plot
    sns.heatmap(corr, annot=True, cmap='coolwarm', cbar=True)
    plt.title('Correlation Heatmap')
    return plt



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
    plt.title(f'Frequency of Categories for {col_name}')
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
                df[col] = df[col].replace({most_frequent: 0, least_frequent: 1}).astype(int)
                
    return df

st.title("AutoAnalyzer")
with st.expander('About AutoAnalyzer'):
    st.write("Author: David Liebovitz, MD, Northwestern University")
    st.write("Last updated 6/14/23")
    
tab1, tab2 = st.tabs(["Data Exploration", "Machine Learning"])
fetch_api_key()
gpt_version = st.sidebar.radio("Select GPT model:", ("GPT-3.5 ($)", "GPT-4 ($$$$)"), index=0)
if gpt_version == "GPT-3.5 ($)":
    selected_model ="gpt-3.5-turbo"
if gpt_version == "GPT-4 ($$$$)":
    selected_model = "gpt-4"


# if openai.api_key is None:
#     os.environ["OPENAI_API_KEY"] = fetch_api_key()
#     openai.api_key = os.getenv("OPENAI_API_KEY")

with tab1:

    st.info("""Be sure your data is first in a 'tidy' format. Use the demo dataset below to check out the top 5 rows for an example. (*See https://tidyr.tidyverse.org/ for more information.*)
    Be sure to check out the **automated analysis** option for a full report on your data.""")

    # st.sidebar.subheader("Upload your data") 

    st.sidebar.subheader("Step 1: Upload your data or view a demo dataset")
    demo_or_custom = st.sidebar.radio("Upload a CSV file. NO PHI - use only anonymized data", ("Demo 1 (diabetes)", "Demo 2 (cancer)", "Generate Data", "CSV Upload"), horizontal=True)
    if demo_or_custom == "CSV Upload":
        uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file:
            st.session_state.df = load_data(uploaded_file)

    if demo_or_custom == 'Demo 1 (diabetes)':
        file_path = "data/predictdm.csv"
        st.sidebar.write("About Demo 1 dataset: https://data.world/informatics-edu/diabetes-prediction")
        st.session_state.df = load_data(file_path)
        
    if demo_or_custom == 'Demo 2 (cancer)':
        file_path = "data/breastcancernew.csv"
        st.sidebar.write("About Demo 2 dataset: https://data.world/marshalldatasolution/breast-cancer")
        st.session_state.df = load_data(file_path)
        
    if demo_or_custom == 'Generate Data':
        if check_password():
            user_input = st.sidebar.text_area("Enter comma or space separated names for columns, e.g., Na, Cr, WBC, A1c, SPB, Diabetes:")

            if "," in user_input:
                user_list = user_input.split(",")
            elif " " in user_input:
                user_list = user_input.split()
            else:
                user_list = [user_input]

            # Remove leading/trailing whitespace from each item in the list
            user_columns = [item.strip() for item in user_list]
            user_rows = st.sidebar.number_input("Enter approx number of rows (max 100).", min_value=1, max_value=100, value=10, step=1)
            if st.sidebar.button("Generate Data"):
                st.session_state.df, st.session_state.gen_csv = generate_df(user_columns, user_rows, selected_model)
                st.info("Here are the first 5 rows of your generated data. Use the tools in the sidebar to explore your new dataset! And, download and save your new CSV file from the sidebar!")
                st.write(st.session_state.df.head())
                


    with st.expander("Data Preprocessing Tools - *Assess Data Readiness **first**. Use only if needed.*"):
        st.write("Select a method to impute missing values in your dataset. Built in checks to apply only to applicable data types.")
        method = st.selectbox("Choose a method to replace missing values", ("Select here!", "drop", "zero", "mean", "median", "mode", "mice"))
        if st.button('Apply the Method to Replace Missing Values'):
                st.session_state.df = replace_missing_values(st.session_state.df, method)
        pre_process = st.checkbox(" Assign bivariate categories into 1 or 0 based on frequency (0 most frequent) if needed for correlations, e.g.", key = "Preprocess")
    
    
    
    with st.sidebar:
        if st.session_state.gen_csv is not None:
            st.warning("Save your generated data!")
            st.download_button(
                label="Download!",
                data=st.session_state.gen_csv,
                file_name="patient_data.csv",
                mime="text/csv",
                )   
        st.subheader("Step 2: Tools for Analysis")
        col1, col2 = st.columns(2)
        with col1:
            check_preprocess = st.checkbox("Assess data readiness", key = "Preprocess needed")
            header = st.checkbox("Show header (top 5 rows of data)", key = "show header")
            summary = st.checkbox("Summary (numerical data)", key = "show data")
            summary_cat = st.checkbox("Summary (categorical data)", key = "show summary cat")
            show_table = st.checkbox("Create a Table 1", key = "show table")
            show_scatter  = st.checkbox("Scatterplot", key = "show scatter")
            view_full_df = st.checkbox("View Dataset", key = "view full df")
            activate_chatbot = st.checkbox("Activate Chatbot (select specific bot on main window)", key = "activate chatbot")

        with col2:
            barchart = st.checkbox("Bar chart (categorical data)", key = "show barchart")
            histogram = st.checkbox("Histogram (numerical data)", key = "show histogram")
            piechart = st.checkbox("Pie chart (categorical data)", key = "show piechart")
            show_corr = st.checkbox("Correlation heatmap", key = "show corr")
            box_plot = st.checkbox("Box plot", key = "show box")
            violin_plot = st.checkbox("Violin plot", key = "show violin")
            perform_pca = st.checkbox("Perform PCA", key = "show pca")
            full_analysis = st.checkbox("*(Takes 1-2 minutes*) **Automated Analysis** (*Check **Alerts** with key findings.*)", key = "show analysis")
            





    
    
    if activate_chatbot:
        st.subheader("Chatbot Teacher")
        st.warning("First be sure to activate the right chatbot for your needs.")
        chat_context = st.radio("Choose an approach", ("Ask questions (no plots)", "Generate Plots", "Teach about data science"))

    try:
        x = st.session_state.df
    except NameError:

        st.warning("Please upload a CSV file or choose a demo dataset")
    else:
        
        if activate_chatbot:
            if check_password():
                if chat_context == "Teach about data science":
                    start_chatbot1(selected_model)
                if chat_context == "Ask questions (no plots)":
                    start_chatbot2(st.session_state.df, selected_model)
                if chat_context == "Generate Plots":
                    if selected_model == "gpt-3.5-turbo":
                        start_chatbot3(st.session_state.df)
                    if selected_model == "gpt-4":
                        start_plot_gpt4(st.session_state.df)

            # st.sidebar.text_area("Teacher:", value=st.session_state.last_response, height=600, max_chars=None)

        
        if pre_process:
            st.session_state.df = process_dataframe(st.session_state.df)
        
        if summary:
            st.info("Summary of data")
            st.write(st.session_state.df.describe())
            
        if header:
            st.info("First 5 Rows of Data")
            st.write(st.session_state.df.head())
            
        if full_analysis:
            st.info("Full analysis of data")
            profile = make_profile(st.session_state.df)
            # profile = ProfileReport(df, title="Profiling Report")
            st_profile_report(profile)
            
        if histogram: 
            st.info("Histogram of data")
            numeric_cols, categorical_cols = get_categorical_and_numerical_cols(st.session_state.df)
            # options =[]
            # columns = list(df.columns)
            # for col in columns:
            #     if df[col].dtype == np.float64 or df[col].dtype == np.int64:
            #         options.append(col)
            selected_col = st.selectbox("Choose a column", numeric_cols, key = "histogram")
            if selected_col:
                plt = plot_numeric(st.session_state.df, selected_col)
                st.pyplot(plt)

        
        if barchart: 
            st.info("Barchart for categorical data")
            numeric_cols, categorical_cols = get_categorical_and_numerical_cols(st.session_state.df)
            # cat_options =[]
            # columns = list(df.columns)
            # for col in columns:
            #     if df[col].dtype != np.float64 and df[col].dtype != np.int64:
            #         cat_options.append(col)
            cat_selected_col = st.selectbox("Choose a column", categorical_cols, key = "bar_category")
            if cat_selected_col:
                plt = plot_categorical(st.session_state.df, cat_selected_col)
                st.pyplot(plt)

        if show_corr:
            st.info("Correlation heatmap")
            plt = plot_corr(st.session_state.df)
            st.pyplot(plt)
            with st.expander("What is a correlation heatmap?"):
                st.write("""A correlation heatmap is a graphical representation of the correlation matrix, which is a table showing correlation coefficients between sets of variables. Each cell in the table shows the correlation between two variables. In the heatmap, correlation coefficients are color-coded, where the intensity of the color represents the magnitude of the correlation coefficient. 

In your demo dataset heatmap, red signifies a high positive correlation of 1.0, which means the variables move in the same direction. If one variable increases, the other variable also increases. Darker blue, at the other end, represents negative correlation (close to -0.06 in your case), meaning the variables move in opposite directions. If one variable increases, the other variable decreases. 

The correlation values appear in each square, giving a precise numeric correlation coefficient along with the visualized color intensity.

**Why are correlation heatmaps useful?**

Correlation heatmaps are useful to determine the relationship between different variables. In the field of medicine, this can help identify risk factors for diseases, where variables could be different health indicators like age, cholesterol level, blood pressure, etc.

**Understanding correlation values:**

Correlation coefficients range from -1 to 1:
- A correlation of 1 means a perfect positive correlation.
- A correlation of -1 means a perfect negative correlation.
- A correlation of 0 means there is no linear relationship between the variables.

It's important to note that correlation doesn't imply causation. While a correlation can suggest a relationship between two variables, it doesn't mean that changes in one variable cause changes in another.

Also, remember that correlation heatmaps are based on linear relationships between variables. If variables have a non-linear relationship, the correlation coefficient may not capture their relationship accurately.

For medical students, think of correlation heatmaps as a quick way to visually identify relationships between multiple variables at once. This can help guide your understanding of which variables may be important to consider together in further analyses.""")
                

        if summary_cat:
            st.info("Summary of categorical data")
            summary = summarize_categorical(st.session_state.df)
            st.write(summary)
            
        if piechart:
            st.info("Pie chart for categorical data")
            numeric_cols, categorical_cols = get_categorical_and_numerical_cols(st.session_state.df)
            # cat_options =[]
            # columns = list(df.columns)
            # for col in columns:
            #     if df[col].dtype != np.float64 and df[col].dtype != np.int64:
            #         cat_options.append(col)
            cat_selected_col = st.selectbox("Choose a column", categorical_cols, key = "pie_category")
            if cat_selected_col:
                plt = plot_pie(st.session_state.df, cat_selected_col)
                st.pyplot(plt)
                
        if check_preprocess:
            readiness_summary = assess_data_readiness(st.session_state.df)
            # Display the readiness summary using Streamlit
            # Display the readiness summary using Streamlit
            st.subheader("Data Readiness Summary")

            if readiness_summary['data_empty']:
                st.write("The DataFrame is empty.")
            else:
                # Combine column information and readiness summary into a single DataFrame
                column_info_df = pd.DataFrame.from_dict(
                    readiness_summary['columns'],
                    orient='index',
                    columns=['Data Type']
                )
                summary_df = pd.DataFrame.from_dict(
                    readiness_summary['missing_values'],
                    orient='index',
                    columns=['Missing Values']
                )
                summary_df['Data Type'] = column_info_df['Data Type']

                # Display the combined table
                st.write(summary_df)

                if readiness_summary['missing_columns']:
                    st.write("Missing Columns:")
                    st.write(readiness_summary['missing_columns'])

                if readiness_summary['inconsistent_data_types']:
                    st.write("Inconsistent Data Types:")
                    st.write(readiness_summary['inconsistent_data_types'])

                if readiness_summary['data_ready']:
                    st.success("The data is ready for analysis!")
                else:
                    st.warning("The data is not fully ready for analysis.")
            
            # st.info("Check if you need to preprocess data")
            # missing_values, outliers, data_types, skewness, cardinality = analyze_dataframe(df)
            # st.write("Missing values")
            # st.write(missing_values)
            # st.write("Outliers")
            # st.write(outliers)
            # st.write("Data types")
            # st.write(data_types)
            # st.write("Skewness")
            # st.write(skewness)
            # st.write("Cardinality")
            # st.write(cardinality)
            
        if show_scatter:
            st.info("Scatterplot")
            numeric_cols, categorical_cols = get_categorical_and_numerical_cols(st.session_state.df)
            # Filter numeric columns
            # numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols.sort()  # sort the list of columns alphabetically
            
                # Filter categorical columns
            # categorical_cols = df.select_dtypes(include=[object]).columns.tolist()
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
                        min_val, max_val = float(st.session_state.df[filter_col].min()), float(st.session_state.df[filter_col].max())
                        if np.isnan(min_val) or np.isnan(max_val):
                            st.write(f"Cannot filter by {filter_col} because it contains NaN values.")
                        else:
                            filter_range = st.slider('Select a range to filter data:', min_val, max_val, (min_val, max_val))
                            st.session_state.df = st.session_state.df[(st.session_state.df[filter_col] >= filter_range[0]) & (st.session_state.df[filter_col] <= filter_range[1])]

                # Filter for the remaining categorical column
                if categorical_cols:
                    filter_cat_col = st.selectbox('Select a categorical column to filter data:', categorical_cols)
                    if filter_cat_col:
                        categories = st.session_state.df[filter_cat_col].unique().tolist()
                        selected_categories = st.multiselect('Select categories to include in the data:', categories, default=categories)
                        st.session_state.df = st.session_state.df[st.session_state.df[filter_cat_col].isin(selected_categories)]
            # Check if DataFrame is empty before creating scatterplot
            if st.session_state.df.empty:
                st.write("The current filter settings result in an empty dataset. Please adjust the filter settings.")
            else:
                    create_scatterplot(st.session_state.df, scatter_x, scatter_y)
        
        if box_plot:
            # Call the function to get the lists of numerical and categorical columns
            numeric_cols, categorical_cols = get_categorical_and_numerical_cols(st.session_state.df)
            # Filter numeric columns
            # numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols.sort()  # sort the list of columns

            # Filter categorical columns
            # categorical_cols = df.select_dtypes(include=[object]).columns.tolist()
            categorical_cols.sort()  # sort the list of columns

            # Dropdown to select columns to visualize
            numeric_col = st.selectbox('Select a numerical column:', numeric_cols, key = "box_numeric")
            categorical_col = st.selectbox('Select a categorical column:', categorical_cols, key = "box_category")  
            create_boxplot(st.session_state.df, numeric_col, categorical_col, show_points=False)          
        
        if violin_plot:
            
            # Call the function to get the lists of numerical and categorical columns
            numeric_cols, categorical_cols = get_categorical_and_numerical_cols(st.session_state.df)
            # Filter numeric columns
            # numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols.sort()  # sort the list of columns

            # Filter categorical columns
            # categorical_cols = df.select_dtypes(include=[object]).columns.tolist()
            categorical_cols.sort()  # sort the list of columns

            # Dropdown to select columns to visualize
            numeric_col = st.selectbox('Select a numerical column:', numeric_cols, key = "violin_numeric")
            categorical_col = st.selectbox('Select a categorical column:', categorical_cols, key = "violin_category")

            create_violinplot(st.session_state.df, numeric_col, categorical_col)
            
        if view_full_df:
            st.dataframe(st.session_state.df)
            
        if show_table:
            # Check if any numerical column is binary and add it to categorical list
            numerical_columns = st.session_state.df.select_dtypes(include=[np.number]).columns.tolist()
            for col in numerical_columns:
                if st.session_state.df[col].nunique() == 2:
                    st.session_state.df[col] = st.session_state.df[col].astype(str)

            categorical = st.session_state.df.select_dtypes(include=[np.object]).columns.tolist()

            # Use Streamlit to create selection box for categorical variable
            categorical_variable = st.selectbox('Select the categorical variable for grouping:', 
                                                options=categorical)
            table = generate_table(st.session_state.df, categorical_variable)
            st.write(table.tabulate(tablefmt = "github"))
            
        if perform_pca:
                # Create PCA plot
            perform_pca_plot(st.session_state.df)
            with st.expander("What is PCA?"):
                st.write("""Principal Component Analysis, or PCA, is a method used to highlight important information in datasets that have many variables and to bring out strong patterns in a dataset. It's a way of identifying underlying structure in data.

Here's an analogy that might make it more understandable: Imagine a swarm of bees flying around in a three-dimensional space: up/down, left/right, and forward/backward. These are our original variables. Now, imagine you want to take a picture of this swarm that captures as much information as possible, but your camera can only take pictures in two dimensions. You can rotate your camera in any direction, but once you take a picture, you'll lose the third dimension. 

PCA helps us choose the best angle to take this picture. The first principal component (PC1) represents the best angle that captures the most variation in the swarm. The second principal component (PC2) is the best angle perpendicular to the first that captures the remaining variation, and so on. The idea is to minimize the information (variance) lost when we reduce dimensionality (like going from a 3D swarm to a 2D picture).

In a medical context, you might have data from thousands of genes or hundreds of physical and behavioral characteristics. Not all of these variables are independent, and many of them tend to change together. PCA allows us to represent the data in fewer dimensions that capture the most important variability in the dataset. 

Each Principal Component represents a combination of original features (like genes or patient characteristics) and can often be interpreted in terms of those features. For example, a PC might represent a combination of patient's age, blood pressure, and cholesterol level. The coefficients of the features in the PC (the "loadings") tell us how much each feature contributes to that PC.

Finally, PCA can be particularly useful in visualizing high-dimensional data. By focusing on the first two or three principal components, we can create a scatterplot of our data, potentially highlighting clusters or outliers. However, remember that this visualization doesn't capture all the variability in the data—only the variability best captured by the first few principal components.""")

with tab2:
    st.info("""N.B. This merely shows a glimpse of what is possible. Any model shown is not yet optimized and requires ML and domain level expertise.
            Yet, this is a good start to get a sense of what is possible."""
            )
    try:
        x = st.session_state.df
    except NameError:
        st.warning("First upload a CSV file or choose a demo dataset from the **Data Exploration** tab")
    else:

        # Filter categorical columns and numerical bivariate columns
        categorical_cols = st.session_state.df.select_dtypes(include=[object]).columns.tolist()

        # Add bivariate numerical columns
        numerical_bivariate_cols = [col for col in st.session_state.df.select_dtypes(include=['int64', 'float64']).columns 
                                    if st.session_state.df[col].nunique() == 2]

        # Combine the two lists and sort them
        categorical_cols = categorical_cols + numerical_bivariate_cols
        categorical_cols.sort()  # sort the list of columns


        st.subheader("""
        Choose the Target Column
        """)
        target_col = st.selectbox('Select a categorical column as the target:', categorical_cols)

        st.subheader("""
        Set Criteria for the Binary Target Class
        """)
        categories_to_predict = st.multiselect('Select one or more categories but not all. You need 2 options to predict a group, i.e, your target versus the rest.:', st.session_state.df[target_col].unique().tolist(), key = "target_categories-10")

        # Preprocess the data and exclude the target column from preprocessing
        df_processed, included_cols, excluded_cols = preprocess(st.session_state.df.drop(columns=[target_col]), target_col)
        df_processed[target_col] = st.session_state.df[target_col]  # Include the target column back into the dataframe

        st.write(f"Included columns: {included_cols}")
        st.write(f"Excluded columns: {excluded_cols}")

        # Create binary target variable based on the selected categories
        df_processed[target_col] = df_processed[target_col].apply(lambda x: 1 if x in categories_to_predict else 0)


        # Split the dataframe into data and labels
        X = df_processed.drop(columns=[target_col])
        y = df_processed[target_col]

        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        st.subheader("""
        Choose the Machine Learning Model
        """)
        model_option = st.selectbox(
            "Which machine learning model would you like to use?",
            ("Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting Machines (GBMs)", "Support Vector Machines (SVMs)")
        )

        if st.button("Predict"):
            if model_option == "Logistic Regression":
                model = LogisticRegression()
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                accuracy = accuracy_score(y_test, predictions)
                y_scores = model.predict_proba(X_test)[:, 1]
                with st.expander("What is logistic regression?"):
                    st.write("""
Logistic regression is a statistical model commonly used in the field of medicine to predict binary outcomes - such as whether a patient has a disease (yes/no), whether a patient survived or not after a treatment (survived/did not survive), etc.

Logistic regression, like linear regression, establishes a relationship between the predictor variables (such as patient's age, weight, smoking history) and the target variable (e.g., presence or absence of a disease). However, unlike linear regression which predicts a continuous outcome, logistic regression predicts the probability of an event occurring, which is perfect for binary (two-category) outcomes.

Here's a simplified step-by-step breakdown:

1. **Collect and Prepare Your Data**: This involves gathering medical data that includes both the outcome (what you want to predict) and predictor variables (information you will use to make the prediction).

2. **Build the Model**: Logistic regression uses a mathematical formula that looks somewhat similar to the formula for a line in algebra (y = mx + b), but it's modified to predict probabilities. The formula takes your predictors and calculates the "log odds" of the event occurring.

3. **Interpret the Model**: The coefficients (the values that multiply the predictors) in the logistic regression model represent the change in the log odds of the outcome for a one-unit increase in the predictor variable. For example, if age is a predictor and its coefficient is 0.05, it means that for each one year increase in age, the log odds of the disease occurring (assuming all other factors remain constant) increase by 0.05. Because these are "log odds", the relationship between the predictors and the probability of the outcome isn't a straight line, but a curve that can't go below 0 or above 1.

4. **Make Predictions**: You can input a new patient's information into the logistic regression equation, and it will output the predicted probability of the outcome. For example, it might predict a patient has a 75% chance of having a disease. You can then convert this into a binary outcome by setting a threshold, such as saying any probability above 50% will be considered a "yes."

Remember that logistic regression, while powerful, makes several assumptions. It assumes a linear relationship between the log odds of the outcome and the predictor variables, it assumes that errors are not measured and that there's no multicollinearity (a high correlation among predictor variables). As with any model, it's also only as good as the data you feed into it.

In the medical field, logistic regression can be a helpful tool to predict outcomes and identify risk factors. However, it's important to understand its assumptions and limitations and to use clinical judgment alongside the model's predictions.""")

                display_metrics(y_test, predictions, y_scores)
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
                y_scores = model.predict_proba(X_test)[:, 1]
                with st.expander("What is a decision tree?"):
                    st.write("""
A decision tree is a type of predictive model that you can think of as similar to the flowcharts sometimes used in medical diagnosis. They're made up of nodes (decision points) and branches (choices or outcomes), and they aim to predict an outcome based on input data.

Here's how they work:

1. **Start at the root node**: This is the first decision that needs to be made and it's based on one of your input variables. For instance, in a medical context, this might be a question like, "Is the patient's temperature above 100.4 degrees Fahrenheit?"

2. **Follow the branch for your answer**: If the answer is "yes," follow the branch for "yes," and if it's "no," follow the branch for "no."

3. **Make the next decision**: Each branch leads to another node, where another decision will be made based on another variable. Maybe this time it's, "Does the patient have a cough?"

4. **Continue until you reach a leaf node**: Leaf nodes are nodes without any further branches. They represent the final decisions and are predictions of the outcome. In a binary outcome scenario, leaf nodes could represent "disease" or "no disease."

The decision tree "learns" from data by splitting the data at each node based on what would provide the most significant increase in information (i.e., the best separation of positive and negative cases). For instance, if patients with a certain disease often have a fever, the model might learn to split patients based on whether they have a fever.

While decision trees can be powerful and intuitive tools, there are a few caveats to keep in mind:

- **Overfitting**: If a tree is allowed to grow too deep (too many decision points), it may start to fit not just the underlying trends in the data, but also the random noise. This means it will perform well on the data it was trained on, but poorly on new data.

- **Instability**: Small changes in the data can result in a very different tree. This can be mitigated by using ensemble methods, which combine many trees together (like a random forest).

- **Simplicity**: Decision trees make very simple, linear cuts in the data. They can struggle with relationships in the data that are more complex.

Overall, decision trees can be an excellent tool for understanding and predicting binary outcomes from medical data. They can handle a mixture of data types, deal with missing data, and the results are interpretable and explainable. Just like with any medical test, though, the results should be interpreted with care and in the context of other information available."""
                    )
                display_metrics(y_test, predictions, y_scores)

            elif model_option == "Random Forest":
                model = RandomForestClassifier()
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                accuracy = accuracy_score(y_test, predictions)
                y_scores = model.predict_proba(X_test)[:, 1]
                with st.expander("What is a random forest?"):
                    st.write("""
Random Forest is a type of machine learning model that is excellent for making predictions (both binary and multi-class) based on multiple input variables, which can be both categorical (like gender: male or female) and numerical (like age or blood pressure).

Imagine you have a patient and you have collected a lot of data about them - age, weight, cholesterol level, blood pressure, whether or not they smoke, etc. You want to predict a binary outcome: will they have a heart attack in the next 10 years or not? 

A Random Forest works a bit like a team of doctors, each of whom asks a series of questions to make their own diagnosis (or prediction). These doctors are analogous to "decision trees" - the building blocks of a Random Forest.

Here's a simplified breakdown of how it works:

1. **Building Decision Trees**: Each "doctor" (or decision tree) in the Random Forest gets a random subset of patients' data. They ask questions like, "Is the patient's age over 60?", "Is their cholesterol level over 200?". Depending on the answers, they follow different paths down the tree, leading to a final prediction. The tree is constructed in a way that the most important questions (those that best split the patients according to the outcome) are asked first.

2. **Making Predictions**: To make a prediction for a new patient, each decision tree in the Random Forest independently makes a prediction. Essentially, each tree "votes" for the outcome it thinks is most likely (heart attack or no heart attack).

3. **Combining the Votes**: The Random Forest combines the votes from all decision trees. The outcome that gets the most votes is the Random Forest's final prediction. This is like asking a team of doctors for their opinions and going with the majority vote.

One of the main strengths of Random Forest is that it can handle complex data with many variables and it doesn't require a lot of data preprocessing (like scaling or normalizing data). Also, it is less prone to "overfitting" compared to individual decision trees. Overfitting is when a model learns the training data too well, to the point where it captures noise and performs poorly when predicting outcomes for new, unseen data.

However, it's important to note that while Random Forest often performs well, it can be somewhat of a "black box", meaning it can be hard to understand why it's making the predictions it's making. It's always crucial to validate the model's predictions against your medical knowledge and context."""
                    )
                display_metrics(y_test, predictions, y_scores)
                
                
            elif model_option == "Gradient Boosting Machines (GBMs)":
                model = GradientBoostingClassifier()
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                accuracy = accuracy_score(y_test, predictions)
                y_scores = model.predict_proba(X_test)[:, 1]
                with st.expander("What is a gradient boosting machine?"):
                    st.write("""Gradient Boosting Machines, like Random Forests, are a type of machine learning model that is good at making predictions based on multiple input variables. These variables can be both categorical (like patient sex: male or female) and numerical (like age, heart rate, etc.). 

Again, suppose we're trying to predict a binary outcome: will this patient develop diabetes in the next five years or not?

A GBM also uses decision trees as its building blocks, but there's a crucial difference in how GBM combines these trees compared to Random Forests. Rather than having each tree independently make a prediction and then voting on the final outcome, GBMs build trees in sequence where each new tree is trying to correct the mistakes of the combined existing trees.

Here's a simplified breakdown of how it works:

1. **Building the First Tree**: A single decision tree is built to predict the outcome based on the input variables. However, this tree is usually very simple and doesn't do a great job at making accurate predictions.

2. **Building Subsequent Trees**: New trees are added to the model. Each new tree is constructed to correct the errors made by the existing set of trees. It does this by predicting the 'residual errors' of the previous ensemble of trees. In other words, it tries to predict how much the current model is 'off' for each patient.

3. **Combining the Trees**: The predictions from all trees are added together to make the final prediction. Each tree's contribution is 'weighted', so trees that do a better job at correcting errors have a bigger say in the final prediction.

GBMs are a very powerful method and often perform exceptionally well. Like Random Forests, they can handle complex data with many variables. But they also have a few additional strengths:

- GBMs can capture more complex patterns than Random Forests because they build trees sequentially, each learning from the last.

- GBMs can also give an estimate of the importance of each variable in making predictions, which can be very useful in understanding what's driving your predictions.

However, GBMs do have their challenges:

- They can be prone to overfitting if not properly tuned. Overfitting happens when your model is too complex and starts to capture noise in your data rather than the true underlying patterns.

- They can also be more computationally intensive than other methods, meaning they might take longer to train, especially with larger datasets.

Just like with any model, it's crucial to validate the model's predictions with your medical knowledge and consider the context. It's also important to remember that while GBMs can make very accurate predictions, they don't prove causation. They can identify relationships and patterns in your data, but they can't tell you why those patterns exist.""")
                    
                display_metrics(y_test, predictions, y_scores)

            elif model_option == "Support Vector Machines (SVMs)":
                model = svm.SVC(probability=True)
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                accuracy = accuracy_score(y_test, predictions)
                y_scores = model.predict_proba(X_test)[:, 1]
                with st.expander("What is a support vector machine?"):
                    st.write("""
Support Vector Machines are a type of machine learning model that can be used for both regression and classification tasks. They can handle both numerical and categorical input variables. In the context of predicting a binary outcome in medical data - let's stick with the example of predicting whether a patient will develop diabetes or not in the next five years - an SVM is a classification tool.

Here's a simplified explanation:

1. **Building the Model**: The SVM algorithm tries to find a hyperplane, or a boundary, that best separates the different classes (in our case, 'will develop diabetes' and 'will not develop diabetes'). This boundary is chosen to be the one that maximizes the distance between the closest points (the "support vectors") in each class, which is why it's called a "Support Vector Machine".

2. **Making Predictions**: Once this boundary is established, new patients can be classified by where they fall in relation to this boundary. If a new patient's data places them on the 'will develop diabetes' side of the boundary, the SVM predicts they will develop diabetes.

Here are some strengths and challenges of SVMs:

Strengths:
- SVMs can model non-linear decision boundaries, and there are many kernels to choose from. This can make them more flexible in capturing complex patterns in the data compared to some other methods.
- They are also fairly robust against overfitting, especially in high-dimensional space.

Challenges:
- However, SVMs are not very easy to interpret compared to models like decision trees or logistic regression. The boundaries they produce can be complex and not easily explainable in terms of the input variables.
- SVMs can be inefficient to train with very large datasets, and they require careful preprocessing of the data and tuning of the parameters.

As with any machine learning model, while an SVM can make predictions about patient health, it's crucial to validate these predictions with medical expertise. Furthermore, an SVM can identify relationships in data, but it doesn't explain why these relationships exist. As always, correlation doesn't imply causation.""")
                display_metrics(y_test, predictions, y_scores)

                


        