import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import yfinance as yf

# Web scraping of NIFTY 50 data
#
@st.cache
def load_data():
    url = 'https://en.wikipedia.org/wiki/NIFTY_50'
    html = pd.read_html(url, header = 0)
    df = html[1]
    return df


# Download NIFTY50 data
#  https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/180
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="NIFTY50.csv">Download CSV File</a>'
    return href



def app():
  #theming
  base="dark"
  primaryColor="purple"
  font="serif"

  st.title('NIFTY 50 App')

  st.markdown("""
This app retrieves the list of the **NIFTY 50** (from Wikipedia) and its corresponding!
* **Python libraries:** base64, pandas, streamlit, numpy, matplotlib, seaborn
* **Data source:** [Wikipedia](https://en.wikipedia.org/wiki/NIFTY_50).
""")

  st.sidebar.header('User Input Features')

  
  df = load_data()
  sector = df.groupby('Sector')

# Sidebar - Sector selection
  sorted_sector_unique = sorted( df['Sector'].unique() )
  selected_sector = st.sidebar.multiselect('Sector', sorted_sector_unique, sorted_sector_unique)

# Filtering data
  df_selected_sector = df[ (df['Sector'].isin(selected_sector)) ]

  st.header('Display Companies in Selected Sector')
  st.write('Data Dimension: ' + str(df_selected_sector.shape[0]) + ' rows and ' + str(df_selected_sector.shape[1]) + ' columns.')
  st.dataframe(df_selected_sector)

  st.markdown(filedownload(df_selected_sector), unsafe_allow_html=True)

# https://pypi.org/project/yfinance/

  data = yf.download(
        tickers = list(df_selected_sector[:10].Symbol),
        period = "ytd",
        interval = "1d",
        group_by = 'ticker',
        auto_adjust = True,
        prepost = True,
        threads = True,
        proxy = None
    )


