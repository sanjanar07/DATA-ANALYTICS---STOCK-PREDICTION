import streamlit as st
from multiapp import MultiApp
from apps import nifty50,sensex,stockapp # import your app modules here

app = MultiApp()
st.title("MAIN PAGE")

st.image('image.png')
# Add all your application here

app.add_app("TREND", stockapp.app)
app.add_app("NIFTY50", nifty50.app)
app.add_app("SENSEX", sensex.app)

# The main app
app.run()