import streamlit as st
import plotly.express as px
import pandas as pd
import time
import requests

st.markdown('''# Let's construct a money-making guaranteed 100% becoming rich in 1 week portfolio''')

options = st.multiselect(
    "Which asset classes would you like to include in your portfolio",
    ["Equities", "Bonds", "Real Estate", "Bitcoin"],
    ["Equities", "Bonds", "Real Estate", "Bitcoin"])

#st.write("You selected:", options)
st.write(" ")
st.write(" ")

# This dataframe has 244 lines, but 4 distinct values for `day`
# df = px.data.tips()
#fig = px.pie(df, values='tip', names='day')
df = pd.DataFrame.from_dict({'Asset Class': ["Equities", "Bonds", "Real Estate", "Bitcoin"], 'Percentage':[0.3,0.4,0.2,0.1]})
dicto= {'A':0.2,'B':0.8} #TODO: Replace with API call

dicto_2 = {'Stocks':dicto.keys(), 'Values':dicto.values()}

# fig = px.pie(df, values='Percentage', names='Asset Class')
fig = px.pie(dicto_2, values='Values', names='Stocks')






if st.button("Let's create that portfolio!", key=None, help=None, on_click=None, args=None, kwargs=None, type="secondary", disabled=False, use_container_width=False):
    progress_text = "Operation in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)


    # url=('https://lwhf-edxf3vliba-ew.a.run.app/predict')

    # params = {
    #     'as_of_date':'2024-05-27',
    #     'n_periods':3
    # }

    # req = requests.get(url, params)
    # res = req.json()

    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1, text=progress_text)
    time.sleep(1)
    my_bar.empty()
    st.plotly_chart(fig, use_container_width=True)
