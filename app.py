import streamlit as st
from pandas_profiling import profile_report
from streamlit_pandas_profiling import st_profile_report
import numpy as np
import pandas as pd
import os
from pycaret.classification import save_model,compare_models,setup,pull
if os.path.exists("data.csv"):
    df=pd.read_csv("data.csv",index_col=None)

with st.sidebar:
    st.title("DemoApp")
    choices=st.radio("navigation",["upload","Profile report","ML","Download"])
if choices=="upload":
    st.title("Upload your Data")
    file=st.file_uploader("Data")
    if file:
        df=pd.read_csv(file,index_col=None)
        df.to_csv("data.csv",index=None)
        st.dataframe(df)
elif choices=="Profile report":
    profile_report=df.profile_report()
    st_profile_report(profile_report)
    
        
elif choices=="ML":
    target=st.selectbox("Select your choice",df.columns)
    if st.button("Train Model"):
        
        setup(df,target=target,silent=True)
        setup_df=pull()
        st.dataframe(setup_df)
        best_model=compare_models()
        compare_df=pull()
        st.dataframe(compare_df)
        best_model
        save_model(best_model,"best_model")

elif choices=="Download":
    with open("best_model.pkl","rb") as f:
        st.download_button("Download File",f,"trained_model.pkl")