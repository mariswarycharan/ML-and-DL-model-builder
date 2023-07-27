import streamlit as st
import pandas as pd


    
# to initial setup for web app  
 
st.set_page_config(page_title="Login")

hide_st_style = """
            <style>
            # MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            # header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

st.title("Login")
user_name = st.text_input("User name")
password =  st.text_input("Password" ,type="password")
login_button = st.button("Login")

read_data = pd.read_csv(r"C:\Users\USER\OneDrive - Kumaraguru College of Technology\Documents\vs code example\ML and DL model builder\all_data_details.csv")
read_data = read_data.astype("str")

if login_button == True:
    predicted = read_data[ (read_data["user_name"]  == str(user_name) ) & (read_data["password"] == str(password) )]
    if user_name in list(predicted["user_name"]) and password in list(predicted["password"]):
        st.success("You are login successfully!!!")
        if "my_input" not in st.session_state:
          st.session_state["my_input"] = ""
        login_state = True
        # login_state = st.session_state["my_input"] 
        st.session_state["my_input"] = login_state
    else:
        st.dataframe(read_data)
        st.warning("incorrect user name or password")
        
        
        
