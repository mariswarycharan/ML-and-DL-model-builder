import streamlit as st
import pandas as pd

# to initial setup for web app  
 
st.set_page_config(page_title="Signup")

hide_st_style = """
            <style>
            # MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            # header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

st.title("Signup")
name = st.text_input("Name")
user_name =  st.text_input("Username")
password = st.text_input("Password",type="password")
Signup_button = st.button("Signup")

read_data = pd.read_csv(r"C:\Users\USER\OneDrive - Kumaraguru College of Technology\Documents\vs code example\ML and DL model builder\all_data_details.csv")
collect_dict = {"name":[name],"user_name":[user_name],"password":[password]}
data = pd.DataFrame(data=collect_dict)

if Signup_button == True:
    if name != "" and user_name != ""  and  password != "":
        if user_name in  list(read_data["user_name"]):
            st.warning("User name already exist...")
        if user_name not in  list(read_data["user_name"]): 
            add_data = pd.concat([read_data,data])
            add_data.to_csv("all_data_details.csv",index=False)
            st.success("regitered successfully!!!")
    else:
        st.warning("Enter all data")

    
    
    