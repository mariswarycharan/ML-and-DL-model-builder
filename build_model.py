# import module
import pandas as pd
import pickle
from tqdm import tqdm
from lightgbm import LGBMClassifier,LGBMRegressor
from xgboost import XGBClassifier,XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,r2_score,mean_squared_error
import streamlit as st
from catboost  import CatBoostClassifier, CatBoostRegressor
from lazypredict.Supervised import LazyClassifier,LazyRegressor
from streamlit_pandas_profiling import st_profile_report
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,ExtraTreesClassifier,ExtraTreesRegressor,AdaBoostClassifier,AdaBoostRegressor,BaggingClassifier,BaggingRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
from sklearn.svm import SVC,NuSVC,NuSVR,LinearSVC,LinearSVR,SVR
from sklearn.linear_model import SGDClassifier,SGDRegressor,LinearRegression,LogisticRegression,RidgeClassifierCV,RidgeClassifier,PassiveAggressiveClassifier,PassiveAggressiveRegressor,Perceptron
from sklearn.semi_supervised import LabelPropagation,LabelSpreading
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor,NearestCentroid
from sklearn.naive_bayes import GaussianNB,BernoulliNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier,DummyRegressor

# to initial setup for web app  


st.set_page_config(page_title="build_model")

hide_st_style = """
            <style>
            # MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            # header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

st.title("build model !!!",anchor="charan")


model_selection= {"DecisionTreeClassifier":DecisionTreeClassifier(),"AdaBoostClassifier":AdaBoostClassifier(), "LinearDiscriminantAnalysis" : LinearDiscriminantAnalysis() ,"SVC" : SVC(),
                  "SGDClassifier" : SGDClassifier() , "RandomForestClassifier" : RandomForestClassifier(), "QuadraticDiscriminantAnalysis" : QuadraticDiscriminantAnalysis() , "NuSVC" : NuSVC(), "LogisticRegression" : LogisticRegression() ,
                  "BaggingClassifier" : BaggingClassifier() , "LGBMClassifier" : LGBMClassifier() , "LabelSpreading":LabelSpreading(), "LabelPropagation":LabelPropagation(), "KNeighborsClassifier" : KNeighborsClassifier() , "GaussianNB" : GaussianNB(),
                  "ExtraTreesClassifier" : ExtraTreesClassifier() , "PassiveAggressiveClassifier" : PassiveAggressiveClassifier() , "Perceptron" : Perceptron(), "LinearSVC" : LinearSVC() , "CalibratedClassifierCV" : CalibratedClassifierCV() , "NearestCentroid" : NearestCentroid() ,
                  "BernoulliNB" : BernoulliNB() , "RidgeClassifier" : RidgeClassifier() , "RidgeClassifierCV" : RidgeClassifierCV() , "DummyClassifier" : DummyClassifier() , 'CatBoostClassifier':CatBoostClassifier()
                  }

model_selection_reg = { "DecisionTreeRegressor":DecisionTreeRegressor(),"AdaBoostRegressor":AdaBoostRegressor(),
                        "LinearRegression" : LinearRegression() , "KNeighborsRegressor" : KNeighborsRegressor(), "LGBMRegressor" : LGBMRegressor(),"DummyRegressor" : DummyRegressor(),"BaggingRegressor" : BaggingRegressor() 
                        , "LinearSVR" : LinearSVR(),"RandomForestRegressor" : RandomForestRegressor()
                        ,"BaggingRegressor" : BaggingRegressor(),"ExtraTreesRegressor" : ExtraTreesRegressor() 
                         , "PassiveAggressiveRegressor" : PassiveAggressiveRegressor(),
                         }
#      'CatBoostRegressor':CatBoostRegressor()                      
#                   }

if "my_input" in st.session_state:
    if st.session_state["my_input"] == True:
        
        st.text_input('Enter project name ')
        st.date_input('date')

        with st.sidebar:
            st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSCQ1SEUC_Rj98QxH-7fbgMMkpIJjtH2lg6TQ&usqp=CAU")
            st.title("Auto_ML")
            type_of_task = st.radio("select :",["None","build machine learning model","build deep learning model"])
            st.info("This project application helps you build and explore your data.")
            logout = st.button("Logout")
            if logout:
                st.session_state["my_input"] = False    

        # ================================ To build model in  machine learning ==================================================
            
        if type_of_task == "build machine learning model":
            load = st.file_uploader('Upload a dataset')

            problem = st.selectbox("type of problem",["none","Regression","classification"])
            if problem == "none":
                pass
            
        # ================================= To build machine learning model for classification problem ============================

            if problem == "classification":
                
                if load == None:
                    pass
                else:
                    df  = pd.read_csv(load)
                    all_information_dataset = st.radio("To see all information about the dataset",["None","See_all_information_about_dataset"])
                    
                    if all_information_dataset == "See_all_information_about_dataset":
                        if "profile_report" not in st.session_state:
                            st.session_state["profile_report"] = "profile_report"
                            pr = df.profile_report()
                            st_profile_report(pr)
                        
                    head = df.head(10)
                    st.success("Dataset : ")
                    st.dataframe(head)
                    st.success("Describe the dataset ")
                    st.write(df.describe()) 
                    st.success("Correlation of dataset :")
                    st.write(df.corr())        
                    
                    df.drop_duplicates(inplace=True)
                    df.dropna(inplace=True)
                    columns = list(df.columns)
                    
                    input_label = st.multiselect("select your input_label :",columns)
                    output_label = st.multiselect("select your output_label :",columns) 
                    print(input_label,output_label)
                    finish = st.radio("select the step ", ["build all the model","select the best model","to train the best model","download the best model"])
                    button_step_select = st.button("Submit")
                    
                    if button_step_select == True :
                        
                        if finish == "build all the model" :
                        # if len(output_label) >= 1:
                            with st.spinner("Just a moment ..."):
                                input_data = df.loc[:,input_label]
                                output_data = df[output_label]


                                for label in input_data.columns:
                                    label_type = str(type(input_data[label][3])).split("'")[1]
                                    if label_type == "numpy.float64" or label_type == "numpy.int64":
                                        pass
                                    else:
                                        input_data[label].replace(list(input_data[label].unique()),[ i for i in range(len(list(input_data[label].unique())))],inplace=True)

                                # standard_data = StandardScaler().fit_transform(input_data)
                                
                                X_train, X_test, Y_train, Y_test = train_test_split(input_data,output_data,random_state=0, train_size = .80)
                                
                                # to see accuracy of all models
                                lazy_model = LazyClassifier(random_state=123,verbose=0)
                                models,predition =lazy_model.fit(X_train, X_test, Y_train, Y_test)
                                
                                # to see accuracy of boosting algorithm
                                model_cat = CatBoostClassifier(iterations=1000,learning_rate=0.01,loss_function='MultiClass',eval_metric='Accuracy',use_best_model=True,custom_loss=['Accuracy'], random_seed=42).fit(X_train,Y_train,eval_set=(X_test,Y_test))
                                pred_cat = model_cat.predict(X_test)
                        
                                
                                boost_df = pd.DataFrame(data = {'Accuracy':[accuracy_score(Y_test,pred_cat)],
                                                                'Balanced Accuracy':[accuracy_score(Y_test,pred_cat)],
                                                                'ROC AUC':["none"],
                                                                'Time Taken':["none"]},
                                                                index = ["CatBoostClassifier"] )
                                models = pd.concat([models,boost_df])
                                
                    
                                all_models_list = str(models.iloc[:,:0]).split("[")[2].split("]")[0].split(",")
                                all_models_list.insert(0,"None")
                                all_models_list = [ i.replace(" ","") for i in all_models_list ]
                                pickle.dump(all_models_list,open("all_models_list.pickle","wb"))
                                pickle.dump(models,open("all_models_trained.pickle","wb"))
                                pickle.dump(X_train,open("X_train.pickle","wb"))
                                pickle.dump(Y_train,open("Y_train.pickle","wb"))
                                pickle.dump(X_test,open("X_test.pickle","wb"))
                                pickle.dump(Y_test,open("Y_test.pickle","wb"))
                            st.success("all model is trained successfully !!!")
                            
                        if finish == "select the best model" :
                            
                            with st.spinner("Just a moment ..."):
                                models = pickle.load(open("all_models_trained.pickle","rb"))
                                st.dataframe(models)
                                all_models_list = pickle.load(open("all_models_list.pickle","rb"))
                                st.line_chart(data=models.iloc[:,:1])
                            
                        if finish == "to train the best model":
                            # it is dataframe of all models 
                            best_model_name = pickle.load(open("all_models_trained.pickle","rb"))
                            
                            best_model_name = list(best_model_name.index)[list(best_model_name["Accuracy"]).index(max(list(best_model_name["Accuracy"])))]
                    
                            X_train = pickle.load(open("X_train.pickle","rb"))
                            Y_train = pickle.load(open("Y_train.pickle","rb"))
                            X_test = pickle.load(open("X_test.pickle","rb"))
                            Y_test = pickle.load(open("Y_test.pickle","rb"))   
                                                     
                            best_model = model_selection[str(best_model_name)].fit(X_train,Y_train)
                            pred_best_model = best_model.predict(X_test)
                            all_paras_dict =  dict(best_model.get_params())
                            DataFrame_params =  pd.DataFrame(data=list(all_paras_dict.values()),index=list(all_paras_dict.keys()),columns=["params"])
                            st.subheader("Final result")
                            st.dataframe(DataFrame_params)
                            
                            save_model = pickle.dump(best_model,open("saving_model.pickle","wb"))
                            st.success("accuracy : "  + str(accuracy_score(Y_test,pred_best_model)*100))
                            st.success(" model is trained successfully !!!")
                            
                        if finish == "download the best model" :
                            with open('saving_model.pickle', 'rb') as fil:
                                st.download_button("download model",fil,file_name = 'saving_model.pickle')
                                st.snow()
                            st.success("best model is downloaded successfully !!!")
                # all_model_dataframe = pickle.load(open("all_models_trained.pickle","rb"))
                # particular_model =  st.selectbox("select the particular model and train it",list(all_model_dataframe.index))
                
                
                # X_train = pickle.load(open("X_train.pickle","rb"))
                # Y_train = pickle.load(open("Y_train.pickle","rb"))
                # X_test = pickle.load(open("X_test.pickle","rb"))
                # Y_test = pickle.load(open("Y_test.pickle","rb"))   
                                            
                # best_model = model_selection[str(best_model_name)].fit(X_train,Y_train)
                # pred_best_model = best_model.predict(X_test)
                # all_paras_dict =  dict(best_model.get_params())
                # DataFrame_params =  pd.DataFrame(data=list(all_paras_dict.values()),index=list(all_paras_dict.keys()),columns=["params"])
                # st.subheader("Final result")
                # st.dataframe(DataFrame_params)
                
                # save_model = pickle.dump(best_model,open("saving_model.pickle","wb"))
                # st.success("accuracy : "  + str(accuracy_score(Y_test,pred_best_model)*100))
                # st.success(" model is trained successfully !!!")
                
        # ================================= To build machine learning model for Regression problem ============================
                

            if problem == "Regression":
                if load == None:
                    pass
                else:
                    df  = pd.read_csv(load)
                    all_information_dataset = st.radio("To see all information about the dataset",["None","See_all_information_about_dataset"])
                    if all_information_dataset == "See_all_information_about_dataset":
                        pr = df.profile_report()
                        st_profile_report(df)
                    head = df.head(10)
                    st.success("Dataset : ")
                    st.dataframe(head)
                    st.success("Describe the dataset ")
                    st.write(df.describe()) 
                    st.success("Correlation of dataset :")
                    st.write(df.corr())        
                    
                    df.drop_duplicates(inplace=True)
                    df.dropna(inplace=True)
                    columns = list(df.columns)
                    
                    input_label = st.multiselect("select your input_label :",columns)
                    output_label = st.multiselect("select your output_label :",columns) 
                    print(input_label,output_label)
                    finish = st.radio("select the step ", ["build all the model","select the best model","to train the best model","download the best model"])
                    button_step_select = st.button("Submit")
                    
                    if button_step_select == True :
                        if finish == "build all the model" :
                            input_data = df.loc[:,input_label]
                            output_data = df[output_label]


                            for label in input_data.columns:
                                label_type = str(type(input_data[label][3])).split("'")[1]
                                if label_type == "numpy.float64" or label_type == "numpy.int64":
                                    pass
                                else:
                                    input_data[label].replace(list(input_data[label].unique()),[ i for i in range(len(list(input_data[label].unique())))],inplace=True)

                            standard_data = StandardScaler().fit_transform(input_data)
                            
                            X_train, X_test, Y_train, Y_test = train_test_split(input_data,output_data,random_state=0, train_size = .80)
                            
                            # to see accuracy of all models
                            
                            r2_score_list_reg = []
                            mse_list = []
                            for name,models in tqdm(model_selection_reg.items()):
                                check_mod = models.fit(X_train,Y_train)
                                pred = check_mod.predict(X_test)
                                r2_score_reg = r2_score(Y_test,pred)
                                mse_value = mean_squared_error(Y_test , pred)
                                r2_score_list_reg.append(float(r2_score_reg))
                                mse_list.append(mse_value)
                                
                            boost_df = pd.DataFrame(data = {'r2_score':r2_score_list_reg,
                                                            'Mean Square Error':mse_list
                                                            },
                                                            index = list(model_selection_reg.keys()) )
                            
                            models = boost_df

                            all_models_list = str(models.iloc[:,:0]).split("[")[2].split("]")[0].split(",")
                            all_models_list.insert(0,"None")
                            all_models_list = [ i.replace(" ","") for i in all_models_list ]
                            pickle.dump(all_models_list,open("all_models_list.pickle","wb"))
                            pickle.dump(models,open("all_models_trained.pickle","wb"))
                            pickle.dump(X_train,open("X_train.pickle","wb"))
                            pickle.dump(Y_train,open("Y_train.pickle","wb"))
                            pickle.dump(X_test,open("X_test.pickle","wb"))
                            pickle.dump(Y_test,open("Y_test.pickle","wb"))
                            st.success("all model is trained successfully !!!")
                            
                        if finish == "select the best model" :
                            models = pickle.load(open("all_models_trained.pickle","rb"))
                            st.dataframe(models)
                            all_models_list = pickle.load(open("all_models_list.pickle","rb"))
                            best_model_name = st.selectbox("Select the best model",all_models_list)
                          
                            st.line_chart(data=models.iloc[:,:1])
                            st.write("Result of all the models")
                            pickle.dump(best_model_name,open("best_model_name.pickle","wb"))
        
                        if finish == "to train the best model":
        
                            best_model_name = pickle.load(open("all_models_trained.pickle","rb"))
                            
                            best_model_name = list(best_model_name.index)[list(best_model_name["r2_score"]).index(max(list(best_model_name["r2_score"])))]
                            
                            X_train = pickle.load(open("X_train.pickle","rb"))
                            Y_train = pickle.load(open("Y_train.pickle","rb"))
                            X_test = pickle.load(open("X_test.pickle","rb"))
                            Y_test = pickle.load(open("Y_test.pickle","rb"))   
                                                     
                            best_model = model_selection_reg[str(best_model_name)].fit(X_train,Y_train)
                            pred_best_model = best_model.predict(X_test)
                            all_paras_dict =  dict(best_model.get_params())
                            DataFrame_params =  pd.DataFrame(data=list(all_paras_dict.values()),index=list(all_paras_dict.keys()),columns=["params"])
                            st.subheader("Final result")
                            st.dataframe(DataFrame_params)
                            
                            save_model = pickle.dump(best_model,open("saving_model.pickle","wb"))
                            st.success("accuracy : "  + str(r2_score(Y_test,pred_best_model)*100))
                            st.success(" model is trained successfully !!!")
                            
                        if finish == "download the best model" :
                            with open('saving_model.pickle', 'rb') as fil:
                                st.download_button("download model",fil,file_name = 'saving_model.pickle')
                                st.snow()
                            st.success("best model is downloaded successfully !!!")
