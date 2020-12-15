import streamlit as st
import tpot
from tpot import TPOTClassifier
import pandas as pd
import numpy as np
import base64
import random
#xgboost ,deap , update_checker , tqdm , stopit , xgboost

def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="corrected.csv">**Download corrected csv file**</a>'
    return href

def label_encoder(df):
    from sklearn.preprocessing import LabelEncoder
    lab = LabelEncoder()
    df = lab.fit_transform(df).astype(np.float)
    return (df,lab.classes_)

def one_hot_encode(df,col):
    from sklearn.preprocessing import OneHotEncoder 
    from sklearn.compose import ColumnTransformer
    columnTransformer = ColumnTransformer([('encoder', 
                                        OneHotEncoder(), 
                                        [col])], 
                                      remainder='passthrough')
    dataset = np.array(columnTransformer.fit_transform(df))
    dataset = pd.DataFrame(dataset)
    return dataset

def tpot_object(metric):
    tpot = TPOTClassifier(generations=30,
                          population_size = 50,
                          scoring = metric,
                          disable_update_check = True,
                          verbosity = 0,
                          config_dict = 'TPOT light')
    return tpot


def std_scaler():
    from sklearn.preprocessing import StandardScaler
    std = StandardScaler()
    return std

def minmax_scaler():
    from sklearn.preprocessing import MinMaxScaler
    msc = MinMaxScaler()
    return msc

def rob_scaler():
    from sklearn.preprocessing import RobustScaler
    rob = RobustScaler()
    return rob


# st.markdown(get_table_download_link(df), unsafe_allow_html=True)


st.set_page_config(page_title='Preprocess data and get the best Sci-kit learn model' , page_icon = ':bar_chart:' , layout='wide')
st.title("Extensive data preprocessing tool and optimal sci-kit learn model chooser :bar_chart:")
st.sidebar.title("Menu")
st.markdown("This application is a Streamlit dashboard used "
            "for **preprocessing data and automating optimal sci-kit learn model choice(s) (+ code for the same!)**")
st.markdown('**Deployed and Maintained by Dipanshu Prasad - https://github.com/dipspilani**')

st.sidebar.subheader('Steps to use the tool:')
st.sidebar.info('1. Preprocess the file according to desired strategy')
st.sidebar.info('1. Use Preprocess-1 for missing values and label/one-hot encoding')
st.sidebar.info('3. Always handle missing values before label/one-hot encoding')
st.sidebar.info('4. Use Preprocess-2 for scaling/normalizing')
st.sidebar.info('5. Select "Choose Best Model and get yourself the best model based on desired strategy and code for the same!')

st.sidebar.subheader('Select Mode')
mode = st.sidebar.radio('Mode' , ('Preprocess-1' ,'Preprocess-2', 'Get Best Model and its code'))
st.sidebar.info('**Next Up:** Text Preprocessing')
if mode == "Code":
    st.balloons()
#st.header('Upload Data Here')
#data = st.file_uploader(label="Select File (.csv or .xlsx)" , type=['csv','xlsx'])

if mode=='Preprocess-2':
    st.markdown('**At each step, data is NOT modified in-place. So you have to download the file at each step**')
    st.header('Upload Data Here')
    data = st.file_uploader(label="Select File (.csv or .xlsx)" , type=['csv','xlsx' , 'data'])
    if data is not None:
        try:
            dataset = pd.read_csv(data)
            st.success('File Uploaded Sucessfully')
            x = st.checkbox('Show head of the dataset')
            if x:
                st.table(dataset.head())
        except:
            st.write('Please choose a valid file')

        if dataset.isnull().values.any():
            st.warning('This dataset contains missing values. For best results, fix it in Preprocess-1')
        #else:
        cols = ['None'] + list(dataset.columns)
        dx = st.selectbox('Select column to normalize/scale its values' , cols)
        if dx != 'None':
                try:
                    dataset[dx] = dataset[dx].astype(float)
                    sc_choice = st.selectbox('Choose scaling strategy' , ['None', 'Standard Scaler' , 'Min-Max Scaler' , 'Robust Scaler'])
                    if sc_choice=="Standard Scaler":
                        st.info('Robust Scaler handles outliers better')
                        obj = std_scaler()
                        dc = dataset[dx].values
                        dc = dc.reshape(-1,1)
                        dataset[dx] = obj.fit_transform(dc)
                        st.markdown(get_table_download_link(dataset), unsafe_allow_html=True)
                        st.dataframe(dataset.head())

                    elif sc_choice=="Min-Max Scaler":
                        st.info('Robust Scaler handles outliers better')
                        obj = minmax_scaler()
                        dc = dataset[dx].values
                        dc = dc.reshape(-1,1)
                        dataset[dx] = obj.fit_transform(dc)
                        st.markdown(get_table_download_link(dataset), unsafe_allow_html=True)
                        st.dataframe(dataset.head())

                    elif sc_choice=="Robust Scaler":
                        obj = rob_scaler()
                        dc = dataset[dx].values
                        dc = dc.reshape(-1,1)
                        dataset[dx] = obj.fit_transform(dc)
                        st.markdown(get_table_download_link(dataset), unsafe_allow_html=True)
                        st.dataframe(dataset.head())
                        
                    
                except:
                    st.warning('This column does not have numerical values')
                



if mode=='Preprocess-1':
    st.markdown('**At each step, data is NOT modified in-place. So you have to download the file at each step**')
    st.header('Upload Data Here')
    data = st.file_uploader(label="Select File (.csv or .xlsx)" , type=['csv','xlsx' , 'data'])
    if data is not None:
        try:
            dataset = pd.read_csv(data)
            st.success('File Uploaded Sucessfully')
            x = st.checkbox('Show head of the dataset')
            if x:
                st.table(dataset.head())
        except:
            st.write('Please choose a valid file')
            
        if not dataset.isnull().values.any():
            st.success('This dataset does not have any missing values')
                        
        else:
            st.warning('This dataset has missing value(s)')
            st.write('Number of missing values in each column:')
            st.dataframe(dataset.isna().sum())
        cols_to_correct = set()
        for z in dataset.columns:
            if dataset[z].dtype == object:
                cols_to_correct.add(z+" (string value)")
                
        a = list(dataset.isna().sum())
        i=0
        for z in dataset.columns:
            if a[i]!=0:
                cols_to_correct.add(z+" (missing values)")
            i+=1




        error = st.selectbox('Select the column to fix (fix missing values first if any)' , ['None'] + list(cols_to_correct))
        if "(string value)" in error:
            ind = error.index("(string value)")
            if dataset[error[:ind-1]].dtype != object:
                st.success('Column Corrected!')
            else:
                choice = st.selectbox('Select strategy' , ["None","Encode Labels" , "One Hot Encode"])
                if choice == "Encode Labels":
                    try:
                        st.warning('Not using One Hot Encoder may add an ordinal column which may cause a dummy variable trap')
                        dataset[error[:ind-1]] , classes1 = label_encoder(dataset[error[:ind-1]])
                        st.markdown(get_table_download_link(dataset), unsafe_allow_html=True)
                        st.dataframe(dataset.head())
                        st.write('**Labels:**')
                        ii=0
                        for i in classes1:
                            st.write(ii ," : " , i )
                            ii+=1
                    except:
                        st.warning('This dataset has missing values/is corrupted')
                elif choice== "One Hot Encode":
                    try:
                        dataset[error[:ind-1]] , classes1 = label_encoder(dataset[error[:ind-1]])
                        cols = list(dataset.columns)
                        ind1 = cols.index(error[:ind-1])
                        dataset = one_hot_encode(dataset,ind1)
                        st.markdown(get_table_download_link(dataset), unsafe_allow_html=True)
                        st.dataframe(dataset.head())
                    except:
                        st.warning('This dataset has missing values/is corrupted')

        elif "(missing values)" in error:
             ind = error.index("(missing values)")
             str_check = True
             try:
                 dataset[error[:ind-1]] = dataset[error[:ind-1]].astype(float)
             except:
                 str_check = False
                 st.warning('Selected Column has string values, so numerical preprocessing will not work')

             if dataset[error[:ind-1]].dtype==float:
                 choicee = st.selectbox('Select Strategy' , ["None" , "Mean Fill" , "Median Fill" , "Mode Fill" , "Random Fill(random value from same column)" , "Custom Fill"])

                 if choicee=="Mean Fill":
                     if dataset[error[:ind-1]].dtype==int:
                         dataset[error[:ind-1]].fillna(int(np.mean(dataset[error[:ind-1]])) , inplace = True)
                     else:
                         dataset[error[:ind-1]].fillna(np.round(dataset[error[:ind-1]].mean()) , inplace = True)
                     st.markdown(get_table_download_link(dataset), unsafe_allow_html=True)
                     st.table(dataset.head(8))

                 elif choicee=="Median Fill":
                     dataset[error[:ind-1]].fillna(dataset[error[:ind-1]].median() , inplace = True)
                     st.markdown(get_table_download_link(dataset), unsafe_allow_html=True)
                     st.table(dataset.head(8))

                 elif choicee=="Mode Fill":
                     dataset[error[:ind-1]].fillna(dataset[error[:ind-1]].mode() , inplace = True)
                     st.markdown(get_table_download_link(dataset), unsafe_allow_html=True)
                     st.table(dataset.head(8))

                 elif choicee=="Random Fill(random value from same column)":
                     dataset[error[:ind-1]].fillna(random.choice(np.array(dataset[error[:ind-1]])) , inplace = True)
                     st.markdown(get_table_download_link(dataset), unsafe_allow_html=True)
                     st.table(dataset.head(8))

                 elif choicee=="Custom Fill":
                     value = st.number_input(label = 'Enter Value')
                     try:
                         value = float(value)
                         dataset[error[:ind-1]].fillna(value , inplace = True)
                         st.markdown(get_table_download_link(dataset), unsafe_allow_html=True)
                         st.table(dataset.head(8))
                     except:
                         st.warning('Please enter numeric/float value only')

                         
             else:
                 choice_text = st.selectbox('Select Strategy' , ["None" , "Custom Fill" , "Mode Fill" , "Random Fill"])
                 if choice_text == "Custom Fill":
                     value = st.text_input('Enter Value' , value = "" , max_chars = 30)
                     dataset[error[:ind-1]].fillna(value , inplace = True)
                     st.markdown(get_table_download_link(dataset), unsafe_allow_html=True)
                     st.table(dataset.head(8))

                 elif choice_text == "Mode Fill":
                     #dataset[error[:ind-1]].fillna(dataset[error[:ind-1]].mode() , inplace = True)
                     nig = dataset[error[:ind-1]].mode()[0]
                     dataset[error[:ind-1]].fillna(nig , inplace = True)   
                     st.markdown(get_table_download_link(dataset), unsafe_allow_html=True)
                     st.table(dataset.head(8))

                 elif choice_text == "Random Fill":
                     
                     dataset[error[:ind-1]].fillna(random.choice(dataset[error[:ind-1]].values.tolist()) , inplace = True)
                   
                     st.markdown(get_table_download_link(dataset), unsafe_allow_html=True)
                     st.table(dataset.head(8))
                     
            

        

        





        
        #st.markdown(get_table_download_link(dataset), unsafe_allow_html=True)            



elif mode=='Get Best Model and its code':
    st.header('Upload Data Here')
    st.warning('Please preprocess the data before proceeding')
    data = st.file_uploader(label="Select File (.csv or .xlsx)" , type=['csv','xlsx','data'])
    if data is not None:
        try:
            dataset = pd.read_csv(data)
            if dataset.isnull().values.any():
                st.warning('This dataset has missing values. Please rectify them in the Preprocess step else the results may be unexpected')
            else:
                pass
        except:
            st.warning('Choose a valid file first')


        colz = list(dataset.columns)
        target = st.selectbox('Choose Target Column' , [None] + colz)
        if target is not None:
            colz.remove(target)
            features = st.multiselect('Choose Feature Column(s)' , colz)
            if features:
                X = dataset.loc[:,features]
                y = dataset.loc[:,[target]]
                scoring = ['None' , 'accuracy','average_precision', 'balanced_accuracy', 'f1', 'f1_macro', 'f1_micro', 'precision'  , 'recall' ,'roc_auc']
                metric = st.selectbox('Choose scoring metric (estimated time : 8-10 minutes)' , scoring)
                if metric != 'None':
                    try:
                        with st.spinner('Finding the best preprocessing steps and sci-kit learn algorithm... :hourglass:'):
                            tpot = tpot_object(metric)
                            tpot.fit(X,y)
                            st.info('Steps:')
                            count = 1
                            for i in tpot.fitted_pipeline_:
                                st.write(count , i)
                                count+=1
                            code = tpot.export()
                            st.success('Code for the above steps:')
                            st.code(code,language='python')
                    except:
                        st.warning('Data in unexpected format')
                        st.info('**TIP:** Preprocess all columns in numercical format' )
    
    
    
          


