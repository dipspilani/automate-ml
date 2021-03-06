import streamlit as st
import tpot
from tpot import TPOTClassifier
import pandas as pd
import numpy as np
import base64
import PIL
from PIL import Image
from io import BytesIO,StringIO
import random
import skimage
import nltk
import textblob
import seaborn as sns
from textblob import TextBlob
import re
import plotly
import plotly.express as px
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')

#xgboost ,deap , update_checker , tqdm , stopit , xgboost
@st.cache(allow_output_mutation=True)
def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="corrected.csv">**Download corrected csv file**</a>'
    return href

@st.cache(allow_output_mutation=True)
def label_encoder(df):
    from sklearn.preprocessing import LabelEncoder
    lab = LabelEncoder()
    df = lab.fit_transform(df).astype(np.float)
    return (df,lab.classes_)

@st.cache(allow_output_mutation=True)
def get_image_download_link(img):
	"""Generates a link allowing the PIL image to be downloaded
	in:  PIL image
	out: href string
	"""
	buffered = BytesIO()
	img.save(buffered, format="JPEG")
	img_str = base64.b64encode(buffered.getvalue()).decode()
	href = f'<a href="data:file/jpg;base64,{img_str}" download="processed_image.jpg">Download result</a>'
	return href

@st.cache(allow_output_mutation=True)
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

@st.cache(allow_output_mutation=True)
def tpot_object(metric):
    tpot = TPOTClassifier(generations=30,
                          population_size = 50,
                          scoring = metric,
                          disable_update_check = True,
                          verbosity = 0,
			  early_stop = 4,
                          config_dict = 'TPOT light')
    return tpot

@st.cache(allow_output_mutation=True)
def std_scaler():
    from sklearn.preprocessing import StandardScaler
    std = StandardScaler()
    return std

@st.cache(allow_output_mutation=True)
def minmax_scaler():
    from sklearn.preprocessing import MinMaxScaler
    msc = MinMaxScaler()
    return msc

@st.cache(allow_output_mutation=True)
def rob_scaler():
    from sklearn.preprocessing import RobustScaler
    rob = RobustScaler()
    return rob


# st.markdown(get_table_download_link(df), unsafe_allow_html=True)


st.set_page_config(page_title='Preprocess data and get the best Sci-kit learn model' , page_icon = ':bar_chart:' , layout='wide', initial_sidebar_state='expanded')
st.title("Extensive data/image/text preprocessing-visualization tool and optimal sci-kit learn pipeline chooser :bar_chart:")
st.sidebar.title("Menu")
st.markdown("This application is a Streamlit dashboard used "
            "for **preprocessing and visualizing data and automating optimal sci-kit learn model choice(s) (+ code for the same!)**")
st.markdown('**Deployed and Maintained by Dipanshu Prasad - https://github.com/dipspilani**')



st.sidebar.subheader('Select Mode')
mode = st.sidebar.radio('Mode' , ('Preprocess-1' ,'Preprocess-2','Preprocess Image','Preprocess Text', 'Get Best Model and its code','Dimensionality Reduction, Visualization and Clusterization'))
st.sidebar.subheader('Steps to use the tool:')

st.sidebar.info('1. Use Preprocess-1 for missing values and label/one-hot encoding')
st.sidebar.info('2. Always handle missing values before label/one-hot encoding')
st.sidebar.info('3. Use Preprocess-2 for scaling/normalizing')
st.sidebar.info('4. Select "Choose Best Model and get yourself the best model based on desired strategy and code for the same!')
fb = st.sidebar.text_area('Your feedback or suggestions or maybe if you just want to wish me a good day' , key='xx')
submit = st.sidebar.button('Send')
if submit and fb:
	f = open('user_comments1.txt' , 'w')
	f.write(fb)
	xx = ""

	
#st.header('Upload Data Here')
#data = st.file_uploader(label="Select File (.csv or .xlsx)" , type=['csv','xlsx'])


if mode=='Dimensionality Reduction, Visualization and Clusterization':
	st.info('Fix any missing values to avoid erorrs')
	st.header('Upload data here')
	data = st.file_uploader(label="Select File (.csv or .xlsx)" , type=['csv','xlsx' , 'data'])
	if data is not None:
		try:
			dataset = pd.read_csv(data)
			st.success('Data Uploaded Sucessfully')
			x = st.checkbox('Show head of dataset')
			if x:
				st.table(dataset.head())
			choice = st.selectbox('Choose Operation',['None','PairPlot','HeatMap','3-D Visualization','2-D visualization','1-D visualization','PCA','NMF','t-SNE','Clusterize data'])
			if choice=='HeatMap':
				try:
					hm = dataset.corr()
					st.dataframe(hm)
					st.plotly_chart(px.imshow(hm,width=500))
				except:
					st.error('Something went wrong')
			if choice=='Clusterize data':
				st.info('For visualizing clusterization, it is suggested that dimensions are reduced to 2-3. Else, you can also visualize by selection of 2-3 columns from the lot')
				cols = list(dataset.columns)
				for i in cols:
					try:
						dataset[i] = dataset[i].astype(float)
					except:
						dataset.drop(columns = [i],inplace=True)
				cho = st.selectbox('Choose clusterizing strategy',['K-Means','Agglomerative','DBSCAN'])
				if cho=='DBSCAN':
					from sklearn.cluster import DBSCAN
					clus = st.number_input('Enter value of eps(suggested: 0.5)',value=0,min_value=0)
					clus1 = st.number_input('Enter value of min_samples(suggested: 5)',value=0,min_value=0)
					if clus!=0 and clus1>=0:
						clus1 = int(clus1)
						try:
							db = DBSCAN(eps=clus,min_samples=clus1)
							y= db.fit_predict(dataset)
							y = pd.DataFrame(y)
							dataset['cluster']=y
							st.info('Download cluster labels column here. You can merge it with PCA/NMF/t-SNE results')
							st.markdown(get_table_download_link(y), unsafe_allow_html=True)
							if len(dataset.columns)==3:
								st.plotly_chart(px.scatter(dataset,x=cols[0],y=cols[1],color='cluster'))
							else:
								cols = list(dataset.columns)
								cols.remove('cluster')
								c1 = st.selectbox('Choose X-axis for Kmeans visualization',cols)
								cols.remove(c1)
								c2 = st.selectbox('Choose Y-axis for Kmeans visualization',cols)
								cols.remove(c2)
								ck = st.checkbox('Show 3-D visualization')
								if ck:
									c3 = st.selectbox('Choose Z-axis for Kmeans visualization',cols)
									st.plotly_chart(px.scatter_3d(dataset,x=c1,y=c2,z=c3,color='cluster'))
								else:	
									st.plotly_chart(px.scatter(dataset,x=c1,y=c2,color='cluster'))
						except:
							st.error('Something went wrong')	
							
				
				
				
				if cho=='Agglomerative':
					from sklearn.cluster import AgglomerativeClustering
					clus = st.number_input('Enter number of clusters',value=0,min_value=0,max_value=len(dataset))
					if clus!=0:
						try:
							km = AgglomerativeClustering(n_clusters=clus)
							y = km.fit_predict(dataset)
							y = pd.DataFrame(y)
							dataset['cluster']=y
							st.info('Download cluster labels column here. You can merge it with PCA/NMF/t-SNE results')
							st.markdown(get_table_download_link(y), unsafe_allow_html=True)
							if len(dataset.columns)==3:
								st.plotly_chart(px.scatter(dataset,x=cols[0],y=cols[1],color='cluster'))
							else:
								cols = list(dataset.columns)
								cols.remove('cluster')
								c1 = st.selectbox('Choose X-axis for Kmeans visualization',cols)
								cols.remove(c1)
								c2 = st.selectbox('Choose Y-axis for Kmeans visualization',cols)
								cols.remove(c2)
								ck = st.checkbox('Show 3-D visualization')
								if ck:
									c3 = st.selectbox('Choose Z-axis for Kmeans visualization',cols)
									st.plotly_chart(px.scatter_3d(dataset,x=c1,y=c2,z=c3,color='cluster'))
								else:	
									st.plotly_chart(px.scatter(dataset,x=c1,y=c2,color='cluster'))
						except:
							st.error('Something went wrong')	
				
				
				if cho=='K-Means':
					from sklearn.cluster import KMeans
					clus = st.number_input('Enter number of clusters',value=0,min_value=0,max_value=len(dataset))
					if clus!=0:
						try:
							km = KMeans(n_clusters=clus)
							y = km.fit_predict(dataset)
							
							y = pd.DataFrame(y)
							dataset['cluster']=y
							st.info('Download cluster labels column here, You can merge it with PCA/NMF/t-SNE results')
							st.markdown(get_table_download_link(y), unsafe_allow_html=True)
							if len(dataset.columns)==3:
								st.plotly_chart(px.scatter(dataset,x=cols[0],y=cols[1],color='cluster'))
							else:
								cols = list(dataset.columns)
								cols.remove('cluster')
								c1 = st.selectbox('Choose X-axis for Kmeans visualization',cols)
								cols.remove(c1)
								c2 = st.selectbox('Choose Y-axis for Kmeans visualization',cols)
								cols.remove(c2)
								ck = st.checkbox('Show 3-D visualization')
								if ck:
									c3 = st.selectbox('Choose Z-axis for Kmeans visualization',cols)
									st.plotly_chart(px.scatter_3d(dataset,x=c1,y=c2,z=c3,color='cluster'))
								else:	
									st.plotly_chart(px.scatter(dataset,x=c1,y=c2,color='cluster'))
						except:
							st.error('Something went wrong')
				
			
			
			
			if choice=='t-SNE':
				st.info('All non-numeric columns will automatically be dropped')
				cols = list(dataset.columns)
				for i in cols:
					try:
						dataset[i] = dataset[i].astype(float)
					except:
						dataset.drop(columns = [i],inplace=True)
						
				from sklearn.manifold import TSNE
				ts = TSNE()
			
				try:
					with st.spinner('Finding t-SNE components... :hourglass:'):
						x = ts.fit_transform(dataset)
						x = pd.DataFrame(x)
						st.markdown(get_table_download_link(x), unsafe_allow_html=True)
						st.info('t-SNE values')
						st.dataframe(x.head())
				except:
					st.error('Something went wrong!!')
			
			
			
			if choice=='NMF':
				st.info('NMF requires that all values be non-negative. All non-numeric columns will automatically be dropped')
				cols = list(dataset.columns)
				for i in cols:
					try:
						dataset[i] = dataset[i].astype(float)
					except:
						dataset.drop(columns = [i],inplace=True)
				from sklearn.decomposition import NMF
				inp = st.number_input('Enter the number of components to use' , value = 0 , min_value = 0,max_value = len(cols))
				if inp!=0:
					nmf = NMF(n_components = inp)
					try:
						with st.spinner('Finding the NMF components... :hourglass:'):
							x = nmf.fit_transform(dataset)
							x = pd.DataFrame(x)
							st.markdown(get_table_download_link(x), unsafe_allow_html=True)
							st.info('NMF values')
							st.dataframe(x.head())
						
					except:
						st.error('There are negative values in the dataset')
			
			if choice=='PCA':
				st.info('PCA requires that all values be in numeric form. Hence, all non-numeric columns are dropped here')
				cols = list(dataset.columns)
				for i in cols:
					try:
						dataset[i] = dataset[i].astype(float)
					except:
						dataset.drop(columns = [i],inplace=True)
				from sklearn.decomposition import PCA
				pca = PCA()
				
				x = pca.fit_transform(dataset)
				comps = pca.explained_variance_ratio_
				k=1
				for i in comps:
					st.write('% Variance explained by Principal Component '+str(k) , np.round(100*i,5))
					k+=1
				inp = st.number_input('Enter the number of components to use' , value = 0 , min_value = 0,max_value = k-1)
				if inp!=0:
					chk = st.checkbox('Scale PC values')
					if chk:
						pca = PCA(n_components = inp,whiten = True)
						x = pca.fit_transform(dataset)
						x = pd.DataFrame(x)
						st.markdown(get_table_download_link(x), unsafe_allow_html=True)
						st.info('Scaled Principal Component values')
						st.dataframe(x.head())
					else:
						pca = PCA(n_components = inp)
						x = pca.fit_transform(dataset)
						x = pd.DataFrame(x)
						st.markdown(get_table_download_link(x), unsafe_allow_html=True)
						st.info('Principal Component values')
						st.dataframe(x.head())
					
			
			if choice=='PairPlot':
				try:
					with st.spinner('Creating pairplot... :hourglass:'):
						st.pyplot(sns.pairplot(dataset,kind='reg'))
				except:
					st.error('Something went wrong')
			if choice=="1-D visualization":
				st.info('Choose a numeric column only')
				cols = list(dataset.columns)
				cx = st.selectbox('Select column' , cols)
				if cx:
					try:
						dataset[cx] = dataset[cx].astype(float)
						#st.pyplot(sns.distplot(dataset[cx]))
						ww = st.number_input(label ='Enter number of bins',value = 10,max_value = 100)
						st.plotly_chart(px.histogram(dataset,x=cx,nbins=ww))
					except:
						st.warning('Non-numeric column')
						st.plotly_chart(px.scatter(dataset,x=cx))
			if choice=='3-D Visualization':
				st.info('Selection of more than 2 non-numeric columns may lead to errors')
				try:	
					cols = list(dataset.columns)
					choice1 = st.selectbox('Choose X-axis',cols)
					if choice1:
							cols.remove(choice1)
							try:
								dataset[choice1] = dataset[choice1].astype(float)
							except:
								st.warning('This is a non-numeric column')
							choice2 = st.selectbox('Choose Y-axis',cols)
							if choice2:
								cols.remove(choice2)
								try:
									dataset[choice2] = dataset[choice2].astype(float)
								except:
									st.warning('This is a non-numeric column')
								choice3 = st.selectbox('Choose Z-axis',cols)
								if choice3:
									try:
										dataset[choice3] = dataset[choice3].astype(float)
									except:
										st.warning('This is a non-numeric column')
									
									st.plotly_chart(px.scatter_3d(dataset,x=choice1,y=choice2,z=choice3,width=1000))		
				
				except:
					st.error('Something went wrong!')						
					
									
							
							
				
				
			if choice=='2-D visualization':
				st.info('Choose X-axis as numerical/categorical, Y-axis must be numerical')
				try:
					cols = list(dataset.columns)
					choice1 = st.selectbox('Choose X-axis',cols)
					if choice1:
						cols.remove(choice1)
						try:
							#dataset[choice1] = dataset[choice1].astype(float)
							choice2 = st.selectbox('Choose Y-axis',cols)
							if choice2:
								cols.remove(choice2)
								try:
									dataset[choice2]=dataset[choice2].astype(float)
									huec = st.checkbox('Add hue')
									if huec:
										hue = st.selectbox('Choose hue',cols)
										if hue:
											try:
												st.plotly_chart(px.scatter(dataset,x=choice1,y=choice2,color=hue))
											except:
												st.warning('Something went wrong, hue not used')
												st.plotly_chart(px.scatter(dataset,x=choice1,y=choice2))
									else:
										
										st.pyplot(sns.jointplot(x=dataset[choice1],y=dataset[choice2],kind='reg'))
								except:
									st.warning('This column does not have numeric values')
									st.plotly_chart(px.scatter(dataset,x=choice1,y=choice2))
						except:
							st.warning('This column does not have numeric values')
				except:
					st.warning('Something went wrong!')
		except:
			st.error('Please choose a valid file!')



if mode=="Preprocess Text":
	num_words = st.slider(label = 'Select the number of words(roughly)' , min_value=0 , max_value = 1000 , value = 50, step = 50)
	st.header('Enter Text Here')
	if num_words:
		text = st.text_area(label = "Text Box" , height = min(200,num_words))
		butt = st.button(label = 'Send for processing')
		if butt:
			from nltk.stem.porter import PorterStemmer
			from nltk.stem import WordNetLemmatizer
			from nltk.tokenize import word_tokenize 
			from nltk import pos_tag, ne_chunk 
			text_lower = text.lower()
			text_no_punc = re.sub(r'[^\w\s]', '', text_lower)
			st.info('Text with punctuations removed')
			st.write(text_no_punc)
			tokens = nltk.word_tokenize(text_no_punc)
			st.info('Tokens')
			st.write(tokens)
			tagged = nltk.pos_tag(tokens)
			st.info('Part of Speech Tags')
			st.write(tagged)
			stop_words = stopwords.words('english')
			filtered_words = [word for word in tokens if word not in stop_words]
			st.info('Stop Words removed')
			st.write(" ".join(filtered_words))
			porter = PorterStemmer()
			stemmed = [porter.stem(word) for word in filtered_words]
			st.info('Stemmed words')
			st.write(" ".join(stemmed))
			lemmatizer=WordNetLemmatizer()
			lemm = [lemmatizer.lemmatize(word) for word in filtered_words]
			st.info('Lemmatized words')
			st.write(" ".join(lemm))
			st.info('Named entity recognition')
			st.write(ne_chunk(tagged))
			
			
			
				
	


if mode=="Preprocess Image":
    st.header('Upload Image Here')
    data = st.file_uploader(label="Select File (.jpg or .gif or .png)" , type=['jpg','png' , 'gif'])
    if data is not None:
        try:
            img = Image.open(data)
            chk = st.checkbox('Display Image')
            st.write("**Image dimension:**")
            st.info(img.size)
            if chk:
                try:
                    st.image(img,caption = "Uploaded Image" , width = 400)
                except:
                    st.warning('Something wrong. Can not display Image')
            opn = st.selectbox('Choose Operation' , ['None','Binarize','Grayscale','BoxBlur','GaussianBlur','Kernel (convolution kernel)',
                                'RankFilter','AutoContrast','Colorize B-W image','Padding','Equalize',
                                'Posterize','Edge Enhance' , 'Contour','Stretch/Shrink'])
            if opn=="Binarize":
                try:
                    bina = img.convert("1")
                    st.markdown(get_image_download_link(bina), unsafe_allow_html=True)
                    st.image(bina,caption = "Modified Image" , width = 400)
                    
                except:
                    st.warning('Something went wrong :((')
		
            if opn=="Stretch/Shrink":
                from skimage.transform import resize
                try:
                    bina = np.array(img)
                    h = st.number_input('Enter height' , value = 100)
                    w = st.number_input('Enter width' , value = 100)
                    skimg = resize(bina , (int(h),int(w)))
                    #binna = Image.fromarray(np.array([skimg])
                    #st.markdown(get_image_download_link(binna), unsafe_allow_html=True)
                    st.image(skimg,caption = "Modified Image" , width = 400)
                    
                except:
                    st.warning('Something went wrong :((')		
		
		
		
		
            if opn=="Edge Enhance":
                from PIL import ImageFilter
                try:
                    bina = img.filter(ImageFilter.EDGE_ENHANCE)
                    st.markdown(get_image_download_link(bina), unsafe_allow_html=True)
                    st.image(bina,caption = "Modified Image" , width = 400)
                except:
                    st.warning('Something went wrong :((')   
		
            if opn=="Contour":
                from PIL import ImageFilter
                try:
                    bina = img.filter(ImageFilter.CONTOUR)
                    st.markdown(get_image_download_link(bina), unsafe_allow_html=True)
                    st.image(bina,caption = "Modified Image" , width = 400)
                except:
                    st.warning('Something went wrong :((')		
		
            if opn=="Grayscale":
                try:
                    bina = img.convert("L")
                    st.markdown(get_image_download_link(bina), unsafe_allow_html=True)
                    st.image(bina,caption = "Modified Image" , width = 400)
                    
                except:
                    st.warning('Something went wrong :((')    
            if opn=="BoxBlur":
                from PIL import ImageFilter
                sizz = st.number_input(label = 'Enter radius')
                try:
                    bina = img.filter(ImageFilter.BoxBlur(radius = sizz))
                    st.markdown(get_image_download_link(bina), unsafe_allow_html=True)
                    st.image(bina,caption = "Modified Image" , width = 400)
                except:
                    st.warning('Something went wrong :((')
	     	
	
	
	
            if opn=="GaussianBlur":
                from PIL import ImageFilter
                sizz = st.number_input(label = 'Enter radius')
                try:
                    bina = img.filter(ImageFilter.GaussianBlur(radius = sizz))
                    st.markdown(get_image_download_link(bina), unsafe_allow_html=True)
                    st.image(bina,caption = "Modified Image" , width = 400)
                except:
                    st.warning('Something went wrong :((')
            
            if opn=="Kernel (convolution kernel)":
                from PIL import ImageFilter
                sizz = st.number_input(label = 'Enter kernel size (choose only 3 or 5)', value = 3)
                try:
                    bina = img.filter(ImageFilter.Kernel((int(sizz),int(sizz)) , [-1,-1,-1,-1,9,-1,-1,-1,-1]+ [-1]*(sizz**2 - 9)))
                    st.markdown(get_image_download_link(bina), unsafe_allow_html=True)
                    st.image(bina,caption = "Modified Image" , width = 400)
                except:
                    st.warning('Something went wrong :((') 

            if opn=="RankFilter":
                from PIL import ImageFilter
                sizz = st.number_input(label = 'Enter kernel size (recommended: 3 or 5)', value = 3)
                rank = st.number_input(label = 'Enter rank (0 for min filter , size*size/2 for median filter , size*size-1 for max filter)', value = 0, max_value = sizz*sizz -1)
                try:
                    bina = img.filter(ImageFilter.RankFilter(size = sizz , rank = rank))
                    st.markdown(get_image_download_link(bina), unsafe_allow_html=True)
                    st.image(bina,caption = "Modified Image" , width = 400)
                except:
                    st.warning('Something went wrong :((') 
            
            if opn=="AutoContrast":
                from PIL import ImageOps
                st.info('Removes cutoff % of lightest and darkest pixels and makes them 255 and 0 respectively')
                sizz = st.number_input(label = 'Enter cutoff', value = 0)
                try:
                    bina = ImageOps.autocontrast(img , cutoff = sizz)
                    st.markdown(get_image_download_link(bina), unsafe_allow_html=True)
                    st.image(bina,caption = "Modified Image" , width = 400)
                except:
                    st.warning('Something went wrong :((')            
            
            if opn=='Colorize B-W image':
                from PIL import ImageOps
                st.info('Colorize black and white images')
   
                sizz = st.text_input(label = 'Color to use for black pixels , enter three numbers between () separated by commas (range 0-255)', value = "()")
               
                if len(sizz)>=7 and len(sizz)<=13 and sizz.count(",")==2:
                    try:
                        sizz = sizz[1:len(sizz)-1]
                        elem = sizz.split(",")
                        for i in range(3):
                            elem[i] = int(elem[i])
                    except:
                        st.warning('Values can not be processed')
                sizz2 = st.text_input(label = 'Color to use for white pixels , enter three numbers between () separated by commas (range 0-255)', value = "()")
               
                if len(sizz2)>=7 and len(sizz2)<=13 and sizz.count(",")==2:
                    try:
                        sizz2 = sizz2[1:len(sizz2)-1]
                        elem2 = sizz2.split(",")
                        for i in range(3):
                            elem2[i] = int(elem2[i])
                    except:
                        st.warning('Values can not be processed')        
                try:
                    bina = ImageOps.colorize(img , black = (elem[0],elem[1],elem[2]) , white = (elem2[0],elem2[1],elem2[2]))
                    st.markdown(get_image_download_link(bina), unsafe_allow_html=True)
                    st.image(bina,caption = "Modified Image" , width = 400)
                except:
                    st.warning('Something went wrong :((')  
            if opn=='Padding':
                from PIL import ImageOps
                sizz = st.number_input(label = 'Enter width', value = 100,max_value = 10000)
                sizz2 = st.number_input(label = 'Enter height', value = 100 , max_value = 10000)
                try:
                    bina = ImageOps.pad(img , size = (int(sizz) , int(sizz2)))
                    st.markdown(get_image_download_link(bina), unsafe_allow_html=True)
                    st.image(bina,caption = "Modified Image" , width = 400)
                except:
                    st.warning('Something went wrong :((') 

            if opn=='Equalize':
                from PIL import ImageOps
                try:
                    bina = ImageOps.equalize(img)
                    st.markdown(get_image_download_link(bina), unsafe_allow_html=True)
                    st.image(bina,caption = "Modified Image" , width = 400)
                except:
                    st.warning('Something went wrong :((') 
            
            if opn=='Posterize':
                from PIL import ImageOps
                st.info('Reduces number of bits for each color channel')
                sizz = st.number_input(label = 'Enter bits to keep', value = 1,max_value = 8 , min_value = 1)
                try:
                    bina = ImageOps.posterize(img , sizz)
                    st.markdown(get_image_download_link(bina), unsafe_allow_html=True)
                    st.image(bina,caption = "Modified Image" , width = 400)
                except:
                    st.warning('Something went wrong :((')    
                    
        except:
            st.warning('Invalid/Corrupted File')








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
                        st.warning('This dataset has missing values/too many values to encode/is corrupted')

        elif "(missing values)" in error:
             ind = error.index("(missing values)")
             str_check = True
             try:
                 dataset[error[:ind-1]] = dataset[error[:ind-1]].astype(float)
             except:
                 str_check = False
                 st.warning('Selected Column has string values, so numerical preprocessing will not work')

             if dataset[error[:ind-1]].dtype==float:
                 choicee = st.selectbox('Select Strategy' , ["None" , "Back Fill" , "Forward Fill","Mean Fill" , "Median Fill" , "Mode Fill" , "Random Fill(random value from same column)" , "Custom Fill"])	
                 if choicee=='Back Fill':
                     dataset[error[:ind-1]].fillna(method='bfill' , inplace=True)
                     st.markdown(get_table_download_link(dataset), unsafe_allow_html=True)
                     st.table(dataset.head(8))
                 if choicee=='Forward Fill':
                     dataset[error[:ind-1]].fillna(method='ffill' , inplace=True)
                     st.markdown(get_table_download_link(dataset), unsafe_allow_html=True)
                     st.table(dataset.head(8))
                     
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
                     dataset[error[:ind-1]].fillna(dataset[error[:ind-1]].mode()[0] , inplace = True)
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
                 choice_text = st.selectbox('Select Strategy' , ["None" ,"Back Fill" , "Forward Fill", "Custom Fill" , "Mode Fill" , "Random Fill"])
                 if choice_text=='Back Fill':
                     dataset[error[:ind-1]].fillna(method='bfill' , inplace=True)
                     st.markdown(get_table_download_link(dataset), unsafe_allow_html=True)
                     st.table(dataset.head(8))
                 if choice_text=='Forward Fill':
                     dataset[error[:ind-1]].fillna(method='ffill' , inplace=True)
                     st.markdown(get_table_download_link(dataset), unsafe_allow_html=True)
                     st.table(dataset.head(8))
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
                    
                     dataset[error[:ind-1]].fillna(random.choice(dataset[dataset[error[:ind-1]] != np.nan][error[:ind-1]]) , inplace=True)
                     st.markdown(get_table_download_link(dataset), unsafe_allow_html=True)
                     st.table(dataset.head(8))
                     
              



elif mode=='Get Best Model and its code':
    st.header('Upload Data Here')
    st.warning('Please preprocess the data before proceeding')
    data = st.file_uploader(label="Select File (.csv or .xlsx or .data)" , type=['csv','xlsx','data'])
    if data is not None:
        try:
            dataset = pd.read_csv(data)
            x = st.checkbox('Show head of the dataset')
            if x:
                st.table(dataset.head())	
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
                metric = st.selectbox('Choose scoring metric' , scoring)
                st.info('Estimated wait time : 8-12 minutes for small to medium datasets. 15-20 minutes for large datasets')
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
    
    
          


