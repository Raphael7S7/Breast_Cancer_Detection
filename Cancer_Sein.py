#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 10:08:25 2021

@author: raphaelmartin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
plt.style.use("seaborn-darkgrid")
import base64
import os

from sklearn.tree import plot_tree
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, plot_confusion_matrix
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

import streamlit as st


data=pd.read_csv("/Users/raphaelmartin/Desktop/Cancer_detection/data.csv",sep=",")
data=data.drop(['id','Unnamed: 32'],axis=1)
label='diagnosis'
choix=np.append(data.columns,'all')

st.title('Aide à la detection du cancer du sein')

def main():
    
    st.image("/Users/raphaelmartin/Desktop/Cancer_detection/download.jpg")
    
    if st.sidebar.checkbox('Contexte'):
    
        st.subheader("Contexte")
            
        st.markdown('''
                L'intelligence artificielle (IA) peut améliorer les performances 
                des radiologues dans la lecture des mammographies de dépistage du cancer du sein,
                selon une étude publiée dans Radiology: Artificial Intelligence.
                Un logiciel de ce type a obtenu la validation de la FDA américaine en mars 2020.''')
            
        st.image("/Users/raphaelmartin/Desktop/Cancer_detection/images.jpg")
            
        st.markdown('''
                Dans le cadre du dépistage du cancer du sein par mammographie,
                de nombreuses lésions malignes passent inaperçues et les résultats suspects
                s'avèrent souvent bénins. Une étude antérieure parue dans 
                la Revue Radiology a révélé qu'en moyenne, seulement 10% des femmes étaient
                rappelées pour un bilan diagnostique supplémentaire basé
                sur des résultats suspects objectivant finalement un cancer. Nous developpons ici un
                modèle de detection de cancer grâce à un IA entrainer sur un echantillon
                de quelque 500 images dont les mesures sont répertoriées notre base de données
                la base de données. 
                ''')
                
        st.image("/Users/raphaelmartin/Desktop/Cancer_detection/images-1.jpg")
             
        st.markdown('''
                     Les algorithmes d'IA représentent une solution prometteuse pour
                     améliorer la précision de la mammographie numérique.
                     Les développeurs « entrainent » l'IA sur des images existantes
                     , lui apprennent à reconnaître les anomalies associées au cancer
                     et à les distinguer des résultats bénins.\n
                     Cet outils permet une aide à la décision pour les radiologues cherchant à 
                     détecter un cancer sur de futurs images.
                     ''')
                     
    if st.sidebar.checkbox('Analyse des données'):
        st.subheader('Affichage des données disponibles')
        
        st.dataframe(data)
        st.write("Dimension : ",data.shape)
        st.markdown('''
                    On dispose de 569 observations avec 33 variables explicatives, toute qualitative. 
                        ''')
            
        st.write(data.columns)
        st.write("\n")
        
        st.subheader('Statistiques de base')
        st.write(data.describe())
        
        st.subheader("Analyse univariée ")
        st.markdown(r'''
                    On cherche à classifier les cancers en deux catégories: Malignes (grave) et Benin (moins grave)).
                    La base de données étudié comporte : 
                    ''')
                    
        fig=plt.figure(figsize=(5,5))
        plt.pie(data['diagnosis'].value_counts(),
        autopct="%.1f%%", 
        explode=[0.05]*2,
        labels=data['diagnosis'].value_counts().index.values,pctdistance=0.5,
        textprops={'fontsize': 20})
        plt.title("Repartition des cancers", fontsize=14)
        st.pyplot(fig)
        
        feature=st.selectbox("Variable",(data_features))
         
        fig=plt.figure(figsize=(6,6))
        sns.distplot(data[feature])
        plt.title(feature+' distribution')
        st.pyplot(fig)
            
        
        st.subheader("Analyse bivariée")
        features=st.multiselect("Variables ", (data.columns.values))
        
        for feat in features:
            fig=plt.figure(figsize=(14,7))
            plt.subplot(2,2,1)
            sns.kdeplot(data[feat].loc[data[label]=='M'],color='darkblue',label='M',shade=True)
            sns.kdeplot(data[feat].loc[data[label]=='B'],color='darkorange',label='B',shade=True)
            plt.title('Distribution de la variable : '+feat,fontsize=20)
            plt.subplot(2,2,2)
            sns.boxplot(x=label,y=feat,data=data)
            st.pyplot(fig)
            
        st.subheader("Test d'indépendance")
        st.markdown('''
                   Cet outil basé sur la théorie des tests permet de déterminer pour un seuil donné quelles
                   variables ont le plus d'influence sur le type de cancer.\n
                   ''')
                   
         
        if st.checkbox('Effectuer un test'):
            
            alpha=st.number_input("Seuil du test : ")
            test=effectuer_test(alpha)
            st.dataframe(test)
            
            st.markdown(r'''
                        Les variables ayant une p-value supérieure au seuil choisi ont peu ou pas d'influence
                        sur la caractérisation du cancer (B ou M).
                        ''')
                        
            st.write(test.loc[test['p-value<'+str(alpha)]==str(False)]['Numerical'])
           
            if st.checkbox("Visualiser"):
                 for feat in test.loc[test['p-value<'+str(alpha)]==str(False)]['Numerical']:
                    fig=plt.figure(figsize=(8,4))
                    plt.subplot(2,2,1)
                    sns.kdeplot(data[feat].loc[data[label]=='M'],color='darkblue',label='M',shade=True)
                    sns.kdeplot(data[feat].loc[data[label]=='B'],color='darkorange',label='B',shade=True)
                    plt.title('Distribution de la variable : '+feat,fontsize=15)
                    plt.subplot(2,2,2)
                    sns.boxplot(x=label,y=feat,data=data)
                    st.pyplot(fig)
                            
            test_sorted=test.sort_values(by='test-statistic',ascending=False)
            st.markdown(r'''
                        A l'inverse les variables ayant le plus d'infuence sur le type de cancer
                        par ordre croissant sont celle avec une p-value faible. On obtient : 
                            ''')
            st.write(test_sorted[['Numerical','p-value']][:5])
            
                
            if st.checkbox("Visualiser 2"):
                for feat in test_sorted['Numerical'][:5]:
                    fig=plt.figure(figsize=(8,4))
                    plt.subplot(2,2,1)
                    sns.kdeplot(data[feat].loc[data[label]=='M'],color='darkblue',label='M',shade=True)
                    sns.kdeplot(data[feat].loc[data[label]=='B'],color='darkorange',label='B',shade=True)
                    plt.title('Distribution de la variable : '+feat,fontsize=15)
                    plt.subplot(2,2,2)
                    sns.boxplot(x=label,y=feat,data=data)
                    st.pyplot(fig)
        
        
        if st.button('Matrice de corrélation : '):
            df=data.copy()
            d=pd.get_dummies(df['diagnosis'])
            df=pd.concat([df,d],axis=1)
            
            fig=plt.figure(figsize=(50,8))
            sns.heatmap(df.corr().iloc[-2:,:],annot=True,fmt='0.1',linewidths=0.5,linecolor='black',annot_kws={"size": 20})
            st.pyplot(fig)
            
            
            
            
    if st.sidebar.checkbox('Outils de prédiction'):
        
        list_model=[f for f in dict_model.keys()]
        model=st.sidebar.selectbox('Modèle :',list_model)
        t=st.sidebar.slider('Taille du jeu de validation (en %) :',1,100,10)
        st.markdown('''
                    L'outil utilisé ici se base sur les principaux algorithmes d'intelligence articficielle
                    permettant de faire de la classification. \n
                    - Etape 1 : Le modèle choisie à gauche s'entraine sur une partie des données (choisie par l'utilisateur).
                    - Etape 2 : On vérifie le bon fonctionnement du modèle grâce a son accuracy et à la table des Faux Positifs/Faux Négatifs
                    - Etape 3 : On entre de nouvelles valeurs inconnues pour déterminer 
                    le type de cancer associé.
                    ''')
                    
        if st.button('RUN'):
            RUN_model(t/100,dict_model[model])
            
        
        if st.button('Effectuer une prédiction'):
            filename=st.text_area('Entrez l emplacement de votre fichier')
            st.write('Vous avez choisi le fichier `%s`' % filename)
            
            try:
                new_data=pd.read_csv(filename,sep=',')
                Pred=predire(model,np.array(new_data))
                st.markdown('''
                            Félicitation votre série de prédictions est : 
                                ''')
                st.dataframe(Pred)
                st.markdown(get_table_download_link(Pred), unsafe_allow_html=True)
            except:
                None
            
        
            
from scipy.stats import ttest_ind
label='diagnosis'
data_features=data.drop([label],axis=1).columns


def effectuer_test(alpha):
    dic={'Categorical':[],'Numerical':[],'p-value':[],'p-value<'+str(alpha):[],'test-statistic':[]}
    assert data[label].unique().size == 2
    
    for feature in data_features:
        value1=data[label].unique()[0]
        value2=data[label].unique()[1]
      
        A=data[data[label]==value1][feature].values
        B=data[data[label]==value2][feature].values
      
        statistic,pval=ttest_ind(A,B)
      
        dic['Categorical'].append(label)
        dic['Numerical'].append(feature)
        dic['p-value'].append(str(pval))
        dic['p-value<'+str(alpha)].append(str(pval<alpha))
        dic['test-statistic'].append(statistic)
        
    return pd.DataFrame(dic)


dict_model={'Regression Logistique':LogisticRegression(),
            'Forêts aléatoires':RandomForestClassifier(),
            'Support Vector Machine (SVM)':SVC(),
            'K-plus-proche-voisin':KNeighborsClassifier(),
            'XGBOOST classifier':xgb.XGBClassifier()}

def RUN_model(size,model):
    
    df=data.drop([label],axis=1)
    X=df
    le=LabelEncoder()
    Y=le.fit_transform(data[label])
    
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = size, random_state = 5)
    
    st.write("Taille jeu d'entrainement : ",x_train.shape)
    st.write('Taille jeu de validation : ',x_test.shape)
    
    
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    
    cf=confusion_matrix(y_test,y_pred)
    fig=plt.figure(figsize=(6,6))
    sns.heatmap(cf,annot=True,fmt='d',linewidths=0.2,
                linecolor='black',
                xticklabels=['B','M'], yticklabels=['B','M'])
    st.pyplot(fig)
    
    st.subheader("Performance obtenue :")
    st.write('Accuracy : ',np.round(accuracy_score(y_test,y_pred),decimals=3))
    st.write('Precision : ',np.round(precision_score(y_test,y_pred),decimals=3))
    
    
def predire(model,x_test):
    prediction=model.predict(x_test)
    return pd.DataFrame(prediction,columns=["Type de cancer"])


def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(
        csv.encode()
    ).decode()  # some strings <-> bytes conversions necessary here
    return f'<a href="data:file/csv;base64,{b64}" download="myfilename.csv">Download csv file</a>'



















main()