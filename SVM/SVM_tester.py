import pandas as pd
import numpy as np
from SVM_lib.SVM_lib import AddScale, ProteinScaler, MaxAvgScaler,ComputeComposition, make_df_predvstrue
from SVM_lib.SVM_scales import hydro, sw_composition, a_helix, charge, aa_list
from sklearn import svm, metrics

#from the models evaluation it has been seen that the best K is 23, so to recreate the training set is sufficient to concatenate back al lthe CV sets with that k length

models=pd.read_table("./models/models_evaluation.tsv")
best_model=models["avarage MCC"].idxmax()
bestK=models.at[best_model,"K"]

#read the first file
df=pd.read_table("./encoded/CV0_FE_K%s.tsv"%(bestK))

#then concatenate all the other cvs
for i in (1,2,3,4):
    df_=pd.read_table("./encoded/CV%s_FE_K%s.tsv"%(i,bestK))
    df=pd.concat([df,df_])

#features of the best model
feature=["hp_max","hp_pos","hp_avg","ch_max","ch_pos","ch_avg"]

#define a vector with the features to extract
vector=aa_list+feature

#set the features to a matrix and the classes as an arrow
x_train=df[vector].to_numpy()
x_test=df["class"]

#create a new model which has the parameter of the best model of the cross validation procedure and will be train on the wall training set 
myTESTSVC = svm.SVC(C=4, kernel='rbf', gamma=1)
myTESTSVC=myTESTSVC.fit(x_train, x_test)

#comlete the scales
hydro=AddScale(hydro)
charge=AddScale(charge)
helix=AddScale(a_helix)

#co mpute the feature encoding for the benchmarking set
names=['sequence', 'class', 'code',]
bench_df=pd.read_table("./SETs/benchmarking_set_parsed_totc.tsv", names=names)
#extract sequence and class
S= bench_df.iloc[:,0].to_list()
C=bench_df.iloc[:,1].to_list()
AC= bench_df.iloc[:,2].to_list()

#define a dataframe to hold the encoding
headers= aa_list+['hp_max', 'hp_pos', 'hp_avg', 'ch_max', 'ch_pos', 'ch_avg', 'ah_max', 'ah_pos', 'ah_avg']
enc_bench_df = pd.DataFrame(columns=headers)
seq_comp=ComputeComposition(sw_composition,S,23)

#compute the feature values for each residue    
BN_hydro_scores=ProteinScaler(S,hydro,5,23)
BN_charge_scores=ProteinScaler(S,charge,5,23)
BN_helix_scores=ProteinScaler(S,helix,7,23)

#compute the average, max and position of the max for each sequence
for i in range(len(seq_comp)):
    hp_max, hp_pos, hp_avg=MaxAvgScaler(hydro,BN_hydro_scores[i])
    ch_max, ch_pos, ch_avg=MaxAvgScaler(charge,BN_charge_scores[i])
    ah_max, ah_pos, ah_avg=MaxAvgScaler(helix,BN_helix_scores[i])
    
    #append all the data to the dataframe
    row_data = seq_comp[i] + [hp_max, hp_pos, hp_avg, ch_max, ch_pos, ch_avg, ah_max, ah_pos, ah_avg]
    enc_bench_df.loc[len(enc_bench_df)]=row_data

#add the class column to the df    
enc_bench_df['class']=C

#save the encoding in a csv file
fn="BN_FE_K23.tsv"
enc_bench_df.to_csv('./SETs/%s'%(fn),sep='\t')

#for the testing set, divide class and features in array and matrix
y_train=enc_bench_df[vector].to_numpy()
y_true=enc_bench_df["class"].to_list()

#predict the classes of the benchmarking set using the model trained on the training set
y_pred = myTESTSVC.predict(y_train)

#define a new dataframe to hold the finalr esult
head=["k","y","C","mcc","acc","rcc","prc","tn","fp","fn","tp"]
result_df=pd.DataFrame(columns=head)

#define all the metrics
auc_=np.round(metrics.roc_auc_score(y_true,y_pred),3)          #auc
mcc=np.round(metrics.matthews_corrcoef(y_true,y_pred),3)         #mcc
acc=np.round(metrics.accuracy_score(y_true,y_pred),3)            #acc
rcc=np.round(metrics.recall_score(y_true,y_pred),3)              #rcc
prc=np.round(metrics.precision_score(y_true,y_pred),3)           #prc
tn,fp,fn,tp=metrics.confusion_matrix(y_true,y_pred).ravel()      #confusion matrix

#append everything to the dataframe
row=[23,1,4,mcc,acc,rcc,prc,tn,fp,fn,tp]
result_df.loc[len(result_df)]=row

#save evrything into a csv file
result_df.to_csv("./SETs/benchmarking_results.tsv",sep="\t")

accesion_codes=bench_df
for entry in range(0,len(C)):
    make_df_predvstrue(".","predicted_vs_true.tsv", [AC[entry],y_pred[entry],C[entry]])