import pandas as pd
import numpy as np
from sklearn import svm, metrics
from SVM_lib.SVM_scales import y, C, K, aa_list

#define the column names for a new sataframe that will hold the results
header=["training","cross validation","testing","best K", "best y", "best C", "MCC cross val", "MCC test"]

#define a set of lists that hold the combination of features to use for the models creation, iteratively
base=["class"]
hd=["hp_max","hp_pos","hp_avg"]
ch=["ch_max","ch_pos","ch_avg"]
hx=["ah_max","ah_pos","ah_avg"]

hydro=hd+base
charge=ch+base
helix=hx+base

hydro_charge=hd+ch+base
hydro_helix=hd+hx+base
charge_helix=ch+hx+base

all=hd+ch+hx+base

#define a list containing the feature combinations
features=[base,hydro,charge,helix,hydro_charge,hydro_helix,charge_helix,all]
#associate a list of codes for naming purposes
codes=["","HP","CH","AH","HP_CH","HP_AH","CH_AH","HP_CH_AH"]

#for combination of features in the feature list
print("starting to create models")
for feature in features:
    print("processing set of features: "+str(feature))
    
    #define a positional index for naming purposes (see the end of the code)
    pos=features.index(feature)
    code=codes[pos]
    
    #define a vector that holds the name of the columns of a dataframe containing the encoded features
    vector=aa_list+feature
    
    #create a new dataframe with the header as the name of columns
    df=pd.DataFrame(columns=header)
    
    #for loop to iterate trough the combination of CV file to use for training/cv/testing
    for i in [0,1,2,3,4]:
        print("processing combination: "+ str(i))
        
        #define list to hold MCC value and parameters for the 80 models that will be generated in the next steps for the combination of K/y/c
        MCC_list=[]
        parameters_list=[]
        
        #for all the k in the list of length of the fragments
        for k in K:
            
            #import as pandas dataframe the csv files containing the feature encoded for different values of k 
            #training sets
            tr1=pd.read_table("encoded/CV%s_FE_K%s.tsv"%(i%5,k))
            tr2=pd.read_table("encoded/CV%s_FE_K%s.tsv"%((i+1)%5,k))
            tr3=pd.read_table("encoded/CV%s_FE_K%s.tsv"%((i+2)%5,k))
            #cv set
            cv=pd.read_table("encoded/CV%s_FE_K%s.tsv"%((i+3)%5,k))
            
            #of the training dfs, take just the columns specified in the vector (corresponding to the wanted features) and concatenate them
            frames=[tr1[vector],tr2[vector],tr3[vector]]
            train=pd.concat(frames)
            
            #exlude the column class form the vector
            vector1=vector[:-1]
            
            #for both training set and cv set transform the features into a numpy matrix and the class column into an array
            x_train=train[vector1].to_numpy()
            x_true=train["class"].to_list()
            y_train=cv[vector1].to_numpy()
            y_true=cv["class"].to_list()

            #try all the combinations of gamma and cost
            for gamma in y:    
                for c in C:
                    
                    #create an SVM
                    mySVC = svm.SVC(C=c, kernel='rbf', gamma=gamma)
                    #fit it with the training data
                    mySVC=mySVC.fit(x_train, x_true)
                    #predict the class on the CV set features
                    y_pred = mySVC.predict(y_train)
                    
                    #calculate the mcc and append it and the parameters to the lists defined at the beginning (parameters appendended as tuple)
                    MCC=metrics.matthews_corrcoef(y_true,y_pred)
                    MCC_list.append(np.round(MCC,3))
                    parameters_list.append((k,gamma,c))
                    
        print("finished processing all 80 models for the feature combination")
        
        #take the index of the best MCC and use it to retrieve the parameters of the model that perfomed best
        index=np.argmax(MCC_list)  
        bestK=parameters_list[index][0]
        bestY=parameters_list[index][1]
        bestC=parameters_list[index][2]
        
        #we want to test this model on the testing cv set, so we have to recreate it
        #this is the same procedure of before but with the endoced feature from the best k and a tetsing set
        tr2_1=pd.read_table("encoded/CV%s_FE_K%s.tsv"%(i%5,bestK))
        tr2_2=pd.read_table("encoded/CV%s_FE_K%s.tsv"%((i+1)%5,bestK))
        tr2_3=pd.read_table("encoded/CV%s_FE_K%s.tsv"%((i+2)%5,bestK))
        test=pd.read_table("encoded/CV%s_FE_K%s.tsv"%((i+4)%5,bestK))

        #concatenate the training subsets with the wanted features
        frames=[tr2_1[vector],tr2_2[vector],tr2_3[vector]]
        train=pd.concat(frames)
        
        vector1=vector[:-1]
        
        #extract the features in a matrix and the classes in list    
        x_train=train[vector1].to_numpy()
        x_true=train["class"].to_list()
        y_train=test[vector1].to_numpy()
        y_true=test["class"].to_list()

        #create a new model with the best parameters 
        testSVC = svm.SVC(C=bestC, kernel='rbf', gamma=bestY)
        #fit it with the correct features
        testSVC=testSVC.fit(x_train, x_true)
        #predict the classes of the testing set
        y_pred = testSVC.predict(y_train)
        #calculate the testing mcc
        TEST_MCC=metrics.matthews_corrcoef(y_true,y_pred)  
        
        print("testing for best model out of the 80 finished \n")
        #define a row with the data to append in the dataframe that will be combination of files specific       
        row=["%s,%s,%s"%(i%5,(i+1)%5,(i+2)%5), "%s"%((i+3)%5), "%s"%((i+4)%5), bestK, bestY,bestC,max(MCC_list),np.round(TEST_MCC,3)]
        df.loc[len(df)]=row    
    
    #save the dataframe for each combination of features
    df.to_csv('./models/%s_C_models.tsv'%(code),sep='\t')

print("process finished")