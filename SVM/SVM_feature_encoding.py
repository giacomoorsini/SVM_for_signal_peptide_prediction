import pandas as pd
from SVM_lib.SVM_lib import AddScale, ProteinScaler, MaxAvgScaler,ComputeComposition
from SVM_lib.SVM_scales import hydro, sw_composition, a_helix, charge, aa_list,K
import os

#complete the scales with special characters
hydro=AddScale(hydro)
charge=AddScale(charge)
helix=AddScale(a_helix)

#with the os module, define a list of all the file in the directory CVs
directory="./CVs/"
CVs=os.listdir(directory)

#for every cv file in the CVs folder
for cv in CVs:
    
    #open the file as a pandas dataframe
    df=pd.read_table(directory+cv)
    
    #make a list out of the column containg sequences and class
    S= df.iloc[:,1].to_list()
    C=df.iloc[:,2].to_list()

    #for all the k in the k list (lengths)
    for k in K:
        
        #define a new dataframe to contain the feature encoding
        headers= aa_list+['hp_max', 'hp_pos', 'hp_avg', 'ch_max', 'ch_pos', 'ch_avg', 'ah_max', 'ah_pos', 'ah_avg']
        cv_df = pd.DataFrame(columns=headers)

        #compute the sequence composition and the scale the other features so to have for each sequence the feature of each residue
        seq_comp=ComputeComposition(sw_composition,S,k)
        CV_hydro_scores=ProteinScaler(S,hydro,5,k)
        CV_charge_scores=ProteinScaler(S,charge,5,k)
        CV_helix_scores=ProteinScaler(S,helix,7,k)
        
        #for index in the length of the sequence compisiton list (it is a list containing lists)
        for i in range(len(seq_comp)):
            
            #retrieve for each sequence the max, average and position of the value
            hp_max, hp_pos, hp_avg=MaxAvgScaler(hydro,CV_hydro_scores[i])
            ch_max, ch_pos, ch_avg=MaxAvgScaler(charge,CV_charge_scores[i])
            ah_max, ah_pos, ah_avg=MaxAvgScaler(helix,CV_helix_scores[i])
            
            #define the row to append to the dataframe
            row_data = seq_comp[i] + [hp_max, hp_pos, hp_avg, ch_max, ch_pos, ch_avg, ah_max, ah_pos, ah_avg]
            
            #append it
            cv_df.loc[len(cv_df)]=row_data

        #add the class column
        cv_df['class']=C
        
        #define a name to save the dataframe in tsv format for later use
        fn=str(cv).split("/")
        fn=fn[-1].split(".")
        fn=fn[0]
        cv_df.to_csv('./encoded/%s_FE_K%s.tsv'%(fn, k),sep='\t')
            
