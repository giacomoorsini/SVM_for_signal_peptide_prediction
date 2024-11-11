import pandas as pd
import numpy as np
from sklearn import svm
import os 

def MinMaxScaler(scale:dict, x:float):
    '''this function performs the min max rescaling, so to rescale the input value between 0 and 1. It requires as input a scale in the form of a dict'''
    
    min_x=min(scale.values())
    max_x=max(scale.values())
    
    return (x-min_x)/(max_x-min_x)

def MaxAvgScaler(scale:dict,score:list):
    '''This function takes as input a propensity scale and a score list. it gives as output the average of the scores, the maximum value and 
    the positional index of the maximum score, which correspond to a residue in the sequence. All these values are scaled between 0 and 1 with the 
    min max function.'''
    
    k=len(score)
    
    Avg=MinMaxScaler(scale, np.mean(score))
    Max=MinMaxScaler(scale, max(score))
    peak=(np.argmax(score)+1)/k
    
    return Avg,Max,peak

def AddScale(scale:dict):
    '''This function complets the scale by adding values for undefined characters'''
    
    scale["X"]=0.0
    scale["B"]=(float(scale["D"])+float(scale["N"]))/2
    scale["Z"]=(float(scale["E"])+float(scale["Q"]))/2
    scale["J"]=(float(scale["I"])+float(scale["L"]))/2
    scale["U"]=float(scale["C"])
    scale["O"]=float(scale["L"])
    
    return scale
        
def ComputeComposition(sw_scale:dict,seqs:list,k:int):
    '''This function takes as input an aminoacidic composition scale, a list of sequences and a value k that represents the length of a segment of the sequences.
    It computes the aminoacidic composition of segment of length k of each sequence in the list. It returns a list of lists containing the composition.'''
    
    #list of aa
    aa=list(sw_scale.keys())
    
    #list that will hold the compositions of each sequence
    composition=[]
    
    #for every sequence in the list of sequences
    for seq in seqs:
        
        #select just a gragment of length k
        seq=seq[0:k]
        #intermediate score holder
        int_score=[]
        
        #count each special character to then remove it
        X=seq.count("X")
        Z=seq.count("Z")
        B=seq.count("B")
        J=seq.count("J")
        U=seq.count("U")
        O=seq.count("O")
        
        #for each aminoacid in the list of amminoacid
        for a in aa:
            
            #count the occurence of the aminoacid in the fragment and normalize the value by dividing for k minus the amount of special characters
            aa_comp=seq.count(a)/(len(seq)-X-Z-B-J-U-O)
            int_score.append(aa_comp)
        
        composition.append(int_score)
    
    return composition
             
def ProteinScaler(seqs:list,scale:dict,sw:int, k:int):
    '''This function takes as input a sequence, a scale and the length of the sliding window to calculate the feature (depending on the scale)
    of each residue in the sequence considering for the computation the surrounding residues'''
    
    #compute the half of the sliding window length to define later the sliding window
    if sw%2==0:
        d=int(sw/2)
    else:
        d=int((sw-1)/2)
    
    #define a list to hold the score of each residue of each sequence
    score_list=[]
    
    #for each sequence in the sequence list
    for seq in seqs:
        #define a fragment of length k
        seq=seq[0:k]
        
        #define a list to hold the score of each residue
        score_int_list=[]
        #iterate trough the sequence so to calculate the score of each residue in the sequence considering the nearby residues with a sliding window
        for i in range(len(seq)):
            score=0 
        #if the residue is at the beginning or at the end shrink the sliding window so to unbyas the calculation
            if i - d < 0 :
                sliding_window=seq[0:i+d+1]
            elif i+d> len(seq):
                sliding_window=seq[i-d:len(seq)]
            else:
                sliding_window=seq[i-d:i+d+1]
        
        #sum the individual score of each residue in the sliding window and divide it by the length of the sliding window    
            for residue in sliding_window:
                score+=scale[residue]  
            score=score/len(sliding_window)
        
        #store the result    
            score_int_list.append(score)      
        score_list.append(score_int_list)
    
    return score_list

def make_df_predvstrue(dir_path, file_name, data_to_append):
    '''This function creates a csv file with the results of the prediction. if the file already exists, it will update it trating as a pandas dataframe'''
    
    file_exists = os.path.isfile('%s/%s'%(dir_path,file_name)) # Check if the file exists
    if file_exists: # Create or load a DataFrame
        df = pd.read_csv(file_name, sep='\t')
        
    else:# Create a new DataFrame with your data structure
        df = pd.DataFrame(columns=["Acession code",'Predicted', 'True'])
    
    new_row = pd.DataFrame([data_to_append], columns=df.columns) # Create a new row using the data list
    
    df = pd.concat([df, new_row]) # Append the new row to the DataFrame

    df.to_csv('%s/%s'%(dir_path,file_name), sep='\t', index=False)
