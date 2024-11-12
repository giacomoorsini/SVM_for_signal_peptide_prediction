import numpy as np
import os
import pandas as pd
from sklearn.metrics import precision_recall_curve, matthews_corrcoef, accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix

def make_cleavage_site(CV):
    '''this function takes as input a tsv file, specifically formatted so to have as first item the length of the signal peptide.
    It gives as output a list containing the cleavage site for every sequence in the tsv file'''
    
    #define a training set, which will be a list of sequences with the same length
    training_set=[]
	#extract the sequences from each line and put it in the training set
    for line in CV:
        if "Sequence" not in line:
            line=line.split("\t")
            end=int(line[0])        #this line is the one containing the length of the SP
            cleavage_site=line[1][end-13:end+2]
            training_set.append(cleavage_site)
    return training_set

def PSPM(S):
    '''This function creates a PSPM '''
    N=len(S)        #length of the MSA, given as a py list
    L=len(S[0])     #length of the sequences alligned
    amminoacids=list("GAVPLIMFWYSTCNQHDEKR")      #list of amminoacids
    
    #creating the M matrix considering the number of amminoacids and length of MSA sequences      
    x=len(amminoacids)
    PSPM=np.ones((x,L))
    #iterating trough the list of MSA sequences and adding values to the M matrix
    for i in range(N):
        for j in range(L):
            if S[i][j] not in "XUZ":              #S[i][j] is the residue at position j in the sequence i of the MSA
                
                k=amminoacids.index(S[i][j])      #k is equal to the position of the residue S[i][j] in the list of amminoacids
                PSPM[k][j]+=1                     #update the M matrix at position j of the amminoacid k row adding a 1
    
    #dividing all the values of the M matrix for the total number of sequences in the MSA
    for aa in range(0,len(amminoacids)):
        for j in range(L):
            PSPM[aa][j]=PSPM[aa][j]/(N+20)
    #return the M matrix
    return PSPM

def PSWM(PSPM):
    '''Compute a PSWM matrix from a PSPM matrix'''
    swissProt_composition={ "G" :7.07,"A": 8.25, "V" :6.85, "P": 4.74, "L": 9.65, "I": 5.91, "M" :2.41, "F" :3.86, "W" :1.10,"Y": 2.92,"S": 6.65, "T" :5.36,"C": 1.38, "N": 4.06, "Q": 3.93, "H": 2.27, "D": 5.46, "E": 6.72, "K": 5.80, "R": 5.53}
    amminoacids=list("GAVPLIMFWYSTCNQHDEKR")
    
    L=(len(PSPM[0]))      #length of the PSPM alligned
    
    #creating the M matrix considering the number of amminoacids and length of MSA sequences      
    x=len(amminoacids)
    PSWM=np.zeros((x,L))

    #iterating trough the PSPM and adding log transformed values to the W matrix 
    for i in range(0,len(PSPM)):    
        for j in range(L):
            k=amminoacids[i]     #k is equal to the amminoacid in position i in the list
            
            if PSPM[i][j]!=0:  
                PSWM[i][j]=np.log(PSPM[i][j]/(swissProt_composition[k]/100))     #the value in position j of amminoacid row i is equal to the log of the value in position i j in the m matrix divided by the frequency of the amminoacid k
    
    #return a PSWM matrix
    return PSWM

def predict_score(dataset, PSWM):
    '''This function predicts a score indicating the probability of having a signal peptide in a set of 90 aa long sequences.
    Each sequence is scanned with a 15 aa long sliding window, inside the sliding window, each residue is compared with a PSWM 
    previously computed. The values of all the residues are summed and a total value for the sliding window is computed.
    In the end, only the value of the best sliding window is kept for each sequence. The output is a list containing the probability of 
    having a SP in each sequence. Moreover, the function also extract the original class of the sequence (being 1 having a SP, 0 not having it)
    and returns a list containing these values.'''
    
    amminoacids=list("GAVPLIMFWYSTCNQHDEKR")
    pos_scores=[]         #creating the list containing the SP probabilities
    
    true_class=[]
    
    for line in dataset: #for sequence in the tsv/txt file 
        if "SP length" not in line:    #skip the first line if there is
            line=line.rstrip()         #eliminate the \n at the end of each line
            line=line.split("\t")      #separate the line at eact tab
            true_class.append(int(line[2]))
            if len(line[1])<90:        #extract the chunk at position 1, it will be the sequence, and take just 90 aa or less
                end=len(line[1])
                sequence=line[1][0:end] 
            else: 
                sequence=line[1][0:90]
                
            SP_probability=[]          #define a list to store the partial scores of each sequence
            
            for i in range(0,len(sequence)-14):        #define the positions of the sliding window
                sliding_window=sequence[i:i+15]      #define the sliding window
                sliding_window_score=0               #define the sliding window score
                
                for residue in range(len(sliding_window)):
                    if sliding_window[residue] not in "XUZ":   
                        j=amminoacids.index(sliding_window[residue])      #j is the number of the aa in the aa list corresponding to the residue in the slding window
                        residue_score=PSWM[j][residue]                    #the score of each residue is equal to the score of the aa j in position residue of the sliding window
                        sliding_window_score+=residue_score               #update the sliding window score
                        
                SP_probability.append(sliding_window_score)          #store the sliding window score in the list of the scores of the sequence
                
            pos_scores.append(max(np.round(SP_probability,3)))       #in the list containing the probability for each sequence, store the max probability between the slding windows of each sequence
    
    #return the list containing the SP probabilities
 
    return pos_scores,true_class

def validation(val_true,val_scores):
    '''this function takes as input the predicted scores and the real class for the validation set. it gives as output the optimal threshold, 
    in order to use this as the threshold to score the testing set'''
    
    #to calculate the precision recall curve use a function imported form the scikit learn module
    precision, recall, thresholds = precision_recall_curve(val_true, val_scores)
    #calculate the f score
    fscore = (2 * precision * recall) / (precision + recall)
	#calculate and store the optimal threshold
    index = np.argmax(fscore)       #the fscore is a list
    optimal_threshold = thresholds[index]
    return optimal_threshold

def testing(test_scores,test_true, opt_thrs):
    '''This function takes as input the predicted scores for the testing set, the true class values and an optimal threshold. It retrieves as 
    ouput parameters that score the goodnes of the prediction'''
    
    #reframe the prediction (having SP or not) by using the threshold to discriminate the previously computed score
    test_pred = [int(t_s >= opt_thrs) for t_s in test_scores]
    #calculate precision and recall
    precision, recall, thresholds = precision_recall_curve(test_true, test_scores)
    
    fscore = (2 * precision * recall) / (precision + recall)
    index = np.argmax(fscore)                                       #fscore
    
    auc_=np.round(roc_auc_score(test_true, test_scores),3)          #auc
    mcc=np.round(matthews_corrcoef(test_true, test_pred),3)         #mcc
    acc=np.round(accuracy_score(test_true, test_pred),3)            #acc
    rcc=np.round(recall_score(test_true, test_pred),3)              #rcc
    prc=np.round(precision_score(test_true, test_pred),3)           #prc
    tn,fp,fn,tp=confusion_matrix(test_true,test_pred).ravel()       #confusion matrix
    
    return test_pred,mcc, np.round(fscore[index],3), acc, prc, rcc, auc_,tn,fp,fn,tp

def append_data_to_tsv(dir_path, file_name, data_to_append):
    '''This function creates a csv file with the results of the prediction. if the file already exists, it will update it trating as a pandas dataframe'''
    
    file_exists = os.path.isfile('%s/%s'%(dir_path,file_name)) # Check if the file exists
    if file_exists: # Create or load a DataFrame
        df = pd.read_csv(file_name, sep='\t')
    
    else:# Create a new DataFrame with your data structure
        df = pd.DataFrame(columns=["Run",'Threshold', 'Precision', 'Recall', 'F1', 'ACC', 'MCC', 'AUC', 'TN','FP','FN','TP'])
    
    new_row = pd.DataFrame([data_to_append], columns=df.columns) # Create a new row using the data list
    
    df = pd.concat([df, new_row]) # Append the new row to the DataFrame

    df.to_csv('%s/%s'%(dir_path,file_name), sep='\t', index=False)
    
def make_df_predvstrue(dir_path, file_name, data_to_append):
    '''This function creates a csv file with the results of the prediction. if the file already exists, it will update it trating as a pandas dataframe'''
    
    file_exists = os.path.isfile('%s/%s'%(dir_path,file_name)) # Check if the file exists
    if file_exists: # Create or load a DataFrame
        df = pd.read_csv(file_name, sep='\t')
        
    else:# Create a new DataFrame with your data structure
        df = pd.DataFrame(columns=["Acession code",'Predicted', 'True', 'Score'])
    
    new_row = pd.DataFrame([data_to_append], columns=df.columns) # Create a new row using the data list
    
    df = pd.concat([df, new_row]) # Append the new row to the DataFrame

    df.to_csv('%s/%s'%(dir_path,file_name), sep='\t', index=False)