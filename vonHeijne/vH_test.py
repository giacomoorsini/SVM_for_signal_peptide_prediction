if __name__=="__main__":
    import numpy as np
    from sys import argv
    import pandas as pd
    from VH_lib.vonheijne import *
    if len(argv)!=3:
        print("The command takes as input three tsv files and a text file that will be the output")
    else:
        print("processing as training set %s and as testing set %s"%(argv[1],argv[2]))
        
        #import a csv file previously computed from which to retrieve the thresholds of the crossvalidation procedure and make an average threshold
        tr_set_results=pd.read_table("training_predictions.tsv")
        av_threshold=np.round(np.average(tr_set_results["Threshold"]),3)
        
        #open the training set, make a MSA of the cleavage sites, calculate the PSPM and PSWM that will be used toscore the testing set
        training_set=open(argv[1],"r")
        msa=make_cleavage_site(training_set)
        pspm=PSPM(msa)
        pswm=PSWM(pspm)
        #open the testing set, score it, and retrieve the prediction results. Save them into a file
        benchmarking_set=open(argv[2],"r")
        scores,true=predict_score(benchmarking_set,pswm)
        benchmarking_set.close()
        
        benchmarking_set=open(argv[2],"r")
        acession_codes=[]
        
        for line in benchmarking_set:
            line=line.rstrip()
            line=line.split("\t")
            ac=line[3]
            acession_codes.append(ac)
        benchmarking_set.close()
        predicted,mcc, fscore, acc, prc, rcc, auc_,tn,fp,fn,tp=testing(scores,true,av_threshold)
        append_data_to_tsv(".","benchmarking_results.tsv",[str((argv[2].split("/"))[2]),av_threshold,prc, rcc,fscore,acc,mcc,auc_,tn,fp,fn,tp])
        
        for entry in range(0,len(acession_codes)):
            make_df_predvstrue(".","predicted_vs_true.tsv", [acession_codes[entry],predicted[entry],true[entry],scores[entry]])
    
    