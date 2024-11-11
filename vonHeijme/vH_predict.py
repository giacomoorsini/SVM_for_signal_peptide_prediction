if __name__=="__main__":
	import numpy as np
	from sys import argv
	from VH_lib.vonheijne import predict_score, testing, validation, append_data_to_tsv, make_df_predvstrue
	
	if len(argv)!=4:
		print("The command takes as input a tsv file as validation set, a text file containg the PSWM and a tsv file as testing set")
	else:
		print("processing as validation set: "+str((argv[1].split("/"))[2]))

		#load the PSWM matrix from a txt file previously computed and saved
		pswm=np.loadtxt("./PSWMs/%s"%(argv[2]))

		#open the dataset and predict the scores (probability of having a SP) for the training validation set. Return also the list of true classes (1 or 0, having SP or not)
		dataset=open(argv[1],"r")
		y_train_scores,val_true=predict_score(dataset, pswm)
		dataset.close()
  
		#calculate precision, recall and trhesholds with a sklearn function on the classes and the scores from the prediction
		optimal_threshold=validation(val_true, y_train_scores)
		#print("Optimal threshold: "+ str(optimal_threshold))
		
		###---->TESTING PROCEDURE
		
		print("processing as testing set: "+str((argv[3].split("/"))[2])+"\n")
		
		#open the training test set and predict the score, as well as the class of the true class
		test_set=open(argv[3],"r")
		y_test_scores,test_true=predict_score(test_set, pswm)
		test_set.close()
		
		#calculate all the values for prediction
		mcc, fscore, acc, prc, rcc, auc_,tn,fp,fn,tp=testing(y_test_scores,test_true,optimal_threshold)
		#print("testing results:  mcc->" + str(np.round(mcc,3)) + "  acc->"+str(np.round(acc,3))+ "  auc->"+str(np.round(auc_,3))+ "\n")
		
		#append the data to a csv file that gets either created or updated
		append_data_to_tsv(".","training_predictions.tsv",[str((argv[3].split("/"))[2]),optimal_threshold,prc, rcc,fscore,acc,mcc,auc_,tn,fp,fn,tp])
