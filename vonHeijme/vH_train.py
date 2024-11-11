if __name__=="__main__":
	from VH_lib.vonheijne import PSPM, PSWM, make_cleavage_site
	from sys import argv
	import numpy as np
	if len(argv)!=5:
		print("The command takes as input three tsv files and a text file that will be the output")
	else:
        
		print("processing as training set: "+str((argv[1].split("/"))[2])+" "+str((argv[2].split("/"))[2])+" "+str((argv[3].split("/"))[2]))
		
		#open the tsv files
		cross_val1=open(argv[1],"r")
		cross_val2=open(argv[2],"r")
		cross_val3=open(argv[3],"r")
		
		#retrieve lists of sequences with the same length (the cleavage site of the SP)
		training_set1=make_cleavage_site(cross_val1)
		training_set2=make_cleavage_site(cross_val2)
		training_set3=make_cleavage_site(cross_val3)
		
		#put all in a list
		training_set=training_set1+training_set2+training_set3
		cross_val3.close()
		cross_val2.close()
		cross_val1.close()

		#calcuate pspm and pswm
		pspm=PSPM(training_set)
		pswm=PSWM(pspm) 
		
		#save the W matrix as the name specified in the command line
		np.savetxt("./PSWMs/%s"%(argv[4]),pswm)

		