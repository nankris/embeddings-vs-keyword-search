from run import data,lexical,save,R
import platform
save(R/'bm25_environment.json',{'platform':platform.platform(),'processor':platform.processor(),'threads':2,'location':'local sandbox CPU'})
for ds in ['scifact','fiqa','nfcorpus','arguana']:lexical(ds,data(ds))
