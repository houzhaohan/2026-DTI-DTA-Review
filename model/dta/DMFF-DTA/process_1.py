import pandas as pd

#Unnamed: 0,compound_iso_smiles,target_sequence,target_key,affinity,target_sequence_len,target_sequence_start,target_sequence_end,uniprot


data=pd.read_csv('kiba_processed_300_add_range.csv')[['target_key','target_sequence','target_sequence_start','target_sequence_end']]
#compound_iso_smiles,target_sequence,target_key,affinity,target_sequence_start,target_sequence_end
#data1=pd.read_csv('benchmark/kiba_dta/b.csv')['Protein_id']

#data1 = data1.drop_duplicates(subset=['Protein_id'])
data = data.drop_duplicates(subset=['target_key'])
#data.columns=['Protein_id']

#data1=data1.to_list()
#data=data.to_list()

#a=set(data1).difference(set(data))

data.to_csv('b.csv',index=False)
print(len(data))
