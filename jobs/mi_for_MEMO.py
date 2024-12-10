import os

# features    = 'max_corr,t_max,corr_coeff'

window_size = str(0) # in secs
aggs        = 'mean,std,min,max,grad,median'
feature     = 'pitch'
modality    = 'audio'
sub_feature = 'nan'


venv_cmd = 'Path_to_venv/venv/bin/activate'
os.system(venv_cmd)
print('activated venv aer_phd ....')

print("Running script.")

command = 'python feature_extractor/mi_extractor.py '+ \
        '--feature ' + feature + \
        ' --modality ' + modality + \
        ' --agg_types ' + aggs + \
        ' --window_size ' + window_size + \
        ' --sub_feature ' + sub_feature
        
os.system(command)
