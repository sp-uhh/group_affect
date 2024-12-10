import os

# features    = 'max_corr,t_max,corr_coeff'

window_size = str(0) # in secs
aggs        = 'mean,std,min,max,grad,median'
feature     = 'aucs'
modality    = 'video'
sub_feature = 'AU43'
sync_type   = 't_max'


venv_cmd = 'Path_to_venv/venv/bin/activate'
os.system(venv_cmd)
print('activated venv aer_phd ....')

print("Running script.")

command = 'python feature_extractor/synchrony_extractor.py '+ \
        '--feature ' + feature + \
        ' --modality ' + modality + \
        ' --agg_types ' + aggs + \
        ' --window_size ' + window_size + \
        ' --sub_feature ' + sub_feature + \
        ' --sync_type ' + sync_type
        
os.system(command)
