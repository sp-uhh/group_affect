import os
# "head_pitch", "head_roll", "head_yaw"
# ,AU01,AU02,AU04,AU05,AU06,AU07,AU09,AU10,AU11,AU12,AU14,AU15,AU17,AU20,AU23,AU24,AU25,AU26,AU28,AU43

window_size = str(0) # in secs
aggs        = 'mean,std,min,max,grad,median'
conv_type   = 'asymmetric'
modality    = 'video'
feature     = 'aucs'
sub_feature = 'AU43'

venv_cmd = 'Path_to_venv/venv/bin/activate'
os.system(venv_cmd)
print('activated venv aer_phd ....')

print("Running script.")

command = 'python feature_extractor/convergence_extractor.py '+ \
        ' --type ' + conv_type + \
        ' --feature ' + feature + \
        ' --modality ' + modality + \
        ' --agg_types ' + aggs + \
        ' --window_size ' + window_size + \
        ' --sub_feature ' + sub_feature
        
os.system(command)
