gpustat

gpu="0"
yaml="Run_pipeline/config.yaml"


# python MAML.py -p ${yaml} -g ${gpu}
# cp -r Output/MAML_all_w0_30/overfit Output/MAML_w0_10_exp2


python train_from_meta.py -p ${yaml} -g ${gpu}
python train_from_random.py -p ${yaml} -g ${gpu}


# python train_from_meta_classify.py -p ${yaml} -g ${gpu}