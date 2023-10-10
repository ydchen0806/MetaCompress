gpustat

gpu="3"
yaml="Run_pipeline/train_data.yaml"


python MAML.py -p ${yaml} -g ${gpu}


python train_data.py -p ${yaml} -g ${gpu}