import sys
from utils.misc import omegaconf2dict, omegaconf2list
from omegaconf import OmegaConf
import os
from os.path import join as opj
from os.path import dirname as opd
from typing import Dict, Union
import time
from torch.utils.tensorboard import SummaryWriter
timestamp = time.strftime("_%Y_%m%d_%H%M%S")
class MyLogger():
    def __init__(self,logdir:str,project_name:str,task_name:str,stdlog:bool=True,tensorboard:bool=True):
        self.project_name = project_name
        self.task_name = task_name
        self.stdlog = stdlog
        self.tensorboard = tensorboard
        if logdir != 'none':
            self.logdir = logdir
        else:
            self.logdir = opj(opd(opd(__file__)),'outputs',project_name,task_name)
        if os.path.exists(self.logdir):
            self.logdir = opj(opd(opd(__file__)),'outputs',project_name+timestamp,task_name)
        os.makedirs(self.logdir,exist_ok=False)
        self.logger_dict:Dict[str,Union[SummaryWriter]] = {}
        if stdlog:
            self.stdlog_init()
        if tensorboard:
            self.tensorboard_init()
    def stdlog_init(self):
        # stdout_handler=open(opj(self.logdir,'stdout.log'), 'w')
        # sys.stdout=stdout_handler
        stderr_handler=open(opj(self.logdir,'stderr.log'), 'w')
        sys.stderr=stderr_handler
    def tensorboard_init(self,):
        self.tblogger = SummaryWriter(self.logdir)
        self.logger_dict['tblogger']=self.tblogger
    def log_opt(self,opt):
        OmegaConf.save(config=opt, f=opj(self.logdir,'opt.yaml'))
        opt_log = omegaconf2list(opt,sep='/')
        for logger_name in self.logger_dict.keys():
            if logger_name == 'tblogger':
                for idx,opt in enumerate(opt_log):
                    self.logger_dict[logger_name].add_text('hparam',opt,idx)    
    def log_metrics(self,metrics_dict: Dict[str,float],iters):
        for logger_name in self.logger_dict.keys():
            if logger_name == 'csvlogger':
                self.logger_dict[logger_name].log_metrics(metrics_dict,iters)
                self.logger_dict[logger_name].save()
            elif logger_name == 'clearml_logger':
                for k in metrics_dict.keys():
                    self.logger_dict[logger_name].report_scalar(k,k,metrics_dict[k],iters)
            elif logger_name == 'tblogger':
                for k in metrics_dict.keys():
                    self.logger_dict[logger_name].add_scalar(k,metrics_dict[k],iters)
    def close(self):
        for logger_name in self.logger_dict.keys():
            if logger_name == 'tblogger':
                self.logger_dict[logger_name].close()