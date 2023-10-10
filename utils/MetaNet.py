import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np

class Hypernetwork(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features):
        super(Hypernetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, out_features)
        )

    def forward(self, input):
        return self.net(input)
    
class LEO(nn.Module):
    def __init__(self, num_meta_steps, hypo_module, loss, init_lr,
                 lr_type='static', first_order=False):
        super().__init__()

        self.hypo_module = hypo_module  # The module whose weights we want to meta-learn.
        self.first_order = first_order
        self.loss = loss
        self.lr_type = lr_type
        self.num_meta_steps = num_meta_steps

        # Define the hypernetwork to generate parameters
        self.hypernetwork = Hypernetwork()  # Replace with your hypernetwork architecture

        param_count = 0
        for param in self.parameters():
            param_count += torch.prod(torch.tensor(param.shape))

        print(param_count)

    def _update_step(self, loss, param_dict, step):
        grads = torch.autograd.grad(loss, param_dict.values(),
                                    create_graph=False if self.first_order else True)
        params = OrderedDict()
        for i, ((name, param), grad) in enumerate(zip(param_dict.items(), grads)):
            if self.lr_type in ['static', 'global']:
                lr = self.lr
                params[name] = param - lr * grad
            elif self.lr_type in ['per_step']:
                lr = self.lr[step]
                params[name] = param - lr * grad
            elif self.lr_type in ['per_parameter']:
                lr = self.lr[i]
                params[name] = param - lr * grad
            elif self.lr_type in ['per_parameter_per_step']:
                lr = self.lr[i][step]
                params[name] = param - lr * grad
            else:
                raise NotImplementedError

        return params, grads

    def forward_with_params(self, query_x, fast_params, **kwargs):
        output = self.hypo_module(query_x, params=fast_params)
        return output

    def generate_params(self, sampled_coords, sampled_data):
        """Specializes the model"""
        x = sampled_coords
        y = sampled_data

        meta_batch_size = x.shape[0]

        # Generate parameters using the hypernetwork
        hyper_params = self.hypernetwork(meta_batch_size)  # Replace with your hypernetwork call

        with torch.enable_grad():
            # Initialize fast parameters
            fast_params = OrderedDict(self.hypo_module.named_parameters())

            prev_loss = 1e6
            intermed_predictions = []
            for j in range(self.num_meta_steps):
                # Using the current set of parameters, perform a forward pass with the context inputs.
                predictions = self.hypo_module(x, params=fast_params)

                # Compute the loss on the context labels.
                loss = self.loss(predictions, y)
                intermed_predictions.append(predictions)

                if float(loss.item()) > prev_loss:
                    print('inner lr too high?')

                # Using the computed loss, update the fast parameters.
                fast_params, grads = self._update_step(loss, fast_params, j)
                prev_loss = loss

        return fast_params, intermed_predictions

    def forward(self, sampled_coords, sampled_data, **kwargs):
        # The meta_batch consists of the "context" set (the observations we're conditioning on)
        # and the "query" inputs (the points where we want to evaluate the specialized model)

        # Specialize the model with the "generate_params" function.
        fast_params, intermed_predictions = self.generate_params(sampled_coords, sampled_data)

        # Compute the final outputs.
        model_output = self.hypo_module(sampled_coords, params=fast_params)
        out_dict = {'model_out': model_output, 'intermed_predictions': intermed_predictions}

        return out_dict

class MAML(nn.Module):
    def __init__(self, num_meta_steps, hypo_module, loss, init_lr,
                 lr_type='static', first_order=False):
        super().__init__()

        self.hypo_module = hypo_module # The module who's weights we want to meta-learn.
        self.first_order = first_order
        self.loss = loss
        self.lr_type = lr_type
        self.log = []

        self.register_buffer('num_meta_steps', torch.Tensor([num_meta_steps]).int())

        if self.lr_type == 'static': 
            self.register_buffer('lr', torch.Tensor([init_lr]))
        elif self.lr_type == 'global':
            self.lr = nn.Parameter(torch.Tensor([init_lr]))
        elif self.lr_type == 'per_step':
            self.lr = nn.ParameterList([nn.Parameter(torch.Tensor([init_lr]))
                                        for _ in range(num_meta_steps)])
        elif self.lr_type == 'per_parameter': # As proposed in "Meta-SGD".
            self.lr = nn.ParameterList([])
            hypo_parameters = hypo_module.parameters()
            for param in hypo_parameters:
                self.lr.append(nn.Parameter(torch.ones(param.size()) * init_lr))
        elif self.lr_type == 'per_parameter_per_step':
            self.lr = nn.ModuleList([])
            for name, param in hypo_module.meta_named_parameters():
                self.lr.append(nn.ParameterList([nn.Parameter(torch.ones(param.size()) * init_lr)
                                                 for _ in range(num_meta_steps)]))

        param_count = 0
        for param in self.parameters():
            param_count += np.prod(param.shape)

        print(param_count)

    def _update_step(self, loss, param_dict, step):
        grads = torch.autograd.grad(loss, param_dict.values(),
                                    create_graph=False if self.first_order else True)
        params = OrderedDict()
        for i, ((name, param), grad) in enumerate(zip(param_dict.items(), grads)):
            if self.lr_type in ['static', 'global']:
                lr = self.lr
                params[name] = param - lr * grad
            elif self.lr_type in ['per_step']:
                lr = self.lr[step]
                params[name] = param - lr * grad
            elif self.lr_type in ['per_parameter']:
                lr = self.lr[i]
                params[name] = param - lr * grad
            elif self.lr_type in ['per_parameter_per_step']:
                lr = self.lr[i][step]
                params[name] = param - lr * grad
            else:
                raise NotImplementedError

        return params, grads

    def forward_with_params(self, query_x, fast_params, **kwargs):
        output = self.hypo_module(query_x, params=fast_params)
        return output

    def generate_params(self, sampled_coords,sampled_data):
        """Specializes the model"""
        x = sampled_coords
        y = sampled_data

        meta_batch_size = x.shape[0]

        with torch.enable_grad():
            # First, replicate the initialization for each batch item.
            # This is the learned initialization, i.e., in the outer loop,
            # the gradients are backpropagated all the way into the 
            # "meta_named_parameters" of the hypo_module.
            fast_params = OrderedDict()
            for name, param in self.hypo_module.meta_named_parameters():
                fast_params[name] = param[None, ...].repeat((meta_batch_size,) + (1,) * len(param.shape))

            prev_loss = 1e6
            intermed_predictions = []
            for j in range(self.num_meta_steps):
                # Using the current set of parameters, perform a forward pass with the context inputs.
                predictions = self.hypo_module(x, params=fast_params)

                # Compute the loss on the context labels.
                loss = self.loss(predictions, y)
                intermed_predictions.append(predictions)

                if float(loss.item()) > prev_loss:
                    print('inner lr too high?')
                
                # Using the computed loss, update the fast parameters.
                fast_params, grads = self._update_step(loss, fast_params, j)
                prev_loss = loss

        return fast_params, intermed_predictions

    def forward(self, sampled_coords,sampled_data, **kwargs):
        # The meta_batch conists of the "context" set (the observations we're conditioning on)
        # and the "query" inputs (the points where we want to evaluate the specialized model)

        # Specialize the model with the "generate_params" function.
        fast_params, intermed_predictions = self.generate_params(sampled_coords,sampled_data)

        # Compute the final outputs.
        model_output = self.hypo_module(sampled_coords, params=fast_params)
        out_dict = {'model_out':model_output, 'intermed_predictions':intermed_predictions}

        return out_dict

class Reptile(nn.Module):
    def __init__(self, num_meta_steps, hypo_module, loss, init_lr,
                 lr_type='static', first_order=False):
        super().__init__()

        self.hypo_module = hypo_module  # The module whose weights we want to meta-learn.
        self.num_meta_steps = num_meta_steps
        self.loss = loss
        self.init_lr = init_lr
        self.lr_type = lr_type
        self.first_order = first_order

    def _update_step(self, loss, param_dict, lr):
        grads = torch.autograd.grad(loss, param_dict.values(),
                                    create_graph=not self.first_order)
        updated_params = OrderedDict()
        for (name, param), grad in zip(param_dict.items(), grads):
            updated_params[name] = param - lr * grad
        return updated_params

    def update_params(self, sampled_coords, sampled_data):
        x = sampled_coords
        y = sampled_data

        # Initialize model parameters
        fast_params = OrderedDict(self.hypo_module.named_parameters())

        for step in range(self.num_meta_steps):
            # Forward pass with current parameters
            predictions = self.hypo_module(x, params=fast_params)



class RIRL(nn.Module):

    def __init__(self, num_meta_steps, hypo_module, task_loss, init_lr, state_dim, lambda_=1.0):
        super().__init__()
        self.hypo_module = hypo_module
        self.task_loss = task_loss
        self.num_meta_steps = num_meta_steps
        self.state_dim = state_dim
        self.lamba_ = lambda_
        # 定义内在奖励模型
        self.irm = nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.lr = init_lr
        
    def generate_params(self, sampled_states, sampled_rewards):
      
        # 计算内在奖励 
        intrinsic_rewards = self.irm(sampled_states)
        
        params = OrderedDict(self.hypo_module.named_parameters())
        
        for i in range(self.num_meta_steps):
                        
            predictions = self.hypo_module(sampled_states, params=params)
            
            task_loss = self.task_loss(predictions, sampled_rewards) 
            irm_loss = intrinsic_rewards.mean()
            loss = task_loss + self.lamba_*irm_loss
            
            grads = torch.autograd.grad(loss, params.values())
            params = OrderedDict(
                (name, param - self.lr*grad) for ((name,param),grad) 
                in zip(params.items(), grads)
            )
            
            irm_loss.backward() # 反向传播更新内在奖励模型
            
        return params
        
    def forward(self, states, rewards):
      
        params = self.generate_params(states, rewards)
        predictions = self.hypo_module(states, params=params)

        return predictions

class ANIL(nn.Module):
    def __init__(self, num_inner_steps, hypo_module, loss, init_lr,
                 lr_type='static', first_order=False):
        super().__init__()

        self.hypo_module = hypo_module  # The module whose weights we want to meta-learn.
        self.first_order = first_order
        self.loss = loss
        self.lr_type = lr_type
        self.log = []

        self.num_inner_steps = num_inner_steps

        if self.lr_type == 'static':
            self.register_buffer('lr', torch.Tensor([init_lr]))
        elif self.lr_type == 'global':
            self.lr = nn.Parameter(torch.Tensor([init_lr]))
        elif self.lr_type == 'per_step':
            self.lr = nn.ParameterList([nn.Parameter(torch.Tensor([init_lr]))
                                        for _ in range(num_inner_steps)])
        elif self.lr_type == 'per_parameter':  # As proposed in "Meta-SGD".
            self.lr = nn.ParameterList([])
            hypo_parameters = hypo_module.parameters()
            for param in hypo_parameters:
                self.lr.append(nn.Parameter(torch.ones(param.size()) * init_lr))
        elif self.lr_type == 'per_parameter_per_step':
            self.lr = nn.ModuleList([])
            for name, param in hypo_module.named_parameters():
                self.lr.append(nn.ParameterList([nn.Parameter(torch.ones(param.size()) * init_lr)
                                                 for _ in range(num_inner_steps)]))

        param_count = 0
        for param in self.parameters():
            param_count += torch.prod(torch.tensor(param.shape))

        print(param_count)

    def _update_step(self, loss, param_dict, step):
        grads = torch.autograd.grad(loss, param_dict.values(),
                                    create_graph=False if self.first_order else True)
        params = OrderedDict()
        for i, ((name, param), grad) in enumerate(zip(param_dict.items(), grads)):
            if self.lr_type in ['static', 'global']:
                lr = self.lr
                params[name] = param - lr * grad
            elif self.lr_type in ['per_step']:
                lr = self.lr[step]
                params[name] = param - lr * grad
            elif self.lr_type in ['per_parameter']:
                lr = self.lr[i]
                params[name] = param - lr * grad
            elif self.lr_type in ['per_parameter_per_step']:
                lr = self.lr[i][step]
                params[name] = param - lr * grad
            else:
                raise NotImplementedError

        return params, grads

    def forward_with_params(self, query_x, fast_params, **kwargs):
        output = self.hypo_module(query_x, params=fast_params)
        return output

    def generate_params(self, context_x, context_y):
        """Specializes the model"""
        x = context_x
        y = context_y

        meta_batch_size = x.shape[0]

        with torch.enable_grad():
            # First, replicate the initialization for each batch item.
            # This is the learned initialization, i.e., in the outer loop,
            # the gradients are backpropagated all the way into the
            # "meta_named_parameters" of the hypo_module.
            fast_params = OrderedDict()
            for name, param in self.hypo_module.named_parameters():
                fast_params[name] = param[None, ...].repeat((meta_batch_size,) + (1,) * len(param.shape))

            prev_loss = 1e6
            intermed_predictions = []
            for j in range(self.num_inner_steps):
                # Using the current set of parameters, perform a forward pass with the context inputs.
                predictions = self.hypo_module(x, params=fast_params)

                # Compute the loss on the context labels.
                loss = self.loss(predictions, y)
                intermed_predictions.append(predictions)

                if float(loss.item()) > prev_loss:
                    print('inner lr too high?')

                # Using the computed loss, update the fast parameters.
                fast_params, grads = self._update_step(loss, fast_params, j)
                prev_loss = loss

        return fast_params, intermed_predictions

    def forward(self, context_x, context_y, query_x, **kwargs):
        # The meta_batch consists of the "context" set (the observations we're conditioning on)
        # and the "query" inputs (the points where we want to evaluate the specialized model)

        # Specialize the model with the "generate_params" function.
        fast_params, intermed_predictions = self.generate_params(context_x, context_y)

        # Compute the final outputs.
        model_output = self.hypo_module(query_x, params=fast_params)
        out_dict = {'model_out': model_output, 'intermed_predictions': intermed_predictions}

        return out_dict


class MetaSGD(nn.Module):
    def __init__(self, num_meta_steps, hypo_module, loss, init_lr,
                 lr_type='static', first_order=False):
        super().__init__()

        self.hypo_module = hypo_module  # The module who's weights we want to meta-learn.
        self.first_order = first_order
        self.loss = loss
        self.lr_type = lr_type
        self.log = []

        self.register_buffer('num_meta_steps', torch.Tensor([num_meta_steps]).int())

        if self.lr_type == 'static':
            self.register_buffer('lr', torch.Tensor([init_lr]))
        elif self.lr_type == 'global':
            self.lr = nn.Parameter(torch.Tensor([init_lr]))
        elif self.lr_type == 'per_step':
            self.lr = nn.ParameterList([nn.Parameter(torch.Tensor([init_lr]))
                                        for _ in range(num_meta_steps)])
        elif self.lr_type == 'per_parameter':
            self.lr = nn.ParameterList([])
            hypo_parameters = hypo_module.parameters()
            for param in hypo_parameters:
                self.lr.append(nn.Parameter(torch.ones(param.size()) * init_lr))
        elif self.lr_type == 'per_parameter_per_step':
            self.lr = nn.ModuleList([])
            for name, param in hypo_module.meta_named_parameters():
                self.lr.append(nn.ParameterList([nn.Parameter(torch.ones(param.size()) * init_lr)
                                                 for _ in range(num_meta_steps)]))
        else:
            raise NotImplementedError

        param_count = 0
        for param in self.parameters():
            param_count += np.prod(param.shape)

        print(param_count)

    def _update_step(self, loss, param_dict, step):
        grads = torch.autograd.grad(loss, param_dict.values(),
                                    create_graph=False if self.first_order else True)
        params = OrderedDict()
        for i, ((name, param), grad) in enumerate(zip(param_dict.items(), grads)):
            if self.lr_type in ['static', 'global']:
                lr = self.lr
                params[name] = param - lr * grad
            elif self.lr_type in ['per_step']:
                lr = self.lr[step]
                params[name] = param - lr * grad
            elif self.lr_type in ['per_parameter']:
                lr = self.lr[i]
                params[name] = param - lr * grad
            elif self.lr_type in ['per_parameter_per_step']:
                lr = self.lr[i][step]
                params[name] = param - lr * grad
            else:
                raise NotImplementedError

        return params, grads

    def forward_with_params(self, query_x, fast_params, **kwargs):
        output = self.hypo_module(query_x, params=fast_params)
        return output

    def generate_params(self, sampled_coords, sampled_data):
        """Specializes the model"""
        x = sampled_coords
        y = sampled_data

        meta_batch_size = x.shape[0]

        with torch.enable_grad():
            # First, replicate the initialization for each batch item.
            # This is the learned initialization, i.e., in the outer loop,
            # the gradients are backpropagated all the way into the
            # "meta_named_parameters" of the hypo_module.
            fast_params = OrderedDict()
            for name, param in self.hypo_module.meta_named_parameters():
                fast_params[name] = param[None, ...].repeat((meta_batch_size,) + (1,) * len(param.shape))

            prev_loss = 1e6
            intermed_predictions = []
            for j in range(self.num_meta_steps):
                # Using the current set of parameters, perform a forward pass with the context inputs.
                predictions = self.hypo_module(x, params=fast_params)

                # Compute the loss on the context labels.
                loss = self.loss(predictions, y)
                intermed_predictions.append(predictions)

                if float(loss.item()) > prev_loss:
                    print('inner lr too high?')

                # Using the computed loss, update the fast parameters.
                fast_params, grads = self._update_step(loss, fast_params, j)
                prev_loss = loss

        return fast_params, intermed_predictions

    def forward(self, sampled_coords, sampled_data, **kwargs):
        # The meta_batch consists of the "context" set (the observations we're conditioning on)
        # and the "query" inputs (the points where we want to evaluate the specialized model)

        # Specialize the model with the "generate_params" function.
        fast_params, intermed_predictions = self.generate_params(sampled_coords, sampled_data)

        # Compute the final outputs.
        model_output = self.hypo_module(sampled_coords, params=fast_params)
        out_dict = {'model_out': model_output, 'intermed_predictions': intermed_predictions}

        return out_dict

class MAMLWithSSL(nn.Module):
    def __init__(self, num_meta_steps, hypo_module, task_loss, ssl_loss, init_lr,
                 lr_type='static', first_order=False):
        super().__init__()

        self.hypo_module = hypo_module
        self.first_order = first_order
        self.task_loss = task_loss  # Primary task loss
        self.ssl_loss = ssl_loss    # Self-supervised task loss
        self.lr_type = lr_type
        self.log = []

        self.register_buffer('num_meta_steps', torch.Tensor([num_meta_steps]).int())

        if self.lr_type == 'static':
            self.register_buffer('lr', torch.Tensor([init_lr]))
        elif self.lr_type == 'global':
            self.lr = nn.Parameter(torch.Tensor([init_lr]))
        elif self.lr_type == 'per_step':
            self.lr = nn.ParameterList([nn.Parameter(torch.Tensor([init_lr]))
                                        for _ in range(num_meta_steps)])
        elif self.lr_type == 'per_parameter':
            self.lr = nn.ParameterList([])
            hypo_parameters = hypo_module.parameters()
            for param in hypo_parameters:
                self.lr.append(nn.Parameter(torch.ones(param.size()) * init_lr))
        elif self.lr_type == 'per_parameter_per_step':
            self.lr = nn.ModuleList([])
            for name, param in hypo_module.meta_named_parameters():
                self.lr.append(nn.ParameterList([nn.Parameter(torch.ones(param.size()) * init_lr)
                                                 for _ in range(num_meta_steps)]))
        else:
            raise NotImplementedError

        param_count = 0
        for param in self.parameters():
            param_count += torch.prod(torch.tensor(param.shape)).item()

        print(param_count)

    def _update_step(self, loss, param_dict, step):
        grads = torch.autograd.grad(loss, param_dict.values(),
                                    create_graph=False if self.first_order else True)
        params = OrderedDict()
        for i, ((name, param), grad) in enumerate(zip(param_dict.items(), grads)):
            if self.lr_type in ['static', 'global']:
                lr = self.lr
                params[name] = param - lr * grad
            elif self.lr_type in ['per_step']:
                lr = self.lr[step]
                params[name] = param - lr * grad
            elif self.lr_type in ['per_parameter']:
                lr = self.lr[i]
                params[name] = param - lr * grad
            elif self.lr_type in ['per_parameter_per_step']:
                lr = self.lr[i][step]
                params[name] = param - lr * grad
            else:
                raise NotImplementedError

        return params, grads

    def forward_with_params(self, query_x, fast_params, **kwargs):
        output = self.hypo_module(query_x, params=fast_params)
        return output

    def generate_params(self, sampled_coords, sampled_data):
        x = sampled_coords
        y = sampled_data

        meta_batch_size = x.shape[0]

        with torch.enable_grad():
            fast_params = OrderedDict()
            for name, param in self.hypo_module.meta_named_parameters():
                fast_params[name] = param[None, ...].repeat((meta_batch_size,) + (1,) * len(param.shape))

            prev_loss = 1e6
            intermed_predictions = []
            for j in range(self.num_meta_steps):
                predictions = self.hypo_module(x, params=fast_params)
                task_loss = self.task_loss(predictions, y)

                # Self-supervised learning: Add SSL loss during inner optimization
                # You can define an appropriate SSL task loss and use it here
                ssl_loss = self.ssl_loss(fast_params, j)

                # Combine task loss and SSL loss
                loss = task_loss + ssl_loss
                intermed_predictions.append(predictions)

                if float(loss.item()) > prev_loss:
                    print('inner lr too high?')

                fast_params, grads = self._update_step(loss, fast_params, j)
                prev_loss = loss

        return fast_params, intermed_predictions

    def forward(self, sampled_coords, sampled_data, **kwargs):
        fast_params, intermed_predictions = self.generate_params(sampled_coords, sampled_data)

        model_output = self.hypo_module(sampled_coords, params=fast_params)
        out_dict = {'model_out': model_output, 'intermed_predictions': intermed_predictions}

        return out_dict