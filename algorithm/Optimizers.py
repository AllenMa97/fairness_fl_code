import torch.optim as optim
import copy


class BERTCLF_Optimizer(object):
    def __init__(self, method, learning_rate, max_grad_norm=0,
                 lr_decay=1, start_decay_steps=None, decay_steps=None,
                 algorithm="FedAvg",
                 beta1=0.9, beta2=0.999,
                 adagrad_accum=0.0,
                 decay_method=None,
                 lamda=0.01,
                 warmup_steps=4000, weight_decay=0, momentum=0, dampening=0., nesterov=False, mu=0.001):
        self.last_ppl = None
        self.algorithm = algorithm
        self.learning_rate = learning_rate
        self.original_lr = learning_rate
        self.max_grad_norm = max_grad_norm
        self.method = method
        self.lr_decay = lr_decay
        self.start_decay_steps = start_decay_steps
        self.decay_steps = decay_steps
        self.start_decay = False
        self.lamda = lamda
        self._step = 0
        self.betas = [beta1, beta2]
        self.adagrad_accum = adagrad_accum
        self.decay_method = decay_method
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.mu = mu
        self.momentum = momentum
        self.nesterov = nesterov
        self.dampening = dampening

    def zero_grad(self):
        self.optimizer.zero_grad()
    def set_parameters(self, params):
        """ ? """
        self.params = []
        self.sparse_params = []
        for k, p in params:
            if p.requires_grad:
                if self.method != 'sparseadam' or "embed" not in k:
                    self.params.append(p)
                else:
                    self.sparse_params.append(p)
        if self.method == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.learning_rate)

        elif self.method == 'adam':
            self.optimizer = optim.Adam(self.params, lr=self.learning_rate,
                                        betas=self.betas, eps=1e-9)

        elif self.method == 'DOSFL_ADAM':
            self.optimizer = optim.Adam(self.params, lr=self.learning_rate,
                                        betas=self.betas, eps=1e-9)

        # elif self.method == 'prox':
        #     self.optimizer = ProxSGD(self.params, lr=self.learning_rate, mu=self.mu,
        #                              momentum=self.momentum, dampening=self.dampening,
        #                              weight_decay=self.weight_decay, nesterov=self.nesterov)
        else:
            raise RuntimeError("Invalid optim method: " + self.method)

        self.optimizer.param_groups[0]["lamda"] = self.lamda
        self.optimizer.param_groups[0]["mu"] = self.mu

    def set_rate(self, learning_rate):
        self.learning_rate = learning_rate
        self.optimizer.param_groups[0]['lr'] = self.learning_rate

    def _set_rate(self, learning_rate):
        self.learning_rate = learning_rate
        self.optimizer.param_groups[0]['lr'] = self.learning_rate

    def step(self, local_weight_updated=None):
        """Update the model parameters based on current gradients.

        Optionally, will employ gradient modification or update learning
        rate.
        """
        if self.method == "adam":
            self._step += 1
            # Decay method used in tensor2tensor.
            self._set_rate(
                self.original_lr *
                min(self._step ** (-0.5),
                    self._step * self.warmup_steps ** (-1.5)))

            self.optimizer.param_groups[0]['lr'] = self.learning_rate

        if self.method == "DOSFL_ADAM":
            self._step += 1
            if (self._step % self.decay_steps == 0) and (self._step != 0):
                self._set_rate(
                    self.original_lr * 0.5)

                self.optimizer.param_groups[0]['lr'] = self.learning_rate

        if self.algorithm == "pFedMe":
            if local_weight_updated == None:
                raise ValueError("If current algorithm is pFedMe,you should pass the model parameters")
            weight_update = copy.deepcopy(local_weight_updated)

            for group in self.optimizer.param_groups:
                for p, localweight in zip(group['params'], weight_update):
                    try:
                        gradient_data = p.grad.data
                    except AttributeError:
                        gradient_data = 0

                    p.data = p.data - group['lr'] * (
                            gradient_data + group['lamda'] * (p.data - localweight.data) + group['mu'] * p.data)

            return group['params']

        self.optimizer.step()


class Scaffold_Optimizer(optim.Optimizer):
    def __init__(self,params,method,learning_rate,max_grad_norm,last_ppl=None,lr_decay=1,
                 start_decay_steps=None,decay_steps=None,start_decay=False,_step=0,adagrad_accum=0.0,decay_method=None,
                 warmup_steps=4000,weight_decay=0,momentum=0,nesterov=False,dampening=0, beta1=0.9, beta2=0.999):
        super(Scaffold_Optimizer, self).__init__(params, {})
        self.params=params
        self.last_ppl = last_ppl
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.method = method
        self.lr_decay = lr_decay
        self.start_decay_steps = start_decay_steps
        self.decay_steps = decay_steps
        self.start_decay = start_decay
        self._step = _step
        self.adagrad_accum = adagrad_accum
        self.decay_method = decay_method
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.nesterov = nesterov
        self.dampening = dampening
        self.betas = [beta1, beta2]

    def set_parameters(self, params):
        """ ? """
        self.params = []
        self.sparse_params = []
        for k, p in params:
            if p.requires_grad:
                if self.method != 'sparseadam' or "embed" not in k:
                    self.params.append(p)
                else:
                    self.sparse_params.append(p)
        if self.method == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.learning_rate)

        elif self.method == 'adam':
            self.optimizer = optim.Adam(self.params, lr=self.learning_rate,
                                        betas=self.betas, eps=1e-9)
        else:
            raise RuntimeError("Invalid optim method: " + self.method)



    def step(self,device, server_controls, client_controls, closure=False):
        """Update the model parameters based on current gradients.

        Optionally, will employ gradient modification or update learning
        rate.
        """
        # self._step += 1


        for group in self.param_groups:
            for p,c,ci in zip(group['params'],server_controls.values(),client_controls.values()):
                if p.grad is None:
                    continue
                #本地更新
                #y_i=y_i - lr * (g(y_i) + c - ci)
                #p表示y_i,即本地模型的参数
                # c.data = c.data.to("cuda:0")
                # ci.data = ci.data.to("cuda:0")
                c.data = c.data.to(device)
                ci.data = ci.data.to(device)
                dp=p.grad.data + c.data - ci.data
                p.data=p.data - dp.data * self.learning_rate

                c.data = c.data.cpu()
                ci.data = ci.data.cpu()
                
        # #使用pytorch内置的step函数进行参数的更新
        # self.optimizer.step()

class Ditto_local_Optimizer(optim.Optimizer):
    def __init__(self,params,method,learning_rate,max_grad_norm,ditto_lambda,last_ppl=None,lr_decay=1,
                 start_decay_steps=None,decay_steps=None,start_decay=False,_step=0,adagrad_accum=0.0,decay_method=None,
                 warmup_steps=4000,weight_decay=0,momentum=0,nesterov=False,dampening=0, beta1=0.9, beta2=0.999, mu = 0):
        self.params=params
        self.last_ppl = None
        self.learning_rate = learning_rate
        self.original_lr = learning_rate
        self.max_grad_norm = max_grad_norm
        self.lr_decay = lr_decay
        self.start_decay_steps = start_decay_steps
        self.decay_steps = decay_steps
        self.start_decay = False
        self._step = 0
        self.betas = [beta1, beta2]
        self.adagrad_accum = adagrad_accum
        self.decay_method = decay_method
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.mu = mu
        self.momentum = momentum
        self.nesterov = nesterov
        self.dampening = dampening
        self.ditto_lambda=ditto_lambda
        self.method = method
        super(Ditto_local_Optimizer,self).__init__(params, {})


    def step(self, updated_global_model_params, closure=None):
        """Update the model parameters based on current gradients.

        Optionally, will employ gradient modification or update learning
        rate.
        """
        self._step += 1

        
        for group in self.param_groups:
            for p, g  in zip(group['params'], updated_global_model_params):
                if p.grad is None:
                    continue;
                # v_k = v_k - η(grad(v_k) + λ(v_K - w^t))
                p.data = p.data - self.learning_rate * (p.grad.data + self.ditto_lambda * (p.data - g.data))

