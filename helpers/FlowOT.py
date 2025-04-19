import torch

class FlowOT():
    def __init__(self, sig_min = 0.0001):
        #initiation
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.sig_min = sig_min

    def psi_t(self, t, x, x_1):
        #conditional flow (21)
        return (1 - (1-self.sig_min) * t ) * x + t * x_1

    def sample_time(self, x_1):
        # Sample time uniformly from [0, 1]
        t = torch.rand((x_1.size(0), ), device = self.device)
        return t.view(-1, 1, 1, 1)

    def sample_noise(self, x_1):
        #sample noise from target
        return torch.randn_like(x_1, device = self.device)

    def loss(self, v_t_psi, x_0, x_1):
        #objective function (23)
        u_t = x_1 - (1 - self.sig_min) * x_0
        return torch.mean((v_t_psi - u_t) ** 2)