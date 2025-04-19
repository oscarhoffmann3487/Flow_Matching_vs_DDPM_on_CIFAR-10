import torch 

class cddpm():
    def __init__(self,
                 num_timesteps=1000,
                 beta_start=0.0001,
                 beta_end=0.02,
                 beta_schedule='linear'
                 ):
        # Initialize parameters
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_timesteps =num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule

        self.betas = self.noiseScheduler().to(self.device)

        self.alphas = 1- self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim = 0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1]).to(self.device), self.alphas_cumprod[:-1]])

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1-self.alphas_cumprod)

        self.posterior_mean_coef1 = (torch.sqrt(self.alphas_cumprod_prev)*self.betas / (1-self.alphas_cumprod))
        self.posterior_mean_coef2 = (torch.sqrt(self.alphas)*(1-self.alphas_cumprod_prev) / (1-self.alphas_cumprod))

        self.reconstruct_x0_coef1 = 1/self.sqrt_alphas_cumprod
        self.reconstruct_x0_coef2 = self.reconstruct_x0_coef1*self.sqrt_one_minus_alphas_cumprod

        self.alphas = self.alphas[:,None,None,None]
        self.alphas_cumprod = self.alphas_cumprod[:,None,None,None]
        self.alphas_cumprod_prev = self.alphas_cumprod_prev[:,None,None,None]
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod[:,None,None,None]
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod[:,None,None,None]
        self.posterior_mean_coef1 = self.posterior_mean_coef1[:,None,None,None]
        self.posterior_mean_coef2 = self.posterior_mean_coef2[:,None,None,None]
        self.reconstruct_x0_coef1 = self.reconstruct_x0_coef1[:,None,None,None]
        self.reconstruct_x0_coef2 = self.reconstruct_x0_coef2[:,None,None,None]

    def noiseScheduler(self):
        betas = torch.linspace(self.beta_start, self.beta_end, self.num_timesteps)
        return betas

    def add_noise(self, x_0, noise, t):
        s1 = self.sqrt_alphas_cumprod[t]
        s2 = self.sqrt_one_minus_alphas_cumprod[t] 
        return s1 * x_0 + s2 * noise

    def q_posterior(self, x_0, x_t, t):
        s1 = self.posterior_mean_coef1[t]
        s2 = self.posterior_mean_coef2[t]
        mu = s1 * x_0 + s2 * x_t
        return mu

    def get_variance(self, t):
        variance = ((1-self.alphas_cumprod_prev[t])/(1-self.alphas_cumprod[t])*self.betas[t])
        return variance

    def reconstruct_x0(self, x_t, t, pred_noise):
        s1 = self.reconstruct_x0_coef1[t]
        s2 = self.reconstruct_x0_coef2[t]
        return s1 * x_t - s2 * pred_noise

    def sample_timesteps(self, n):
        t = torch.randint(low=0, high=self.num_timesteps, size=(n,))
        return t


    def step(self, pred_noise, t, x_t):
     
        noise = torch.randn_like(pred_noise)


        pred_x_0 = self.reconstruct_x0(x_t, t, pred_noise)
        # compute posterior mean
        mean = self.q_posterior(pred_x_0, x_t, t)
        # compute posterior variance (different if t == 0 or if t > 0!)
        if t == 0:
            variance = 0
        if t > 0:
            variance = self.get_variance(t)
        # sample based on posterior and variance
        pred_prev_sample = mean + (variance ** 0.5) * noise

        return pred_prev_sample