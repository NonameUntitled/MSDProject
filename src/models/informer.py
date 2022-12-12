from models.base import TorchModel


class Informer(TorchModel, config_name='informer'):
    pass


class LongTermAttention(nn.Module):
    # main class to compute continuous attention, with unbounded memory and sticky memories
    # like all 3rd part of article is in this class
    def __init__(self, 
                 head_size: int, 
                 length: int, 
                 target_len: int,  
                 attn_func: str,  # only "softmax"
                 attn_num_basis: int, # number of basis functions
                 attn_drop: float, 
                 infinite_memory: bool, # True
                 n_heads: int, 
                 d_model: int, # embeding size
                 mask: bool, 
                 mask_type: str, # "cnn"
                 kl_regularizer, 
                 sigma_0, 
                 mu_0, 
                 **kwargs):

        super(LongTermAttention, self).__init__()

        #==============INITIALIZING===================

        self.device = 'cuda'

        self.infinite_memory = infinite_memory # whether the memory is infinite
        self.length = length                   # memory length
        self.target_len = target_len           # target length / transformer length
        self.head_size = head_size
        self.attn_num_basis = attn_num_basis   # N - num of basis functions
        self.attn_func = attn_func             # normalizing function
        self.n_head = n_heads                  # number of heads

        self.nb_samples = 512                  # number of samples used for update # TODO: why?
        self.tau = 0.75                        # compressing factor # TODO: why?
        self.ridge_penalty = 1                 # ridge penalty
        self.spacing = 'linear'                # how to get t from [0,1]

        self.mask = mask
        self.mask_type = mask_type
        self.kl_regularizer = kl_regularizer
        self.sigma_0 = sigma_0
        self.mu_0 = mu_0
        
        self.proj_query = nn.Linear(n_heads * head_size, n_heads * head_size, bias=False)
        self.proj_key   = nn.Linear(n_heads * head_size, n_heads * head_size, bias=False)
        self.proj_value = nn.Linear(n_heads * head_size, n_heads * head_size, bias=False)

        self.x_past = None # previous memory vectors
        self.B_past = None # previous coefficient matrix

        padding = True 

        self.attn_dropout = nn.Dropout(attn_drop)
        self.attn_out = nn.Linear(n_heads * head_size, d_model, bias=False)
        self.mu = nn.Linear(attn_num_basis, 1, bias=False)
        self.sigma = nn.Linear(attn_num_basis, 1, bias=False)
        self.softplus = torch.nn.Softplus()

        if self.mask_type == 'cnn':
            # see cnn smoothing (see 3.4)
            self.mask_net = torch.nn.Conv1d(n_heads * head_size, n_heads * head_size, 3, padding=1)

        # normalizing function
        if attn_func == 'softmax':
            self.transform = ContinuousSoftmax(psi=None)
        else:
            assert False

        #================== G matrix computation ======================

        def compute_G(l, psi, positions, padding=True):
            # G = X.T @ F.T @ (F @ F.T + ridge_penalty * I)^(-1)
            # where F = [psi(t_1), ..., psi(t_L)], psi - vector of N RBFs
            # l - length of sequence
            # positions - t

            F = torch.zeros(self.attn_num_basis, positions.size(0))

            basis_functions = psi
            F[:, :] = basis_functions.evaluate(positions.unsqueeze(1)).t()

            I = torch.eye(self.attn_num_basis)
            G = F.t().matmul((F.matmul(F.t()) + self.ridge_penalty * I).inverse()) # actual formula

            if padding: # TODO for what?
                if l % 2:
                    G = G[((l-1)//2):(-(l-1)//2), :]
                else:
                    G = G[(l//2):-(l//2), :]

            return G.to(self.device)

        # ================ POSITIONS, PSIs and Gs ====================
        # get basis functions psi
        sigmas = [.01,.05] # basis function sigmas # TODO why?? heuristic??
        if attn_num_basis % len(sigmas): # TODO why??
            attn_num_basis += (len(sigmas) - attn_num_basis % len(sigmas)) # TODO why??

        self.psi = [None]
        self.Gs  = [None for _ in range(length+1)]
        self.psi = [None]
        lengths  = []

        for i in range(length): # length - memory length 
            self.psi.append([])
            if (i+1) % target_len == 0:
                lengths.append(i+1)
        if length not in lengths:
            lengths.append(length)
            
        for l in lengths:
            # get positions for memory vectors
            self.add_gaussian_basis_functions(self.psi[l], attn_num_basis, sigmas, device=self.device)

            if self.spacing == 'linear': # there was also for "log" but I don't think that we need it
                if padding:
                    if l % 2:
                        shift = 1 / float(l)
                        positions = torch.linspace(-0.5 + shift, 1.5 -shift, 2 * l - 1).to(self.device)
                    else:
                        shift = 1 / float(2*l)
                        positions = torch.linspace(-0.5 + shift, 1.5 - shift, 2 * l).to(self.device)
                else:
                    shift = 1 / float(2*l)
                    positions = torch.linspace(shift, 1-shift, l).to(self.device)
        
            # compute basis functions
            self.Gs[l] = compute_G(l, self.psi[l][0], positions, padding=padding) # [L,N]
            self.positions = positions[int(l/2):-int(l/2)]

        #=============== WORK WITH INF MEMORY ==================
        # compute samples for memory update
        if self.infinite_memory:
            tm_tau = torch.arange(1, self.nb_samples + 1).float()
            tm_l = torch.arange(self.nb_samples + 1, length + self.nb_samples + 1).float()
            tm_tau = tm_tau * self.tau / self.nb_samples # positions of old vectors
            tm_l = self.tau + (1 - self.tau) * (tm_l - self.nb_samples) / length # positions of new vectors

            positions_inf = torch.cat([tm_tau, tm_l],0).to(self.device) # positions

            if padding:
                if l % 2:
                    shift = 1 / float(length + self.nb_samples)
                    positions_pad_ = torch.linspace(-0.5 + shift, 0, 2 * (length + self.nb_samples) - 1).to(self.device)
                else:
                    shift = 1 / float(2 * length + self.nb_samples)
                    positions_pad = torch.linspace(-0.5 + shift, 1.5 - shift, 2 * (length + self.nb_samples)).to(self.device)
                positions_pad_ = torch.FloatTensor([i for i in positions_pad if i < 0]).to(self.device)
                positions_pad__ = torch.FloatTensor([i for i in positions_pad if i > 1]).to(self.device)
                positions_inf = torch.cat([positions_pad_,positions_inf,positions_pad__], dim=0)

            self.samples = None
            for t in tm_tau:
                if self.samples is None:
                    self.samples = self.psi[length][0].evaluate(t / self.tau)
                else:
                    self.samples = torch.cat([self.samples, self.psi[length][0].evaluate(t / self.tau)], dim=0)
            # compute G for the infinite case
            self.G_inf = compute_G(self.nb_samples + length, self.psi[length][0], positions_inf, padding = padding) #[L+nb_samples,N]


    def add_gaussian_basis_functions(self, psi, nb_basis, sigmas, device):
        mu, sigma = torch.meshgrid(torch.linspace(0, 1, nb_basis // len(sigmas)), torch.Tensor(sigmas))
        mu = mu.flatten().to(device)
        sigma = sigma.flatten().to(device)
        assert mu.size(0) == nb_basis
        psi.append(GaussianBasisFunctions(mu=mu, sigma=sigma))

    def score(self, query, keys):
        query = query / (self.d_head ** 0.5) # divide by sqrt(d_head) [B,h,q,d]
        keys = keys.transpose(-1, -2) #[B,h,d,N]
        scores = torch.matmul(query, keys) #[B,h,q,N] 
        return scores

    def value_function(self, x, inf = False):
        if inf:
            G = self.G_inf # [nb_sample + L, N]
        else:
            G = self.Gs[x.size(-1)] # [L, N]

        B = torch.matmul(x, G) # [B, e, N]
        B = B.permute(0,2,1) # [B, N, e]

        return B

    def update_inf(self, x):
        if self.B_past is not None:       
            xm_tau = self.B_past.transpose(-1,-2).matmul(self.samples.transpose(0,1)) # [B,e,nb_samples]
            
            x = torch.cat([xm_tau,x], dim=2) # [B, e, nb_samples + L]
            B = self.value_function(x, inf = True) # [B,N,e]
        else:
            B = self.value_function(x)
        
        self.B_past=B.detach()
        self.x_past=x
        return B

    def forward(self, k, q, new_doc = False, reg_mask = None):
        
        batch_size = k.size(1) #batch size
        qlen = q.size(0) #query length
        klen = k.size(0) #key length
        self.d_head = self.head_size #head size

        # clean memory if going through different document
        if new_doc:
            self.B_past = None 
            self.x_past = None

        k = k.permute(1,2,0) # [B,e,L]
        if self.mask:
            reg_mask = torch.sigmoid(self.mask_net(k))
            k = k * reg_mask

        # perform memory update
        if self.infinite_memory:
            B = self.update_inf(k)
        else: 
            B = self.value_function(k.view(klen,batch_size,-1)) # [B,N,e]
        
        query  = q.permute(1,0,2)
        keys   = self.proj_key(B)
        values = self.proj_value(B)

        query  = query.view(batch_size, qlen, self.n_head, self.d_head).transpose(1,2) # [B,h,q,d]
        keys   = keys.view(batch_size, self.attn_num_basis, self.n_head, self.d_head).transpose(1,2) # [B,h,N,d]
        values = values.view(batch_size, self.attn_num_basis, self.n_head, self.d_head).transpose(1,2) # [B,h,N,d]
        
        #compute scores
        scores = self.score(query, keys) #[B,h,q,N] 

        mu = torch.sigmoid(self.mu(scores)) #[B,h,q] 
        sigma_sq = self.softplus(self.sigma(scores))#[B,h,q] 
        
        mu = mu.view(-1)
        sigma_sq = torch.clamp(sigma_sq, min=1e-6).view(-1)

        if self.kl_regularizer:
            sigma_0_sq = self.sigma_0**2
            if self.mu_0 is None:
                kl_reg = 1/2 * (sigma_sq.view(batch_size, -1) / sigma_0_sq - 
                            torch.log(sigma_sq.view(batch_size, -1) / sigma_0_sq) - 1)
            else:
                kl_reg = 1/2 * (sigma_sq.view(batch_size, -1) / sigma_0_sq - 
                            torch.log(sigma_sq.view(batch_size,-1) / sigma_0_sq) - 1 +
                            (mu.view(batch_size,-1) - self.mu_0) ** 2 / sigma_0_sq )


        theta = torch.zeros(batch_size * self.n_head * qlen, 2, device = self.device)  # [B*h*q, 2]
        theta[:, 0] = mu / sigma_sq
        theta[:, 1] = -1. / (2. * sigma_sq)

        # get basis functions
        self.transform.psi = self.psi[klen]

        #compute basis functions expectation
        r = self.transform(theta) # [B*h*q,N] 

        r = r.view(batch_size, self.n_head, qlen, self.attn_num_basis).permute(0,1,3,2) # [B,h,N,q]

        values = values.transpose(-1,-2) # [B,h,d,N]
        
        context = torch.matmul(values,r) # [B,h,d,q]

        context = context.permute(3,0,1,2) # [q,B,h,d]
        context = context.contiguous().view(qlen, batch_size, self.n_head * self.d_head) # [q,B,e]

        context = self.attn_out(context) # the Long Term Memory (LTM) representation

        if self.kl_regularizer:
            return context, kl_reg
        else:
            return context
        
    @property
    def _query_dim(self):
        return self.query_layer.in_features

    def __repr__(self):
        return "ContinuousAttention" 