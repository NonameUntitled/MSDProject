import torch
import math
import torch.nn as nn

from models.base import TorchModel
from models.baseline import BaselineProjector, BaselineEncoder


class GaussianBasisFunctions(object):
    """Function phi(t) = Gaussian(t; mu, sigma_sq)."""

    def __init__(self, mu, sigma):
        self.mu = mu.unsqueeze(0)
        self.sigma = sigma.unsqueeze(0)

    def __repr__(self):
        return f"GaussianBasisFunction(mu={self.mu}, sigma={self.sigma})"

    def __len__(self):
        """Number of basis functions."""
        return self.mu.size(1)

    def _phi(self, t):
        # N(t|0,1)
        return 1. / math.sqrt(2 * math.pi) * torch.exp(-.5 * t ** 2)

    def _Phi(self, t):
        return .5 * (1 + torch.erf(t / math.sqrt(2)))

    def _integrate_product_of_gaussians(self, mu, sigma_sq):
        sigma = torch.sqrt(self.sigma ** 2 + sigma_sq)
        return self._phi((mu - self.mu) / sigma) / sigma

    def evaluate(self, t):
        # N(t|mu, sigma)
        return self._phi((t - self.mu) / self.sigma) / self.sigma

    def batch_evaluate(self, t):
        t = t.repeat(self.mu.size(0), 1) - self.mu.repeat(t.size(0), 1).transpose(1, 0)
        return self._phi(t / self.sigma) / self.sigma

    def integrate_t2_times_psi(self, a, b):
        """Compute integral int_a^b (t**2) * psi(t)."""
        return (self.mu ** 2 + self.sigma ** 2) * (
                self._Phi((b - self.mu) / self.sigma) - self._Phi((a - self.mu) / self.sigma)
        ) - (
                       self.sigma * (b + self.mu) * self._phi((b - self.mu) / self.sigma)
               ) + (
                       self.sigma * (a + self.mu) * self._phi((a - self.mu) / self.sigma)
               )

    def integrate_t_times_psi(self, a, b):
        """Compute integral int_a^b t * psi(t)."""
        return self.mu * (
                self._Phi((b - self.mu) / self.sigma) - self._Phi((a - self.mu) / self.sigma)
        ) - self.sigma * (
                       self._phi((b - self.mu) / self.sigma) - self._phi((a - self.mu) / self.sigma)
               )

    def integrate_psi(self, a, b):
        """Compute integral int_a^b psi(t)."""
        return self._Phi((b - self.mu) / self.sigma) - self._Phi((a - self.mu) / self.sigma)

    def integrate_t2_times_psi_gaussian(self, mu, sigma_sq):
        """Compute integral int N(t; mu, sigma_sq) * t**2 * psi(t)."""
        S_tilde = self._integrate_product_of_gaussians(mu, sigma_sq)
        mu_tilde = (
                           self.mu * sigma_sq + mu * self.sigma ** 2
                   ) / (
                           self.sigma ** 2 + sigma_sq
                   )
        sigma_sq_tilde = ((self.sigma ** 2) * sigma_sq) / (self.sigma ** 2 + sigma_sq)
        return S_tilde * (mu_tilde ** 2 + sigma_sq_tilde)

    def integrate_t_times_psi_gaussian(self, mu, sigma_sq):
        """Compute integral int N(t; mu, sigma_sq) * t * psi(t)."""
        S_tilde = self._integrate_product_of_gaussians(mu, sigma_sq)
        mu_tilde = (
                           self.mu * sigma_sq + mu * self.sigma ** 2
                   ) / (
                           self.sigma ** 2 + sigma_sq
                   )
        return S_tilde * mu_tilde

    def integrate_psi_gaussian(self, mu, sigma_sq):
        """Compute integral int N(t; mu, sigma_sq) * psi(t)."""
        return self._integrate_product_of_gaussians(mu, sigma_sq)


class ContinuousSoftmaxFunction(torch.autograd.Function):

    @classmethod
    def _expectation_phi_psi(cls, ctx, mu, sigma_sq):
        """Compute expectation of phi(t) * psi(t).T under N(mu, sigma_sq)."""
        num_basis = [len(basis_functions) for basis_functions in ctx.psi]
        total_basis = sum(num_basis)
        V = torch.zeros((mu.shape[0], 2, total_basis), dtype=ctx.dtype, device=ctx.device)
        offsets = torch.cumsum(torch.IntTensor(num_basis).to(ctx.device), dim=0)
        start = 0
        for j, basis_functions in enumerate(ctx.psi):
            V[:, 0, start:offsets[j]] = basis_functions.integrate_t_times_psi_gaussian(mu, sigma_sq)
            V[:, 1, start:offsets[j]] = basis_functions.integrate_t2_times_psi_gaussian(mu, sigma_sq)
            start = offsets[j]
        return V

    @classmethod
    def _expectation_psi(cls, ctx, mu, sigma_sq):
        """Compute expectation of psi under N(mu, sigma_sq)."""
        num_basis = [len(basis_functions) for basis_functions in ctx.psi]
        total_basis = sum(num_basis)
        r = torch.zeros(mu.shape[0], total_basis, dtype=ctx.dtype, device=ctx.device)
        offsets = torch.cumsum(torch.IntTensor(num_basis).to(ctx.device), dim=0)
        start = 0
        for j, basis_functions in enumerate(ctx.psi):
            r[:, start:offsets[j]] = basis_functions.integrate_psi_gaussian(mu, sigma_sq)
            start = offsets[j]
        return r

    @classmethod
    def _expectation_phi(cls, ctx, mu, sigma_sq):
        """Compute expectation of phi under N(mu, sigma_sq)."""
        v = torch.zeros(mu.shape[0], 2, dtype=ctx.dtype, device=ctx.device)
        v[:, 0] = mu.squeeze(1)
        v[:, 1] = (mu ** 2 + sigma_sq).squeeze(1)
        return v

    @classmethod
    def forward(cls, ctx, theta, psi):
        # We assume a Gaussian.
        # We have:
        # theta = [mu/sigma**2, -1/(2*sigma**2)],
        # phi(t) = [t, t**2],
        # p(t) = Gaussian(t; mu, sigma**2).
        ctx.dtype = theta.dtype
        ctx.device = theta.device
        ctx.psi = psi
        sigma_sq = (-.5 / theta[:, 1]).unsqueeze(1)
        mu = theta[:, 0].unsqueeze(1) * sigma_sq
        r = cls._expectation_psi(ctx, mu, sigma_sq)
        ctx.save_for_backward(mu, sigma_sq, r)
        return r

    @classmethod
    def backward(cls, ctx, grad_output):
        mu, sigma_sq, r = ctx.saved_tensors
        J = cls._expectation_phi_psi(ctx, mu, sigma_sq)
        e_phi = cls._expectation_phi(ctx, mu, sigma_sq)
        e_psi = cls._expectation_psi(ctx, mu, sigma_sq)
        J -= torch.bmm(e_phi.unsqueeze(2), e_psi.unsqueeze(1))
        grad_input = torch.matmul(J, grad_output.unsqueeze(2)).squeeze(2)
        return grad_input, None


class ContinuousSoftmax(nn.Module):
    def __init__(self, psi=None):
        super(ContinuousSoftmax, self).__init__()
        self.psi = psi

    def forward(self, theta):
        return ContinuousSoftmaxFunction.apply(theta, self.psi)


class LongTermAttention(nn.Module):
    # main class to compute continuous attention, with unbounded memory and sticky memories
    # like all 3rd part of article is in this class
    def __init__(self,
                 head_size: int,
                 length: int,
                 attn_num_basis: int,  # number of basis functions
                 attn_drop: float,
                 n_heads: int,
                 d_model: int,  # embeding size
                 sigma_0=1.,
                 mu_0=0.,
                 **kwargs):

        super(LongTermAttention, self).__init__()

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.length = length  # memory length
        self.head_size = head_size
        self.attn_num_basis = attn_num_basis  # N - num of basis functions
        self.n_head = n_heads  # number of heads
        self.nb_samples = 512  # number of samples (from past) used for update
        self.tau = 0.75  # compressing factor
        self.ridge_penalty = 1  # ridge penalty
        self.sigma_0 = sigma_0
        self.mu_0 = mu_0

        self.B_past = None  # previous coefficient matrix

        padding = True

        self.proj_query = nn.Linear(n_heads * head_size, n_heads * head_size, bias=False)
        self.proj_key = nn.Linear(n_heads * head_size, n_heads * head_size, bias=False)
        self.proj_value = nn.Linear(n_heads * head_size, n_heads * head_size, bias=False)

        self.attn_dropout = nn.Dropout(attn_drop)
        self.attn_out = nn.Linear(n_heads * head_size, d_model, bias=False)
        self.mu = nn.Linear(attn_num_basis, 1, bias=False)
        self.sigma = nn.Linear(attn_num_basis, 1, bias=False)
        self.softplus = torch.nn.Softplus()
        self.mask_net = torch.nn.Conv1d(n_heads * head_size, n_heads * head_size, 3, padding=1)
        self.transform = ContinuousSoftmax(psi=None)

        # get basis functions psi
        sigmas = [0.01, 0.05]  # basis function sigmas
        # maybe also need to be changed,
        # this parameters was taken from original repository
        if attn_num_basis % len(sigmas):
            attn_num_basis += (len(sigmas) - attn_num_basis % len(sigmas))

        lengths = []
        for l in range(length):
            lengths.append(l + 1)

        self.psi = [self.get_gaussian_basis_functions(attn_num_basis, sigmas, device=self.device)]

        # ========================== G ========================
        # computation of G = F.T @ (F @ F.T + ridge_penalty * I)^(-1)
        def compute_G(l, psi, positions):
            # G = F.T @ (F @ F.T + ridge_penalty * I)^(-1)
            F = torch.zeros(self.attn_num_basis, positions.size(0))  # [N,L]
            F[:, :] = psi.evaluate(positions.unsqueeze(1)).t()  # [N,L]
            I = torch.eye(self.attn_num_basis)  # [N,N]
            G = F.t().matmul((F.matmul(F.t()) + self.ridge_penalty * I).inverse())  # actual formula #[L,N]
            return G.to(self.device)

        # compute G 
        self.Gs = []
        for l in lengths:
            simple_positions = []
            for i in range(l):
                simple_positions.append((i + 1) / l)
            simple_positions = torch.tensor(simple_positions).to(self.device)
            self.Gs.append(compute_G(l, self.psi[0], simple_positions))

        # compute G for the infinite case
        self.Ginf = []
        for l in lengths:
            inf_positions = []
            tm_tau = torch.arange(1, self.nb_samples + 1).float()
            tm_l = torch.arange(self.nb_samples + 1, l + self.nb_samples + 1).float()
            tm_tau = tm_tau * self.tau / self.nb_samples  # positions of old vectors
            tm_l = self.tau + (1 - self.tau) * (tm_l - self.nb_samples) / length  # positions of new vectors
            positions_inf = torch.cat([tm_tau, tm_l], 0).to(self.device)
            self.Ginf.append(compute_G(l, self.psi[0], positions_inf))

        # ======================== end G ====================

        samples = None
        tm_tau = torch.arange(1, self.nb_samples + 1).float()
        tm_l = torch.arange(self.nb_samples + 1, l + self.nb_samples + 1).float()
        tm_tau = tm_tau * self.tau / self.nb_samples
        for t in tm_tau:
            if samples is None:
                samples = self.psi[0].evaluate(t / self.tau)
            else:
                samples = torch.cat([samples, self.psi[0].evaluate(t / self.tau)], dim=0)
        self.samples = samples

    def get_gaussian_basis_functions(self, nb_basis, sigmas, device):
        mu, sigma = torch.meshgrid(torch.linspace(0, 1, nb_basis // len(sigmas)), torch.Tensor(sigmas))
        mu = mu.flatten().to(device)
        sigma = sigma.flatten().to(device)
        assert mu.size(0) == nb_basis
        return GaussianBasisFunctions(mu=mu, sigma=sigma)

    def score(self, query, keys):
        query = query / (self.d_head ** 0.5)  # divide by sqrt(d_head) [B,h,q,d]
        keys = keys.transpose(-1, -2)  # [B,h,d,N]
        scores = torch.matmul(query, keys)  # [B,h,q,N]
        return scores

    def value_function(self, x, inf=False):
        # x : [B, e, L]
        if inf:
            G = self.Ginf[x.size(-1) - 1 - self.nb_samples]  # [nb_sample + L, N]
        else:
            G = self.Gs[x.size(-1) - 1]  # [L, N]

        B = torch.matmul(x, G)  # [B, e, N]
        B = B.permute(0, 2, 1)  # [B, N, e]
        return B

    def update_inf(self, x):
        l = x.shape[-1]
        if self.B_past is not None:
            xm_tau = self.B_past.transpose(-1, -2).matmul(self.samples.transpose(0, 1))  # [B,e,nb_samples]
            x = torch.cat([xm_tau, x], dim=2)  # [B, e, nb_samples + L]
            B = self.value_function(x, inf=True)  # [B, N, e]
        else:
            B = self.value_function(x)  # [B, N, e]
        self.B_past = B.detach()
        return B

    def forward(self, k, q, new_doc=False, reg_mask=None):
        # k, q: [L, batch size, emb]
        batch_size = k.size(1)  # batch size
        qlen = q.size(0)  # query length
        klen = k.size(0)  # key length
        self.d_head = self.head_size  # head size

        # clean memory if going through different document
        if new_doc:
            self.B_past = None

        k = k.permute(1, 2, 0)  # [B,e,L]
        reg_mask = torch.sigmoid(self.mask_net(k))
        k = k * reg_mask

        # perform memory update
        B = self.update_inf(k)

        query = q.permute(1, 0, 2)
        keys = self.proj_key(B)
        values = self.proj_value(B)

        query = query.view(batch_size, qlen, self.n_head, self.d_head).transpose(1, 2)  # [B,h,q,d]
        keys = keys.view(batch_size, self.attn_num_basis, self.n_head, self.d_head).transpose(1, 2)  # [B,h,N,d]
        values = values.view(batch_size, self.attn_num_basis, self.n_head, self.d_head).transpose(1, 2)  # [B,h,N,d]

        # compute scores
        scores = self.score(query, keys)  # [B,h,q,N]

        # computing mu and sigma
        mu = torch.sigmoid(self.mu(scores))  # [B,h,q]
        sigma_sq = self.softplus(self.sigma(scores))  # [B,h,q]
        mu = mu.view(-1)
        sigma_sq = torch.clamp(sigma_sq, min=1e-6).view(-1)

        sigma_0_sq = self.sigma_0 ** 2
        # computing kl_loss
        if self.mu_0 is None:
            kl_reg = 1 / 2 * (sigma_sq.view(batch_size, -1) / sigma_0_sq -
                              torch.log(sigma_sq.view(batch_size, -1) / sigma_0_sq) - 1)
        else:
            kl_reg = 1 / 2 * (sigma_sq.view(batch_size, -1) / sigma_0_sq -
                              torch.log(sigma_sq.view(batch_size, -1) / sigma_0_sq) - 1 +
                              (mu.view(batch_size, -1) - self.mu_0) ** 2 / sigma_0_sq)
        theta = torch.zeros(batch_size * self.n_head * qlen, 2, device=self.device)  # [B*h*q, 2]
        theta[:, 0] = mu / sigma_sq
        theta[:, 1] = -1. / (2. * sigma_sq)

        # get basis functions
        self.transform.psi = self.psi

        # compute basis functions expectation
        r = self.transform(theta)  # [B*h*q,N]
        r = r.view(batch_size, self.n_head, qlen, self.attn_num_basis).permute(0, 1, 3, 2)  # [B,h,N,q]

        values = values.transpose(-1, -2)  # [B,h,d,N]
        context = torch.matmul(values, r)  # [B,h,d,q]
        context = context.permute(3, 0, 1, 2)  # [q,B,h,d]
        context = context.contiguous().view(qlen, batch_size, self.n_head * self.d_head)  # [q,B,e]
        context = self.attn_out(context)  # the Long Term Memory (LTM) representation     # [q,B,d_model]

        return context, kl_reg


class SimpleInformerModel(nn.Module):

    def __init__(self,
                 num_types,
                 num_codes,
                 max_sequence_len,
                 num_heads,
                 embedding_dim,
                 n_heads,
                 target_len=70,
                 attn_num_basis=100,
                 attn_drop=0.,
                 sigma_0=1.,
                 mu_0=0.
                 ):
        super().__init__()
        self._projector = BaselineProjector(embedding_dim=32,
                                            num_types=num_types,
                                            num_codes=num_codes,
                                            max_sequence_len=max_sequence_len)
        assert embedding_dim % n_heads == 0
        self._encoder = LongTermAttention(head_size=int(embedding_dim / n_heads),
                                          length=max_sequence_len,
                                          target_len=target_len,
                                          attn_num_basis=attn_num_basis,
                                          attn_drop=attn_drop,
                                          n_heads=n_heads,
                                          d_model=1,
                                          sigma_0=sigma_0,
                                          mu_0=mu_0)

    def forward(self, inputs):
        embeddings, mask = self._projector(inputs)  # [bs, L, emb_dim]
        embeddings = embeddings.permute(1, 0, 2)  # [L, bs, emb_dim]
        embeddings, kl_loss = self._encoder(embeddings, embeddings)  # [L, bs, d_model], [bs, L * num_heads]
        embeddings = embeddings.permute(1, 0, 2)  # [bs, L, d_model]
        embeddings = embeddings[:, 0, :]
        kl_loss = kl_loss.mean()

        return {"predictions": embeddings, "kl_loss": kl_loss}


class InformerAttention(nn.MultiheadAttention):

    def __init__(
            self,
            embed_dim,
            num_heads,
            length,
            nb_features,
            dropout=0.,
            bias=True,
            add_bias_kv=False,
            add_zero_attn=False,
            kdim=None,
            vdim=None,
            batch_first=False,
            device=None,
            dtype=None,
            sigma_0=1.0,
            mu_0=0.0,
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            kdim=kdim,
            vdim=vdim,
            batch_first=batch_first,
            device=device,
            dtype=dtype
        )
        assert embed_dim % num_heads == 0

        self.attention = LongTermAttention(
            head_size=embed_dim // num_heads,
            length=length,
            attn_num_basis=nb_features,
            attn_drop=dropout,
            n_heads=num_heads,
            d_model=embed_dim,
            sigma_0=sigma_0,
            mu_0=mu_0
        )

    def forward(
            self,
            q, k, v,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True
    ):
        # k,q,v : [bs, L, emb]
        k = k.permute(1, 0, 2)
        q = q.permute(1, 0, 2)  # k,q : [L, bs, emb]

        v = self.attention(k, q)[0].permute(1, 0, 2)

        return v


class InformerAttentionLayer(nn.TransformerEncoderLayer):

    def __init__(
            self,
            d_model,
            nhead,
            length,
            nb_features,
            dim_feedforward=None,
            dropout=0.0,
            activation=nn.ReLU,
            layer_norm_eps=1e-5,
            batch_first=False,
            norm_first=False,
            device=None,
            dtype=None,
            **kwargs
    ):
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward or 4 * d_model,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first,
            device=device,
            dtype=dtype
        )
        assert d_model % nhead == 0

        self.self_attn = InformerAttention(
            embed_dim=d_model,
            num_heads=nhead,
            length=length,
            nb_features=nb_features
        )


class Informer(TorchModel, config_name='informer'):

    def __init__(self, projector, encoder):
        super().__init__()
        self._projector = projector
        self._encoder = encoder

    @classmethod
    def create_from_config(cls, config, **kwargs):
        projector = BaselineProjector.create_from_config(config['projector'], **kwargs)
        encoder = BaselineEncoder.create_from_config(
            config['encoder'],
            input_dim=projector.output_dim,
            attention_layer_cls=InformerAttentionLayer,
            **kwargs
        )
        return cls(projector, encoder)

    def encoder_only(self, embeddings):
        mask = embeddings.new_ones(embeddings.shape[:-1]).bool()
        embeddings, mask = self._encoder(embeddings, mask)
        return embeddings

    def forward(self, inputs):
        embeddings, mask = self._projector(inputs)
        embeddings, mask = self._encoder(embeddings, mask)
        return embeddings
