class Fastattention(nn.Module):
    """ 
    Wu et al., "Fastformer: Additive Attention Can Be All You Need"
    https://arxiv.org/pdf/2108.09084.pdf
    https://github.com/espnet/espnet/blob/master/espnet2/asr/layers/fastformer.py
    
    Arguments
    ---------
    enc_dim: int
        One of "mixing", "mixingv2", "sumonly"
        mixing = original Rogier with low rank
        mixingv2 = original rogier without low rank (S is repeated T times)
        sumonly = Shucong idea
    nhead : int
        Number of mixing heads.

    Example
    -------
    >>> x = torch.rand(2,4,8)
    >>> sum = SummaryMixing(8)
    >>> out = sum(x)
    >>> print(out)
    torch.Size([2, 4, 8])
    """

    def __init__(
        self,
        enc_dim,
        nhead,
        dropout=0.0
        ):
        super(Fastattention, self).__init__()
        
        assert enc_dim % nhead == 0
        self.nhead = nhead
        self.head_dim = enc_dim // nhead
        
        self.query_proj = torch.nn.Linear(enc_dim, enc_dim)
        self.query_attn = torch.nn.Linear(enc_dim, nhead)
         
        self.key_proj = torch.nn.Linear(enc_dim, enc_dim)
        self.key_attn = torch.nn.Linear(enc_dim, nhead)
        
        self.out_proj = torch.nn.Linear(enc_dim, enc_dim)
        
        self.dropout = torch.nn.Dropout(dropout)
        
        if next(self.parameters()).dtype == torch.float16:
            self.attn_fill_value = -65000
        else:
            self.attn_fill_value = -float("inf")
        
    def split_n_heads(self, x):
        """Reshape and transpose to compute scores.
        Args:
            x: (batch, time, enc_dim = n_heads * head_dim)
        Returns:
            (batch, n_heads, time, head_dim)
        """

        new_x_shape = x.shape[:-1] + (
            self.nhead,
            self.head_dim,
        )
        return x.reshape(*new_x_shape).transpose(1, 2)
    
    def linear_attn(self, scores, content, padding_mask=None):
        """
        Args:
            scores (torch.Tensor): batch, time, n_heads
            content (torch.Tensor): batch, time, enc_dim
            padding_mask (Tensor): batch, time
        """        
        scores = scores.transpose(1, 2) / self.head_dim ** 0.5 # (batch, n_heads, time)
        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask.unsqueeze(1), self.attn_fill_value)
            scores = torch.softmax(scores, dim=-1).masked_fill(padding_mask.unsqueeze(1), 0)
        else:
            scores = torch.softmax(scores, dim=-1)
            
        scores = scores.unsqueeze(2)  # (batch, n_heads, 1, time)
        
        summary = self.split_n_heads(content) # (batch, n_heads, time, head_dim)
        summary = torch.matmul(scores, summary) # (batch, n_heads, 1, head_dim)
        summary = self.dropout(summary)
        
        return summary # (batch, n_heads, 1, head_dim)
        
        
    
    def forward(self, x, padding_mask=None):
        '''
        x : torch.Tensor
            (B, T, E) where L is the target sequence length,
            B is the batch size, E is the embedding dimension.
        padding_mask : torch.Tensor, optional
            (B, T) where B is the batch size, S is the source sequence
            length. If a ByteTensor is provided, the non-zero positions will
            be ignored while the position with the zero positions will be
            unchanged. If a BoolTensor is provided, the positions with the
            value of True will be ignored while the position with the value
            of False will be unchanged.
        '''
        
        B, T, E = x.shape
        
        query = self.query_proj(x) # (batch, time, enc_dim) 
        
        query_scores = self.query_attn(query) # (batch, time, n_heads)
        query_summary = self.linear_attn(query_scores, query, padding_mask) # (batch, n_heads, 1, head_dim)
        query_summary = query_summary.squeeze(2) # (batch, n_heads, head_dim)
        
        key = self.key_proj(x)
        query_summary = query_summary.view(-1, self.head_dim * self.nhead) # (batch, enc_dim)
        query_summary = query_summary.unsqueeze(1).repeat(1, T, 1) # (batch, time, enc_dim)
        key_content = key * query_summary # (batch, time, enc_dim)
        key_scores = self.key_attn(key_content) # (batch, time, n_heads)
        key_summary = self.linear_attn(key_scores, key_content, padding_mask) # (batch, n_heads, 1, head_dim)
        
        # NOTE: value = query, due to param sharing used in the paper
        value = self.split_n_heads(query) # split into (batch, n_heads, time, head_dim) 
        value = key_summary * value # (batch, n_heads, time, head_dim) 
        value = value.transpose(1,2) # (batch, time, n_heads, head_dim) 
        value = value.reshape([B, T, E]) # (batch, time, enc_dim)

        # 10/May remove self.dropout of self.out_proj(value) + query
        value = self.out_proj(value) + query
        
        return value
