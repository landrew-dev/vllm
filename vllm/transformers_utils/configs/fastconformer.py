from transformers import PretrainedConfig

class FastConformerConfig(PretrainedConfig):
    model_type = "fastconformer"

    def __init__(self,
                 feat_dim: int = 80,
                 n_layers: int = 17,
                 d_model: int = 768,
                 n_heads: int = 8,
                 k_conv: int = 9,
                 ff_mult: int = 4,
                 attn_win_left: int = 70,
                 **kwargs):
        self.feat_dim = feat_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.k_conv = k_conv
        self.ff_mult = ff_mult
        self.attn_win_left = attn_win_left
        super().__init__(**kwargs)
