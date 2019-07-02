from fastai.torch_core import *
from fastai.data_block import *
# from fastai.callback import *

### Custom OCR transformer code ###
# Loss and Metrics
class LabelSmoothing(nn.Module):
    pass

class CER(nn.Module):
    pass

class TeacherForce(nn.Module):
    pass

def subsequent_mask(size):
    attn_shape = torch.ones((size,size), dtype=torch.int)
    return torch.tril(attn_shape).unsqueeze(0)


# ModelData
def custom_collater(samples:BatchSamples, pad_idx:int=0):
    "Function that collect samples and pads end of labels."
    data = to_data(samples)
    ims, lbls = zip(*data)
    imgs = torch.stack(list(ims))
    if len(data) is 1:
        labels = torch.zeros(1,1).long()
        return imgs, labels
    max_len = max([len(s) for s in lbls])
    labels = torch.zeros(len(data), max_len).long() + pad_idx
    for i,lbl in enumerate(lbls):
        labels[i,:len(lbl)] = torch.from_numpy(lbl)  #padding end    
    return imgs, labels

class SequenceItem(ItemBase):
    def __init__(self,data,vocab): self.data,self.vocab = data,vocab        
    def __str__(self): return self.textify(self.data)
    def __hash__(self): return hash(str(self))
    def textify(self, data): return ''.join([self.vocab[i] for i in data[:-1]])

class ArrayProcessor(PreProcessor):
    "Convert df column (string of ints) into np.array"
    def __init__(self, ds:ItemList=None): None
    def process_one(self,item): return np.array(item.split(), dtype=np.int64)
    def process(self, ds): super().process(ds)
        
class ItosProcessor(PreProcessor):
    def __init__(self, ds:ItemList=None): self.itos = ds.itos
    def process(self, ds:ItemList): ds.itos = self.itos
        
class SequenceList(ItemList):
    _processor = [ItosProcessor, ArrayProcessor]
    
    def __init__(self, items:Iterator, itos:List[str]=None, **kwargs):
        super().__init__(items, **kwargs)
        self.itos = itos
        self.copy_new += ['itos']
        self.c = len(self.items)

    def get(self, i):
        o = super().get(i)
        return SequenceItem(o, self.itos)

    def reconstruct(self,t):
        o = t.numpy()
        o = o[np.nonzero(o)]
        return SequenceItem(o, self.itos)
    
    def analyze_pred(self,pred):
        return torch.argmax(pred, dim=-1)

# Transformer Modules
class SublayerConnection(nn.Module):
    "A residual connection followed by a layer norm.  Note: (for code simplicity) norm is first."
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size, eps=1e-4)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size, eps=1e-4)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class EncoderLayer(nn.Module):
    "Encoder: self-attn and feed forward"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)

    def forward(self, x):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x))
        return self.sublayer[1](x, self.feed_forward)

class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size, eps=1e-4)
        
    def forward(self, x, src, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, src, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    "Decoder: self-attn, src-attn, and feed forward"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, src, tgt_mask=None):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, src, src))
        return self.sublayer[2](x, self.feed_forward)

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    depth = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(depth)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e4)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class SingleHeadedAttention(nn.Module):
    def __init__(self, d_model, dropout=0.2):
        super(SingleHeadedAttention, self).__init__()
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):        
        query, key, value = [l(x) for l, x in zip(self.linears, (query, key, value))]
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        return self.linears[-1](x)

class GeLU(nn.Module):
    def forward(self, x): return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.2):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_model*4)
        self.w_2 = nn.Linear(d_model*4, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GeLU() #nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=2000):
        super(PositionalEncoding, self).__init__()
        
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0.0, max_len).unsqueeze(1)
        log_increment = math.log(1e4) / d_model
        div_term = torch.exp(torch.arange(0.0, d_model, 2) * -log_increment)  
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe.unsqueeze_(0)
        self.register_buffer('pe', pe) 
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


# Custom Architecture
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def forward(self, src, tgt, tgt_mask=None):
        return self.decode(self.encode(src), tgt, tgt_mask)
    
    def encode(self, src):
        return self.encoder(src)
    
    def decode(self, src, tgt, tgt_mask=None):
        return self.decoder(self.tgt_embed(tgt), src, tgt_mask)
    
    def generate(self, outs):
        return self.generator(outs)

class ResnetBase(nn.Module):
    def __init__(self, em_sz, d_model):
        super().__init__()
        
        net = models.resnet34(True)
        modules = list(net.children())[:-3]
        self.base = nn.Sequential(*modules)
        self.linear = nn.Linear(em_sz, d_model)
        
    def forward(self, x):
        x = self.base(x).flatten(2,3).permute(0,2,1)
        x = self.linear(x) * 8
        return x

def make_full_model(vocab, d_model=512, N=4, drops=0.2):
    c = deepcopy
    attn = SingleHeadedAttention(d_model)
    ff = PositionwiseFeedForward(d_model, drops)

    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), drops), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), drops), N),
        nn.Sequential(
            Embeddings(d_model, vocab), PositionalEncoding(d_model, drops, 2000)
        ),
        nn.Linear(d_model, vocab)
    )
        
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
                    
    return model

class Img2Seq(nn.Module):
    def __init__(self, img_encoder, transformer, seq_len=500):
        super(Img2Seq, self).__init__()
        self.img_enc = img_encoder
        self.transformer = transformer
        self.seq_len = seq_len
        
    def forward(self, src, tgt=None, tgt_mask=None): 
        # inference
        if tgt is None:
            with torch.no_grad():
                feats = self.transformer.encode(self.img_enc(src))
                bs = src.size(0)
                tgt = torch.ones((bs,1), dtype=torch.long)

                res = []
                for i in progress_bar(range(self.seq_len)):
                    mask = subsequent_mask(tgt.size(-1))
                    dec_outs = self.transformer.decode(feats, tgt, mask)
                    prob = self.transformer.generate(dec_outs[:,-1])
                    res.append(prob)
                    pred = torch.argmax(prob, dim=-1, keepdim=True)
                    if (pred==0).all(): break
                    tgt = torch.cat([tgt,pred], dim=-1)
                out = torch.stack(res).transpose(1,0).contiguous()
                
        #training        
        else:
            feats = self.img_enc(src)
            dec_outs = self.transformer(feats, tgt, tgt_mask)    # ([bs, sl, d_model])
            out = self.transformer.generate(dec_outs)            # ([bs, sl, vocab])
        return out