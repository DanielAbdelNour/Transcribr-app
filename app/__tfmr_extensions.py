from fastai.torch_core import *
from fastai.data_block import *
from fastai.text.data import TokenizeProcessor, Text
from fastai.text.transform import BaseTokenizer
import sentencepiece as spm

# from fastai.callback import *

### Custom OCR transformer code ###
# Loss and Metrics
class LabelSmoothing(nn.Module):
    pass

class CER(nn.Module):
    pass

class TeacherForce(nn.Module):
    pass

def rshift(tgt, bos_token=1):
    "Shift y to the right by prepending token"
    bos = torch.zeros((tgt.size(0),1)).type_as(tgt) + bos_token
    return torch.cat((bos, tgt[:,:-1]), dim=-1)

def subsequent_mask(size):
    return torch.tril(torch.ones((1,size,size)).byte())


# ModelData
#tfms = get_transforms(do_flip=False, max_zoom=1, max_rotate=2, max_warp=0.1, max_lighting=0.5)

def force_gray(image): return image.convert('L').convert('RGB')

def rm_useless_spaces(t:str) -> str:
    "Remove multiple spaces in `t`."
    return re.sub(' {2,}', ' ', t)

def add_cap_tokens(text):  # before encode
    re_caps = re.compile(r'[A-Z]+')
    return re_caps.sub(_replace_caps, text)
    
def _replace_caps(m):
    tok = '[UP]' if m.end()-m.start() > 1 else '[MAJ]'
    return tok + m.group().lower()

def remove_cap_tokens(text):  # after decode
    text = re.sub(r'\[UP\]\w+', lambda m: m.group()[4:].upper(), text)  #cap entire word
    text = re.sub(r'\[MAJ\]\w?', lambda m: m.group()[5:].upper(), text) #cap first letter
    return text

def label_collater(samples:BatchSamples, pad_idx:int=0):
    "Function that collect samples and pads ends of labels."
    data = to_data(samples)
    ims, lbls = zip(*data)
    imgs = torch.stack(list(ims))
    if len(data) is 1 and lbls[0] is 0:   #predict
        labels = torch.zeros(1,1).long()
        return imgs, labels    
    max_len = max([len(s) for s in lbls])
    labels = torch.zeros(len(data), max_len+1).long() + pad_idx  # add 1 to max_len to account for bos token
    for i,lbl in enumerate(lbls):
        labels[i,:len(lbl)] = torch.from_numpy(lbl)  #padding end    
    return imgs, labels

class SPTokenizer(BaseTokenizer):
    "Wrapper around a SentncePiece tokenizer to make it a `BaseTokenizer`."
    def __init__(self, model_prefix:str):
        self.tok = spm.SentencePieceProcessor()
        self.tok.Load(f'{model_prefix}.model')
        self.tok.SetEncodeExtraOptions("eos")

    def tokenizer(self, t:str) -> List[str]:
        return self.tok.EncodeAsIds(t)[1:]
      
class CustomTokenizer():
    def __init__(self, tok_func:Callable, model_prefix:str):
        self.tok_func, self.model_prefix = tok_func,model_prefix
        self.pre_rules = [rm_useless_spaces, add_cap_tokens]
        
    def __repr__(self) -> str:
        res = f'Tokenizer {self.tok_func.__name__} using `{self.model_prefix}` model with the following rules:\n'
        for rule in self.pre_rules: res += f' - {rule.__name__}\n'
        return res        

    def process_one(self, t:str, tok:BaseTokenizer) -> List[str]:
        "Processe one text `t` with tokenizer `tok`."
        for rule in self.pre_rules: t = rule(t)  
        toks = tok.tokenizer(t) 
        return toks 
                                                                         
    def process_all(self, texts:Collection[str]) -> List[List[str]]: 
        "Process a list of `texts`." 
        tok = self.tok_func(self.model_prefix)
        return [self.process_one(t, tok) for t in texts]
        
class SPList(ItemList):
    def __init__(self, items:Iterator, **kwargs):
        super().__init__(items, **kwargs)
        model_prefix = self.path/'spm_full_10k'
        cust_tok = CustomTokenizer(SPTokenizer, model_prefix)
        self.processor = TokenizeProcessor(tokenizer=cust_tok, include_bos=False)
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(str(model_prefix)+'.model')
        self.sp.SetDecodeExtraOptions("bos:eos")
        
        self.pad_idx = 0
        self.copy_new += ['sp']
    
    def get(self, i):
        o = self.items[i]
        return Text(o, self.textify(o))
    
    def reconstruct(self, t:Tensor):
        nonzero_idxs = (t != self.pad_idx).nonzero()
        idx_min = 0
        idx_max = nonzero_idxs.max() if len(nonzero_idxs) > 0 else 0
        return Text(t[idx_min:idx_max+1], self.textify(t[idx_min:idx_max+1]))

    def analyze_pred(self, pred:Tensor):
        return torch.argmax(pred, dim=-1)        
    
    def textify(self, ids):
        if isinstance(ids, torch.Tensor): ids = ids.tolist()
        st = self.sp.DecodeIds(ids)
        st = remove_cap_tokens(st)
        return st

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

class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model, h=8, dropout=0.2):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h        # assume d_v always equals d_k
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, mask=None):
        if mask is not None: mask = mask.unsqueeze(1)
        bs = q.size(0)
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        q, k, v = [l(x).view(bs, -1, self.h, self.d_k).transpose(1,2) for l, x in zip(self.linears, (q, k, v))]
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(q, k, v, mask=mask, dropout=self.dropout)
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous().view(bs, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.2):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_model*4)
        self.w_2 = nn.Linear(d_model*4, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GeLU() #nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.w_2(self.dropout(F.gelu(self.w_1(x))))


# Custom Architecture
class LearnedPositionalEmbeddings(nn.Module):
    def __init__(self, d_model, vocab, dropout=0.1):
        super(LearnedPositionalEmbeddings, self).__init__()
        self.nl_tok  = 4
        self.d_model = d_model

        self.embed = nn.Embedding(vocab, d_model, 0)
        self.rows = nn.Embedding(15, d_model//2, 0)
        self.w_cols = nn.Embedding(60, d_model//2, 0)
        
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        rows,cols = self.encode_spatial_positions(x)
        
        row_t = self.rows(rows)            
        col_t = self.w_cols(torch.clamp(cols, max=self.w_cols.num_embeddings-1))  # clamp to max column value
        pos_enc = torch.cat((row_t, col_t), dim=-1)
                
        x = self.embed(x)
        x = (x + pos_enc) * math.sqrt(self.d_model)
        return self.dropout(x)
    
    def encode_spatial_positions(self, x):
        rows,cols = torch.zeros_like(x),torch.zeros_like(x)
        for ii,batch in enumerate(x.unbind()):
            nls = torch.nonzero(batch==self.nl_tok).flatten()
            last = torch.nonzero(batch).flatten()[-1][None]
            splits = torch.cat([nls,last])

            p=0
            for i,n in enumerate(splits, start=1):
                rows[ii,p:n+1] = i
                cols[ii,p:n+1] = torch.arange(1,n-p+2)
                p = n+1
        return rows,cols

class ResnetBase(nn.Module):
    def __init__(self, em_sz):
        super().__init__()
        
        slices = {128: -4, 256: -3, 512: -2}
        s = slices[em_sz]

        net = models.resnet18(True)
        modules = list(net.children())[:s]
        self.base = nn.Sequential(*modules)                  #32x32 : 256
        
    def forward(self, x):
        return self.base(x)

class Adaptor(nn.Module):
    def forward(self, x):
        x = x.flatten(2,3).permute(0,2,1)
        return x.mul(8)

class WordTransformer(nn.Module):
    def __init__(self, encoder, decoder, embeddings, generator):
        super(WordTransformer, self).__init__()
        self.encoder = encoder
        self.w_decoder = decoder
        self.embed = embeddings
        self.generator = generator
            
    def forward(self, src, tgt):
        tgt = rshift(tgt, 1).long()
        mask = subsequent_mask(tgt.size(-1))
        return self.w_decoder(self.embed(tgt), self.encoder(src), mask)

    def generate(self, outs):
        return self.generator(outs)

class Img2Seq(nn.Module):
    def __init__(self, img_encoder, adaptor, transformer):
        super(Img2Seq, self).__init__()
        self.img_enc = img_encoder
        self.adaptor = adaptor
        self.transformer = transformer
        
    def forward(self, src, tgt=None, seq_len=300):
        if tgt is not None:   #train
            feats = self.adaptor(self.img_enc(src))
            outs = self.transformer(feats, tgt)
            return self.transformer.generate(outs)
        else:                 #predict
            self.eval()
            with torch.no_grad():
                feats = self.transformer.encoder(self.adaptor(self.img_enc(src)))
                tgt = torch.ones((src.size(0),1), dtype=torch.long)

                res = []
                for i in progress_bar(range(seq_len)):
                    emb = self.transformer.embed(tgt)
                    #mask = subsequent_mask(tgt.size(-1))
                    dec_outs = self.transformer.w_decoder(emb, feats)#, mask)
                    prob = self.transformer.generate(dec_outs[:,-1])
                    res.append(prob)
                    pred = torch.argmax(prob, dim=-1, keepdim=True)
                    if (pred==0).all(): break
                    tgt = torch.cat([tgt,pred], dim=-1)
                return torch.stack(res).transpose(1,0).contiguous()