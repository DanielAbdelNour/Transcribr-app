from fastai.torch_core import *
from fastai.data_block import *
from fastai.basic_train import Learner
from fastai.text.data import TokenizeProcessor, Text
from fastai.text.transform import BaseTokenizer
import sentencepiece as spm


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

# load model
def load_data(path:PathOrStr, file:PathLikeOrBinaryStream='data.pkl'):
    state = torch.load(str(path/file), map_location='cpu')
    src = LabelLists.load_state(path, state)
    return src.databunch()

def load_graph(path:PathOrStr, file:PathLikeOrBinaryStream, data:PathLikeOrBinaryStream):
    data = load_data(path, data)
    graph = torch.jit.load(str(path/file), map_location='cpu')
    return Learner(data, graph)