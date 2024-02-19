import torch
import torch.nn as nn 
import math

#Input embedding 
class InputEmbeddings(nn.Module):
    #d 차원 설정, vocab size 설정(얼마나 많은 단어 넣을건지)
    def __init__(self,d_model : int, vocab_size : int):
        #다른 클래스(Nn.Module)의 속성 및 메소드를 자동으로 불러와 해당 클래스에서도 사용이 가능하도록
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        #Input Embedding (단어 사이즈와 차원)
        self.embedding = nn.Embedding(vocab_size, d_model)
        
    def forward(self,x):
        #루트(d차원)을 곱해줌 (가중치 개념으로)
        return self.embedding(x) * math.sqrt(self.d_model) 
     

#Positional Encoding
class PositionalEncoding(nn.Module):
    
    #함수 리턴 값의 주석 역할(-> None)
    #해당 함수의 반환 타입의 예상 타입을 나타내기 위해 사용한다고(only for 코드 가독성)
    def __init__(self, d_model : int, seq_len: int, dropout:float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        #1. 빈 텐서 셍성 (seq_len, d_model)
        pe = torch.zeros(seq_len,d_model)
        #2. row 방향으로 (0~seq_len) 생성 (unsqueeze(dim=1))
        # 단어의 위치를 의미함
        position = torch.arange(0,seq_len, dtype = torch.float).unsqueeze(1)
        #3. col 방향으로 step=2를 활용하여 i의 2배수를 만듦 (0~2i) 
        _2i = torch.exp(torch.arange(0,d_model,2,dtype=torch.float))
        #4. cos, sine 함수 제작
        #열 기준 step 2씩 가겠다는 의미 (0::2)
        pe[:,0::2] = torch.sin(position/10000**(_2i/d_model))
        pe[:,1::2] = torch.cos(position/10000**(_2i/d_model))

        #차원 추가
        #(0번째에 => batch 차원을 나타내기 위함임)
        #기존 : seq_len, d_model => 1,seq_len,d_model
        pe = pe.unsqueeze(0) #(1,Seq_len,d_model)
        
        #https://velog.io/@nawnoes/pytorch-%EB%AA%A8%EB%8D%B8%EC%9D%98-%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0%EB%A1%9C-%EB%93%B1%EB%A1%9D%ED%95%98%EC%A7%80-%EC%95%8A%EA%B8%B0-%EC%9C%84%ED%95%9C-registerbuffer
        #버퍼는 데이터를 한 곳에서 다른 한 곳으로 전송하는 동안 일시적으로 그 데이터를 보관하는 메모리의 영역
        #모델의 학습 가능한 매개변수는 아니지만 모델과 함께 저장 및 로드되어야 하는 상태를 의미함!
        #해당 모듈을 사용하면 모델이 저장될 때나 불러올 때 해당 버퍼도 함께 저장 및 로드
        self.register_buffer('pe',pe)
    
    def forward(self,x):
        # input x + positional encoding
        # 입력 데이터의 각 위치에 해당하는 위치 임베딩을 가져오는 부분
        # 역전파 할 필요 x
        x = x + (self.pe[:,:x.shape[1],:]).requires_grad_(False)
        return self.dropout(x)
   
   
#레이어 정규화   
class LayerNormalization(nn.Module):
    
    def __init__(self, eps: float = 10**-6) -> None : 
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) #Multiplied
        self.bias = nn.Parameter(torch.zeros(1)) #Added
        
    def forward(self,x):
        mean = x.mean(dim =-1, keepdim=True)
        std = x.std(dim= -1, keepdim =True)
        return self.alpha * (x-mean) / (std+ self.eps) + self.bias
    

#FeedForward
class FeedForwardBlock(nn.Module):
    
    def __init__(self, d_model : int, d_ff : int, dropout : float) -> None:
        super().__init__()
        # feed forward upwards projection size(d_ff=2048)
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) #차원을 다시 512차원으로
        
    def forward(self,x):
        
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))      


#Multi-Head Attention
class MultiHeadAttentionBlock(nn.Module):
    
    #h : head 개수
    def __init__(self,d_model : int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        #AssertionError 실행
        #나누어 떨어지지 않으면 중지시킴
        assert d_model % h ==0, 'd_model is not divisible by h'      
        
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model) #Wq
        self.w_k = nn.Linear(d_model,d_model) #Wk
        self.w_v = nn.Linear(d_model,d_model) #Wv
        
        #concat 하는 부분에서의 wo값
        self.w_o = nn.Linear(d_model,d_model) #Wo
        self.dropout = nn.Dropout(dropout)
    
    #class 밖에서 선언된 def 함수와 같음(정적메서드)
    #굳이 인스턴스를 생성하지 않고도 호출할 수 있다.
    #ex) MultiHeadAttention.attention()
    #특정 인스턴스의 상태에 의존하지 않고 클래스 수준에서 수행되어야 할 때 유용
    @staticmethod
    def attention(query, key, value, mask, dropout : nn.Dropout):
        d_k = query.shape[-1] #(Batch, seq_len, d_model)
        
        #(Batch, h, Seq_len, d_k) -> (Batch, h, Seq_len, Seq_len)
        #쿼리와 키 간의 내적 값 구하기 -> 스케일링
        attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)
        #mask 부분 : 만약 mask가 주어졌다면,
        #0이 있는 부분을 매우 작은 값(-1e9)으로 채워 마스킹
        if mask is not None:
            attention_scores.masked_fill_(mask==0,-1e9)
        #소프트맥스 함수를 사용하여 어텐션 스코어를 확률 분포로 변환
        #각 위치에 대한 어텐션 가중치가 계산
        attention_scores = attention_scores.softmax(dim=-1) #(Batch,h,seq_len, seq_len)
        #드롭아웃이 제공되었다면 어텐션 가중치에 드롭아웃을 적용
        if dropout is not None : 
            attention_scores = dropout(attention_scores)
        #최종적으로 어텐션 가중합된 결과와 어텐션 가중치를 반환
        return (attention_scores @ value), attention_scores
    
        
    def forward(self, q, k, v, mask):
        #1. Q,K,V를  d_k, d_k, d_v 차원으로 projection
        query = self.w_q(q) #(Batch, seq_len, d_model) -> (Batch, seq_len, d_model)
        key = self.w_k(k)
        value = self.w_v(v)
        
        #Q,K,V를 head 수 만큼 분리해주기 
        #(Batch, seq_len, d_model) -> (Batch, Seq_len, h, d_k) -> (Batch, h, Seq_len, d_k)
        query = query.view(query.shape[0],query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0],key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0],value.shape[1], self.h, self.d_k).transpose(1,2)
        
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query,key,value,mask,self.dropout)
        
        #(Batch, h, Seq_len, d_k) -> (Batch, Seq_len, h,  d_k) -> (Batch,Seq_len, d_k)
        #https://ebbnflow.tistory.com/351
        #contiguous(인접한) : Tensor의 각 값들이 메모리에도 순차적으로 저장되어 있는지 여부를 의미
        x = x.transpose(1,2).contiguous().view(x.shape[0],-1, self.h*self.d_k) # -1은 나머지 차원을 자동으로 조정하라는 의미
        
        #(Batch,Seq_len, d_model) -> (Batch, seq_len, d_model)
        return self.w_o(x)
    

#ResidualConnection
class ResidualConnection(nn.Module):
    
    def __init__(self, dropout : float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()
    
    #sublayer ?    
    def forward(self,x,sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
#EncoderBlock
class EncoderBlock(nn.Module):
    
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block : FeedForwardBlock, dropout : float) -> None:
        super().__init__()
        self.self_atttention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        #ResidualConnection(dropout) for _ in range(2) : self attention 부분, Feed Forward 부분에서 두 번의 skip connection이 실행
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
        
    def forward(self,x,src_mask):
        #attention 부분 skip connection 실행
        x = self.residual_connections[0](x, lambda x: self.self_atttention_block(x,x,x,src_mask))
        #feed forward 부분 skip connection 실행
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    

#Encoder
class Encoder(nn.Module):
    
    def __init__(self, layers : nn.ModuleList) -> None : 
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
        
    def forward(self,x,mask):
        for layer in self.layers:
            x = layer(x,mask)
        return self.norm(x)
    
#DecoderBlock
class DecoderBlock(nn.Module):
    
    def __init__(self,self_attention_block : MultiHeadAttentionBlock, cross_attention_block : MultiHeadAttentionBlock, feed_forward_block : FeedForwardBlock,dropout:float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.Module([ResidualConnection(dropout) for _ in range(3)])
    
    #tgt_mask:  디코더의 현재 위치 이후의 단어들을 가려주는 마스크 
    #src_mask:  인코더 출력에서 패딩 토큰에 해당하는 위치를 0으로, 실제 단어에 해당하는 위치를 1로 채운 이진 마스크
    #단어 길이를 맞추기 위함(연산량 감소)   
    def forward(self,x,encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x,tgt_mask))
        #encoder output 값을 받는다(cross_attention)
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x,encoder_output,encoder_output,src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x

#Decoder
class Decoder(nn.Module):
    
    def __init__(self,layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm  = LayerNormalization()
        
    def forward(self,x,encoder_output,src_mask,tgt_mask):
        for layer in self.layers:
            x = layer(x,encoder_output,src_mask,tgt_mask)
        return self.norm(x)


#ProjectionLayer
class ProjectionLayer(nn.Module):
    
    def __init__(self,d_model : int, vocab_size : int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
        
    def forward(self,x):
        #(Batch, seq_len,d_model) -> (Batch,seq_len,vocab_size)
        return torch.log_softmax(self.proj(x), dim=-1)

#Transformer
class Transformer(nn.Module):
    
    def __init__(self,encoder :Encoder, decoder : Decoder, src_embed : InputEmbeddings, tgt_embed : InputEmbeddings, src_pos : PositionalEncoding, tgt_pos :PositionalEncoding, projection_layer : ProjectionLayer) -> None: 
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
        
        
    def encode(self, src,src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src,src_mask)
    
    def decode(self,encoder_output,src_mask,tgt,tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self,x):
        return self.projection_layer(x)
    
def build_transformer(src_vocab_size : int, tgt_vocab_size : int, src_seq_len : int, tgt_seq_len : int, d_model : int=512, N:int = 6, h : int = 8, dropout : float=0.1, d_ff : int=2048 ) -> Transformer:
    #Create Embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)   
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)
    
    # Create positional encoding layers
    src_pos = PositionalEncoding(d_model,src_seq_len,dropout)
    tgt_pos = PositionalEncoding(d_model,tgt_vocab_size,dropout)
    
    #Create encoder blocks
    encoder_blocks=[]
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model,h,dropout)
        feed_forward_block = FeedForwardBlock(d_model,d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
        
    #Create decoder blocks
    decoder_blocks=[]
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model,h,dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model,h,dropout)
        feed_forward_block = FeedForwardBlock(d_model,d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_self_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    #Create encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))
    
    #Create projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    #Create the transformer
    transformer = Transformer(encoder, decoder, src_embed,tgt_embed, src_pos, tgt_pos, projection_layer)
    
    #initial parameters
    for p in transformer.parameters():
        if p.dim() >1 : 
            nn.init.xavier_uniform_(p)
            
    return transformer
    