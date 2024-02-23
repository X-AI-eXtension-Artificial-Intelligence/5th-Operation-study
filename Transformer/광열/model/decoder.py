import torch
import torch.nn as nn

from model.attention import MultiHeadAttention
from model.positionwise import PositionWiseFeedForward
from model.ops import create_positional_encoding, create_position_vector, create_target_mask

class DecoderLayer(nn.Module):
    def __init__(self, params):
        super(DecoderLayer, self).__init__()
        self.layer_norm = nn.LayerNorm(params.hidden_dim, eps=1e-6)
        self.self_attention = MultiHeadAttention(params) # 함수 이동
        self.encoder_attention = MultiHeadAttention(params) # 함수 이동
        self.position_wise_ffn = PositionWiseFeedForward(params)

    def forward(self, target, encoder_output, target_mask, dec_enc_mask):
        # target          = [batch size, target length, hidden dim]
        # encoder_output  = [batch size, source length, hidden dim]
        # target_mask     = [batch size, target length, target length]
        # dec_enc_mask    = [batch size, target length, source length]

        norm_target = self.layer_norm(target) # layer norm 적용
        output = target + self.self_attention(norm_target, norm_target, norm_target, target_mask)[0] # decoder self attention

        # 디코더 스택에서 쿼리는 Encoder의 출력이고 키와 값은 인코더의 출력입니다.
        norm_output = self.layer_norm(output)
        sub_layer, attn_map = self.encoder_attention(norm_output, encoder_output, encoder_output, dec_enc_mask) # decoder encoder-decoder attention
        output = output + sub_layer

        norm_output = self.layer_norm(output)
        output = output + self.position_wise_ffn(norm_output)
        # output = [batch size, target length, hidden dim]

        return output, attn_map


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.token_embedding = nn.Embedding(params.output_dim, params.hidden_dim, padding_idx=params.pad_idx) # embedding 설정
        nn.init.normal_(self.token_embedding.weight, mean=0, std=params.hidden_dim**-0.5)
        self.embedding_scale = params.hidden_dim ** 0.5 # embedding scale
        self.pos_embedding = nn.Embedding.from_pretrained(
            create_positional_encoding(params.max_len+1, params.hidden_dim), freeze=True) # 똑같음

        self.decoder_layers = nn.ModuleList([DecoderLayer(params) for _ in range(params.n_layer)])
        self.dropout = nn.Dropout(params.dropout)
        self.layer_norm = nn.LayerNorm(params.hidden_dim, eps=1e-6)

    def forward(self, target, source, encoder_output):
        # target              = [batch size, target length]
        # source              = [batch size, source length]
        # encoder_output      = [batch size, source length, hidden dim]
        target_mask, dec_enc_mask = create_target_mask(source, target) # 함수 이동 -> 마스크 생성
        target_pos = create_position_vector(target)  # [batch size, target length] # position vector 생성

        target = self.token_embedding(target) * self.embedding_scale #
        target = self.dropout(target + self.pos_embedding(target_pos)) # target 생성
        # target = [batch size, target length, hidden dim]

        for decoder_layer in self.decoder_layers:
            target, attention_map = decoder_layer(target, encoder_output, target_mask, dec_enc_mask)
        # target = [batch size, target length, hidden dim]

        target = self.layer_norm(target)
        output = torch.matmul(target, self.token_embedding.weight.transpose(0, 1)) # weight 곱
        # output = [batch size, target length, output dim]
        return output, attention_map