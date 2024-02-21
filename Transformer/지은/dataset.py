import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):

    #클래스를 생성할 때 실행되는 생성자
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None:
        super().__init__()

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.Tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64)
        self.eos_token = torch.Tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64)
        self.pad_token = torch.Tensor([tokenizer_src.token_to_id('[PAD]')], dtype=torch.int64)

    #원소의 개수를 셀 때 접근되는 메서드
    def __len__(self):
        return len(self.ds)
    
    #인덱스에 접근할 때 호출되는 메서드
    def __getitem__(self, index : Any) -> Any:
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang] 

        # 토크나이저를 사용하여 텍스트를 토큰화하고, 그 결과에서 토큰 IDs를 추출
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids        

        #seq_len(시퀀스 길이)에 맞추기 위해 패딩을 추가해줌
        #원본 언어의 입력 시퀀스에 추가해야 하는 패딩 토큰의 수
        # SOS, EOS token (-2)
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        #대상 언어의 출력 시퀀스에 추가해야 하는 패딩 토큰의 수
        # EOS token (-1)
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1
        
        #self.seq_len : 10
        #enc_input_tokens : ["안녕", "하세요", "반갑습니다"]
        #dec_input_tokens : ["Hello", "World"]
        #enc_num_padding_tokens: 10 - len(["안녕", "하세요", "반갑습니다"]) - 2 = 10 - 3 - 2 = 5
        #dec_num_padding_tokens: 10 - len(["Hello", "World"]) - 1 = 10 - 2 - 1 = 7
        #['sos',"안녕", "하세요", "반갑습니다", 'eos',패딩, 패딩, 패딩, 패딩, 패딩]
        #['Hello', 'World', 'EOS', 패딩, 패딩, 패딩, 패딩, 패딩, 패딩, 패딩]
        
        #0보다 작을 경우, 에러 발생 ex. 10 - 9 - 2 = -1 
        #전체 문장 길이보다 input 문장 길이가 더 긴 경우,
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")
        

        # 토큰 Concat

        # Add <s> and </s> token
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )        
        

        # Add only <sos> token
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Add only <eos> token
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Double check the size of the tensors to make sure they are all seq_len long
        # Asser  : if 개념
        # 최대 시퀀스와 input+패딩 한 결과가 동일한 길이인지를 물어봄
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)

            #차원을 맞춰주기 위해 실행
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            # 현재 위치 이후의 토큰 마스크 처리 (패딩 토큰이 아닌 위치에 대해 True, 패딩 토큰인 위치에 대해 False 반환) 
            # -> 특정 위치에 토큰에 대한 attention을 수행할지 말지를 결정
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len),
            "label": label,  # (seq_len) 디코더의 실제 출력에 해당하는 값
            "src_text": src_text, #input text
            "tgt_text": tgt_text, #target text
        }
    
def causal_mask(size):
    #https://incredible.ai/nlp/2020/02/29/Transformer/#241-padding-mask-%ED%95%B5%EC%8B%AC-%EB%82%B4%EC%9A%A9
    #triu : 정사각형의 n x n 이 있을 때, 아랫부분은 0으로, 위쪽 삼각부분만 1로 리턴
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0