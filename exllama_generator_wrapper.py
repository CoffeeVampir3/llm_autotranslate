import sys, os, random
import torch
from enum import Enum
import pygtrie
# A requirement for using exllamav2 api
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav2 import(
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer,
)

from exllamav2.generator import (
    ExLlamaV2BaseGenerator,
    ExLlamaV2Sampler
)

class StopStatus(Enum):
    CONTINUE = 1
    WAITING = 2
    STOP = 3

class StopBuffer:
    def __init__(self, stop_sequences, case_sensitive=False):
        self.trie = pygtrie.CharTrie()
        self.case_sensitive = case_sensitive
        for seq in stop_sequences:
            if not self.case_sensitive:
                seq = seq.lower()
            self.trie[seq] = True

    def append_check(self, text):
        text = text.strip()
        if not self.case_sensitive:
            text = text.lower()
        # If exact match, stop generating this is a stop condition
        if text in self.trie:
            return StopStatus.STOP
        # If partial match, continue generating but do not yield anything yet
        if self.trie.has_subtrie(text):
            return StopStatus.WAITING
        # No match, this will cause us to yield the current buffer.
        return StopStatus.CONTINUE

class ExLlamaV2StreamGenerator(ExLlamaV2BaseGenerator):
    def __init__(self, model, cache, tokenizer):
        super().__init__(model, cache, tokenizer)
    
    # Largely a copy of the original generate_simple from exllamav2, modified to support stopping sequences and streaming generation.
    def generate_step(self, prompt: str or list,
                        gen_settings: ExLlamaV2Sampler.Settings,
                        num_tokens: int,
                        token_healing = False,
                        stop_sequences = [],
                        stop_token = -1):
        
        loras = None
        if stop_token == -1: stop_token = self.tokenizer.eos_token_id
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        ids, position_offsets = self.tokenizer.encode(prompt, encode_special_tokens = False, return_offsets = True)
        if batch_size == 1: position_offsets = None
        
        stop_checker = StopBuffer(stop_sequences)
        
        overflow = ids.shape[-1] + num_tokens - self.model.config.max_seq_len
        if overflow > 0: ids = ids[:, overflow:]
        
        mask = self.tokenizer.padding_mask(ids) if batch_size > 1 else None
        
        unhealed_token = None
        if ids.shape[-1] < 2: token_healing = False
        if token_healing:
            unhealed_token = ids[:, -1:]
            ids = ids[:, :-1]
        
        self._gen_begin_base(ids, mask, loras, position_offsets = position_offsets)
        
        id_to_piece = self.tokenizer.get_id_to_piece_list()
        if unhealed_token is not None:
            unhealed_token_list = unhealed_token.flatten().tolist()
            heal = [id_to_piece[x] for x in unhealed_token_list]
        else:
            heal = None
        gen_settings.begin_filters(heal)
        
        batch_eos = [False] * batch_size

        prior_len = 0
        for i in range(num_tokens):

            logits = self.model.forward(self.sequence_ids[:, -1:], self.cache, input_mask = mask, loras = loras, position_offsets = position_offsets).float().cpu()
            token, _, _ = ExLlamaV2Sampler.sample(logits, gen_settings, self.sequence_ids, random.random(), self.tokenizer, prefix_token = unhealed_token)

            eos = False
            if stop_token is not None:
                for b in range(batch_size):
                    if token[b, 0].item() == stop_token:
                        batch_eos[b] = True
                        if all(batch_eos): eos = True
                    if batch_eos[b]:
                        token[b, 0] = self.tokenizer.pad_token_id

            self.sequence_ids = torch.cat([self.sequence_ids, token], dim = 1)
            next_step = self.tokenizer.decode(self.sequence_ids, decode_special_tokens = False)
            step_v = next_step[0]
            fragment = step_v[prior_len:len(step_v)]
            
            stop_status = stop_checker.append_check(fragment)
            if stop_status == StopStatus.CONTINUE:
                yield(fragment)
                prior_len = len(step_v)
            elif stop_status == StopStatus.STOP:
                print(f"\n: Encountered stop, fragment: {fragment}")
                return
                
            gen_settings.feed_filters(token)
            unhealed_token = None
            if eos: break


def load_model(model_directory):
    config = ExLlamaV2Config()
    config.model_dir = model_directory
    config.prepare()
    config.max_seq_len = 2048
    config.max_attention_size = 2048**2

    model = ExLlamaV2(config)

    cache = ExLlamaV2Cache(model, lazy = True)
    model.load_autosplit(cache)

    tokenizer = ExLlamaV2Tokenizer(config)

    generator = ExLlamaV2StreamGenerator(model, cache, tokenizer)
    
    return config, tokenizer, cache, generator

def generate_response(settings, max_length, stop_sequences=[]):
    print("loaded!")
    buffer = ""
    for bit in generator.generate_step("Test prompt", settings, max_length, token_healing=True, stop_sequences = stop_sequences):
        buffer += bit
    return buffer