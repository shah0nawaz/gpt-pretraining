import torch
import numpy as np

def text_to_ids(text, tokenizer):
    ids = torch.tensor(tokenizer.encode(text))
    ids = ids.unsqueeze(0)
    return ids
    
def ids_to_text(token_ids, tokenizer):
    return tokenizer.decode(token_ids.squeeze(0).tolist())


def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):

        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Get the idx of the vocab entry with the highest logits value
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx

def generate(model, idx, max_new_tokens, 
            context_size, temperature=0.0,
            top_k=None, eos_id=None):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):

        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Get the idx of the vocab entry with the highest logits value
        # idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)
        if top_k is not None:
            top_logits, top_pos = torch.topk(logits, top_k)
            new_logits = torch.where(
                logits < top_logits[0][-1],
                torch.tensor(float('-inf')),
                other=logits
            )
        if temperature > 0.0:
            logits = logits / temperature
            probas = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probas, num_samples=1)
        
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)
        if idx_next == eos_id:
            break
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))

def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])
    for b in range(len(params["blocks"])):
        q_w, k_w, v_w  = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
    
        gpt.trf_blocks[b].mha.W_Q.weight  = assign(
            gpt.trf_blocks[b].mha.W_Q.weight, q_w.T
        )
        gpt.trf_blocks[b].mha.W_K.weight  = assign(
            gpt.trf_blocks[b].mha.W_K.weight, k_w.T
        )
        gpt.trf_blocks[b].mha.W_V.weight  = assign(
            gpt.trf_blocks[b].mha.W_V.weight, v_w.T
        )
    
        q_b, k_b, v_b = np.split(
        (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].mha.W_Q.bias = assign(
            gpt.trf_blocks[b].mha.W_Q.bias, q_b)
        gpt.trf_blocks[b].mha.W_K.bias = assign(
            gpt.trf_blocks[b].mha.W_K.bias, k_b)
        gpt.trf_blocks[b].mha.W_V.bias = assign(
            gpt.trf_blocks[b].mha.W_V.bias, v_b)
    
    
        gpt.trf_blocks[b].mha.proj.weight = assign(
            gpt.trf_blocks[b].mha.proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        
        gpt.trf_blocks[b].mha.proj.bias = assign(
            gpt.trf_blocks[b].mha.proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])
        
        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])
        
        gpt.trf_blocks[b].layernorm1.scale = assign(
            gpt.trf_blocks[b].layernorm1.scale,
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].layernorm1.shift = assign(
            gpt.trf_blocks[b].layernorm1.shift,
             params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].layernorm2.scale = assign(
        gpt.trf_blocks[b].layernorm2.scale,
        params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].layernorm2.shift = assign(
            gpt.trf_blocks[b].layernorm2.shift,
            params["blocks"][b]["ln_2"]["b"])
    
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])
