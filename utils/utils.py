import torch

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
