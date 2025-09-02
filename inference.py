import torch
import tiktoken
from model.model import GPT
from utils.config import get_config
from utils.utils import text_to_ids, ids_to_text, generate

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
        decoded_text = ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()

if __name__=="__main__":
    start_context = "you should aim for you goal"
    GPT_CONFIG_124M  = get_config()
    device = GPT_CONFIG_124M['device']
    model = GPT(GPT_CONFIG_124M)
    tokenizer = tiktoken.get_encoding('gpt2')
    generate_and_print_sample(model, tokenizer, device, start_context)
 