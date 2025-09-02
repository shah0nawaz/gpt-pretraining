import torch
import tiktoken

from model.model import GPT
from utils.utils import get_config
from data.dataset import create_dataloader
from inference import generate_and_print_sample


def batch_loss(pred, target):
    pred_flat = pred.flatten(0,1)
    target_batch_flat = target.flatten(0,1)
    loss = torch.nn.functional.cross_entropy(pred_flat, target_batch_flat)
    return loss

def calc_loss_batch(model, input_batch, target_batch, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    preds = model(input_batch)

    preds_batch_flat = preds.flatten(0,1)
    target_batch_flat = target_batch.flatten(0,1)
    loss = torch.nn.functional.cross_entropy(preds_batch_flat, target_batch_flat)
    return loss

def calc_loss_loader(model, dataloader, device, num_batches=None):
    total_loss = 0.0
    if len(dataloader)==0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(dataloader)
    
    else:
        num_batches = min(num_batches, len(dataloader))
    
    for i, (input_batch, target_batch) in enumerate(dataloader):
        if i < num_batches:
            loss = calc_loss_batch(model, input_batch, target_batch, device)
            total_loss +=loss.item()
        else:
            break
    return total_loss  / num_batches

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(model,
        train_loader, device
        )
        val_loss = calc_loss_loader(model,
        val_loader, device,
        )
    model.train()
    return train_loss, val_loss

def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1
    #print(len(train_loader))
    #print(len(val_loader))
    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for i, (input_batch, target_batch) in enumerate(train_loader):
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(model, input_batch, target_batch, device)
            loss.backward()  # Calculate loss gradients
            optimizer.step()  # Update model weights using loss gradients
            tokens_seen += input_batch.numel()
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
            #print(i)
        # Print a sample text after each epoch
        #print(epoch)
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen     

with open("the-verdict.txt", "r") as f:
    raw_text = f.read()


train_ratio = 0.90
split_idx = int(train_ratio * len(raw_text))
GPT_CONFIG_124M = get_config()
settings = GPT_CONFIG_124M
train_loader = create_dataloader(
        raw_text[:split_idx],
        drop_last=True,
        shuffle=True,
        num_workers=0
    
    )
val_loader = create_dataloader(
    raw_text[split_idx:],
    drop_last=False,
    shuffle=False,
    num_workers=0
)


tokenizer = tiktoken.get_encoding("gpt2")
device = GPT_CONFIG_124M['device']
torch.manual_seed(123)
model = GPT(GPT_CONFIG_124M)
# model.to(device)
# optimizer = torch.optim.AdamW(model.parameters(),lr=0.001, weight_decay=0.1)
# num_epochs = 100
# train_losses, val_losses, track_tokens_seen = train_model_simple(
#     model, train_loader, val_loader, optimizer, device,
#     num_epochs=num_epochs, eval_freq=5, eval_iter=5,
#     start_context="Every effort moves you", tokenizer=tokenizer
#     )
