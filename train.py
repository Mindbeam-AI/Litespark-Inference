
import argparse
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer
from datasets import load_dataset

from src.models.hgrn_bit.configuration_hgrn_bit import HGRNBitConfig
from src.models.hgrn_bit.modeling_hgrn_bit import HGRNBitForCausalLM

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size, args):
    setup(rank, world_size)

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create model
    if "HGRNBit" in args.model_name:
        config = HGRNBitConfig.from_pretrained(args.model_name)
        model = HGRNBitForCausalLM(config).to(rank)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name).to(rank)
    
    ddp_model = DDP(model, device_ids=[rank])

    # Load and preprocess data
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
    tokenized_datasets.set_format("torch")

    train_dataset = tokenized_datasets["train"]
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)

    # Create optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

    # Training loop
    for epoch in range(args.epochs):
        ddp_model.train()
        train_sampler.set_epoch(epoch)
        for i, batch in enumerate(train_loader):
            batch = {k: v.to(rank) for k, v in batch.items()}
            outputs = ddp_model(**batch, labels=batch["input_ids"])
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if rank == 0 and i % 10 == 0:
                print(f"Epoch: {epoch}, Step: {i}, Loss: {loss.item()}")

        scheduler.step()

        if rank == 0:
            print(f"Epoch {epoch} finished.")

    if rank == 0:
        # Save the trained model
        ddp_model.module.save_pretrained(args.save_dir)
        tokenizer.save_pretrained(args.save_dir)
        print(f"Model saved to {args.save_dir}")

    cleanup()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="ridger/MMfreeLM-370M", help="Name of the model to train.")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory where the data is stored.")
    parser.add_argument("--save_dir", type=str, default="output", help="Directory to save the trained model.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train for.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for training.")
    parser.add_argument("--world_size", type=int, default=torch.cuda.device_count(), help="Number of GPUs to use.")
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    mp.spawn(train,
             args=(args.world_size, args),
             nprocs=args.world_size,
             join=True)

if __name__ == '__main__':
    main()
