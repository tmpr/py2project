import torch
import json
from tqdm import trange, tqdm
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scoring import main as score

from datasets import DisectedSet
from architectures import CNN
from functional import crop_mse

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(32)


def main(json_path: Path):
    mse = torch.nn.MSELoss()
    model, optimizer, loaders, scheduler, epochs = _setup(json_path)

    curr_best = None
    for epoch in trange(1, epochs + 1):

        _ = train_network(model, optimizer, loaders['train'], mse)
        eval_loss = eval_network(model, loaders['eval'], mse)

        scheduler.step(eval_loss)

        print(f"Epoch {epoch}, Evaluation Loss: {eval_loss}")
        if not curr_best or eval_loss < curr_best:
            print('Improvenment: ', eval_loss - (curr_best or 0))
            torch.save(model, 'models/model.pt')
            curr_best = eval_loss

        print("Would score on the testset:")
        score()


def train_network(model, optimizer, train_loader, lss_fc) -> None:
    """Train Network for one Epoch."""
    train_losses = []
    for batch in tqdm(train_loader, total=len(train_loader)):
        optimizer.zero_grad()

        input_tensor, original = batch
        input_tensor = input_tensor.to('cuda')

        out = model(input_tensor)
        loss = crop_mse(original=original,
                        out=out,
                        mask=input_tensor[:, 1],
                        mse=lss_fc)

        loss.backward()
        optimizer.step()
        train_losses.append(loss.detach())

    return sum(train_losses)/len(train_losses)


def eval_network(model, eval_loader, mse) -> torch.tensor:
    """Return average loss over whole evaluation set."""
    eval_losses = []
    for batch in tqdm(eval_loader, total=len(eval_loader)):
        input_tensor, original = batch
        input_tensor = input_tensor.to('cuda')

        out = model(input_tensor)
        loss = crop_mse(original=original,
                        out=out,
                        mask=input_tensor[:, 1],
                        mse=mse)

        eval_losses.append(loss.detach())

    return sum(eval_losses)/len(eval_losses)


def _setup(config_path: Path):
    config = json.loads(config_path.read_text())
    batch_size = config['batch_size']
    
    loaders = {
        'train': DataLoader(DisectedSet('data/processed/train'),
                            batch_size=batch_size, shuffle=True,
                            num_workers=14),
        'eval':  DataLoader(DisectedSet('data/processed/eval'),
                            batch_size=4, shuffle=True,
                            num_workers=14)
    }

    model = _load_model(config['network_config'])
    print("Training the following architecture:")
    print(model)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config['learningrate'],
                                 weight_decay=config['weight_decay'])

    scheduler = ReduceLROnPlateau(optimizer, 
                                  patience=config['scheduler']['patience'], 
                                  threshold=config['scheduler']['threshold'], 
                                  verbose=True)

    epochs = config['epochs']

    return model, optimizer, loaders, scheduler, epochs


def _load_model(network_config: dict) -> nn.Module:
    if input("Use preloaded model? (Only 'no' counts as 'no'.)") != 'no':
        model = torch.load(f'models/model.pt')
    else:
        model = CNN(
            n_hidden=network_config['n_hidden'],
            kernel_size=network_config['kernel_size'],
            n_kernels=network_config['n_kernels'],
            n_in_channels=network_config['n_in_channels']
        ).to(DEVICE)
    
    return model


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main(Path('config.json'))
