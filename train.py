import pytorch_lightning as pl
from argparse import ArgumentParser
import yaml

from model.lightning_module import LightningModel
from data.dataset import MyDataset
from torch.utils.data import DataLoader

def main(args):
    # Load configuration
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    # Initialize the data set
    train_dataset = MyDataset(config['data']['train_data_path'])
    val_dataset = MyDataset(config['data']['val_data_path'])

    # Initialize data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['model']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['model']['batch_size'], shuffle=False)

    # Initialize model
    model = LightningModel(config)

    # Initialize PyTorch Lightning trainer
    trainer = pl.Trainer(
        max_epochs=config['trainer']['max_epochs'],
        gpus=config['trainer']['gpus']
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config.yaml file')
    args = parser.parse_args()

    main(args)
