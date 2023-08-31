import logging
import yaml
import torch
from torch import nn
from torch.utils.data import DataLoader

from data.data import load_data, collate_molgraphs
from gnn.utils import setup_logger, create_dirs, model_from_config, save_checkpoint
from gnn.test import metrics


def train_model(config):
    # save dir
    save_dir = 'runs/' + config['run_name']
    
    # create dirs
    create_dirs(save_dir)

    # dump config
    with open(f'{save_dir}/config.yaml', 'w') as f:
        yaml.dump(config, f)

    # logging
    setup_logger(save_dir,
                 log_name="train.log",
                 debug=config['debug'])
    
    # datasets
    train_data, valid_data, test_data = load_data(config)
    batch_size = config['batch_size']
    train_loader = DataLoader(dataset=train_data,
                                batch_size=batch_size,
                                collate_fn=collate_molgraphs)
    valid_loader = DataLoader(dataset=valid_data,
                                batch_size=batch_size,
                                collate_fn=collate_molgraphs)
    test_loader = DataLoader(dataset=test_data,
                                batch_size=batch_size,
                                collate_fn=collate_molgraphs)
    
    dataset_sizes = (len(train_data), len(valid_data), len(test_data))
    logging.info(f"Train, val, test sizes: {dataset_sizes}")

    # model and optimizer parameters
    device = torch.device(config['device'])
    model = model_from_config(config)
    model.to(device)
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    max_epochs = config['max_epochs']
    early_stopping_epochs = config['early_stopping_epochs']
    best_valid_error = None
    best_epoch = 0
    for epoch in range(1, max_epochs+1):
        train_loss = train_epoch(model, train_loader, loss_criterion, optimizer)
        valid_loss = eval_epoch(model, valid_loader, loss_criterion)

        if best_valid_error is None or valid_loss <= best_valid_error:
            best_valid_error = valid_loss
            best_epoch = epoch

            save_checkpoint(save_dir, model, optimizer, epoch, train_loss, valid_loss)

        logging.info(f'epoch {epoch}/{max_epochs}, train_loss: {train_loss:.2f}, valid_loss: {valid_loss:.2f}, best valid score {best_valid_error:.2f}')

        if epoch > best_epoch + early_stopping_epochs:
            break

    model.load_state_dict(torch.load(f'{save_dir}/checkpoints/best_model.pt')['model_state_dict'])
    metrics(model, test_loader)

    return

def train_epoch(model, data_loader, loss_criterion, optimizer):
    model.train()
    total_loss = 0
    n_mol = 0
    for batch_id, batch_data in enumerate(data_loader):
        graph, labels = batch_data
        labels = labels.float()
        logits = model(graph, graph.ndata['node'], graph.edata['edge'])
        loss = loss_criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        total_loss = total_loss + loss * len(labels)
        n_mol = n_mol + len(labels)
        optimizer.step()

    average_loss = total_loss / n_mol
    return average_loss

def eval_epoch(model, data_loader, loss_criterion):
    model.eval()
    total_loss = 0
    n_mol = 0
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            graph, labels = batch_data
            labels = labels.float()
            logits = model(graph, graph.ndata['node'], graph.edata['edge'])
            loss = loss_criterion(logits, labels)
            total_loss = total_loss + loss * len(labels)
            n_mol = n_mol + len(labels)

    average_loss = total_loss / n_mol
    return average_loss