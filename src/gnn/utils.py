import logging
import sys
from pathlib import Path
import torch

from gnn.models import RGCN, WLNClassifier


def setup_logger(save_dir, log_name="output.log", debug=False):
    """setup_logger.

    Args:
        save_dir:
        log_name:
        debug:
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    log_file = save_dir / log_name

    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)

    file_handler = logging.FileHandler(log_file)

    file_handler.setLevel(level)

    # Define basic logger
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            stream_handler,
            file_handler,
        ],
    )

    # configure logging at the root level of lightning
    # logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

    # configure logging on module level, redirect to file
    logger = logging.getLogger("pytorch_lightning.core")
    logger.addHandler(logging.FileHandler(log_file))

    return

def create_dirs(save_dir):
    Path(f'{save_dir}/').mkdir(exist_ok=True)
    Path(f'{save_dir}/checkpoints/').mkdir(exist_ok=True)

    return

def save_checkpoint(save_dir, model, optimizer, epoch, train_loss, valid_loss):
    torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss,
            }, 
        f'{save_dir}/checkpoints/model-{epoch}.pt'
    )
    torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss,
            }, 
        f'{save_dir}/checkpoints/best_model.pt'
    )

def model_from_config(config):
    if config['model'] == 'RGCN':
        model = RGCN(config['node_feat_size'], config['hidden_size'], config['gnn_layers'], config['fc_layers'], config['rgcn_dropout'], config['fc_dropout'])
    elif config['model'] == 'WLNClassifier':
        model = WLNClassifier(config['node_feat_size'], config['edge_feat_size'], config['num_classes'], config['hidden_size'], config['gnn_layers'])
    else:
        raise NotImplementedError



    return model