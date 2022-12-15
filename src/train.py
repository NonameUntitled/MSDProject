from datasets import BaseDataset, BatchProcessor
from models import BaseModel
from callbacks import BaseCallback

from utils import parse_args, create_logger, fix_random_seed, log_memory_info
from utils import DEVICE, PATH_TO_DATA, GLOBAL_TENSORBOARD_WRITER
from utils import TensorboardTimer

import os

import torch
from torch.utils.data import DataLoader

logger = create_logger(name=__name__)
SEED_VAL = 42


def train(dataloader, model, optimizer, loss_function, callback, epoch_cnt):
    step_num = 0

    logger.debug('Start training...')

    for epoch in range(epoch_cnt):
        logger.debug(f'Start epoch {epoch}')
        for step, inputs in enumerate(dataloader):
            model.train()
            optimizer.zero_grad()

            for key, values in inputs.items():
                inputs[key] = inputs[key].to(DEVICE)

            with TensorboardTimer('forward', step_num, GLOBAL_TENSORBOARD_WRITER):
                predicts = model(inputs).squeeze()
                labels = inputs['labels'].float()
                loss = loss_function(predicts, labels)

            with TensorboardTimer('backward', step_num, GLOBAL_TENSORBOARD_WRITER):
                loss.backward()

            log_memory_info(step_num, GLOBAL_TENSORBOARD_WRITER)

            # if clip_grad_threshold is not None:
            #     torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_threshold)

            with TensorboardTimer('step', step_num, GLOBAL_TENSORBOARD_WRITER):
                optimizer.step()

            callback({'predicts': predicts, 'labels': labels, 'loss': loss}, step_num)
            step_num += 1

    logger.debug('Training procedure has been finished!')


def main():
    fix_random_seed(SEED_VAL)
    config = parse_args()

    path_to_transactions = os.path.join(PATH_TO_DATA, 'all_transactions.csv')
    path_to_train_labels = os.path.join(PATH_TO_DATA, 'gender_train.csv')
    path_to_test_labels = os.path.join(PATH_TO_DATA, 'gender_test_kaggle_sample_submission.csv')

    config['dataset'].update({
        'path_to_transactions': path_to_transactions,
        'path_to_train_labels': path_to_train_labels,
        'path_to_test_labels': path_to_test_labels
    })

    dataset = BaseDataset.create_from_config(config['dataset'])

    logger.debug(
        f'Dataset meta-information: num_types={dataset.num_types}, '
        f'num_codes={dataset.num_codes}, '
        f'max_sequence_len={dataset.max_sequence_len}'
    )

    train_dataset, val_dataset, _ = dataset.datasets

    train_dataloader = DataLoader(
        train_dataset, **config['dataloader']['train'], collate_fn=BatchProcessor()
    )

    val_dataloader = DataLoader(
        val_dataset, **config['dataloader']['val'], collate_fn=BatchProcessor()
    )

    model = BaseModel.create_from_config(
        config['model'],
        num_types=dataset.num_types,
        num_codes=dataset.num_codes,
        max_sequence_len=dataset.max_sequence_len,
        length=dataset.max_sequence_len + 1,
        nb_features=config['model'].get('nb_features', 256)
    )
    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), **config['optimizer'])
    loss_function = torch.nn.BCEWithLogitsLoss()
    callback = BaseCallback.create_from_config(
        config['callback'],
        model=model,
        dataloader=train_dataloader,
        optimizer=optimizer,
        loss_function=loss_function,
    )

    # Train process
    train(
        dataloader=val_dataloader,
        model=model,
        optimizer=optimizer,
        loss_function=loss_function,
        callback=callback,
        epoch_cnt=config['epochs_num']
    )

    logger.debug('Saving model...')
    checkpoint_path = '../checkpoints/{}_final_state.pth'.format(config['experiment_name'])
    torch.save(model.state_dict(), checkpoint_path)
    logger.debug('Saved model as {}'.format(checkpoint_path))


if __name__ == '__main__':
    main()
