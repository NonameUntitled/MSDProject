from datasets import BaseDataset, BatchProcessor
from models import BaseModel
from metric import BaseMetric

from utils import parse_args, create_logger, fix_random_seed
from utils import DEVICE, PATH_TO_DATA

import os
import time
import numpy as np
from collections import Counter

import torch
from torch.utils.data import DataLoader

logger = create_logger(name=__name__)
SEED_VAL = 42


def benchmark(model, input_shape, nwarmup=50, nruns=1000, period=100):
    input_data = torch.randn(tuple(input_shape))
    input_data = input_data.to(DEVICE)

    logger.debug("Warm up ...")
    with torch.no_grad():
        for _ in range(nwarmup):
            features = model.encoder_only(input_data)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    logger.debug("Start timing ...")
    timings = []
    with torch.no_grad():
        for i in range(1, nruns + 1):
            start_time = time.time()

            features = model.encoder_only(input_data)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            end_time = time.time()

            timings.append(end_time - start_time)
            if i % period == 0:
                logger.debug('Iteration %d/%d, ave batch time %.2f ms' % (i, nruns, np.mean(timings) * 1000))

    logger.info('Input shape: {}'.format(input_data.size()))
    logger.info('Output features size: {}'.format(features.size()))
    logger.info('Average batch time: {} ms '.format(round(np.mean(timings) * 1000, 2)))


def inference(dataloader, model, metrics):
    logger.debug('Model inference...')
    running_params = Counter()

    model.eval()
    with torch.no_grad():
        for batch in dataloader:

            for key, value in batch.items():
                batch[key] = value.to(DEVICE)

            predicts = model(batch).squeeze()
            labels = batch['labels'].float()

            for metric_name, metric_function in metrics.items():
                running_params[metric_name] += metric_function(
                    inputs={'predicts': predicts.cpu(), 'labels': labels.cpu()}
                )

    for label, value in running_params.items():
        logger.info('Inference {}: {}'.format(label, value / len(dataloader)))

    logger.debug('Model inference done!')


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

    _, val_dataset, test_dataset = dataset.datasets

    val_dataset = DataLoader(
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

    checkpoint_path = '../checkpoints/{}_final_state.pth'.format(config['experiment_name'])
    logger.debug('Loading model from {}'.format(checkpoint_path))
    model.load_state_dict(torch.load(checkpoint_path))
    logger.debug('Loading model done!')

    metrics = {
        metric_name: BaseMetric.create_from_config(metric_cfg)
        for metric_name, metric_cfg in config['metrics'].items()
    }

    # Benchmark process
    benchmark(
        model=model,
        **config.get('benchmark_params', {})
    )

    # Inference process
    inference(
        dataloader=val_dataset,
        model=model,
        metrics=metrics
    )


if __name__ == '__main__':
    main()
