import time

import torch
from torch.utils.tensorboard import SummaryWriter

LOGS_DIR = '../tensorboard_logs'
GLOBAL_TENSORBOARD_WRITER = None
BYTES_IN_MEGABYTES = 1024 ** 2
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class TensorboardWriter(SummaryWriter):

    def __init__(self, experiment_name):
        self._experiment_name = experiment_name
        super().__init__(log_dir='{}/{}'.format(LOGS_DIR, experiment_name))

    def add_scalar(self, *args, **kwargs):
        super().add_scalar(*args, **kwargs)


def log_memory_info(step_num, writer):
    writer.add_scalar(
        'memory/memory_allocated, MB',
        round(torch.cuda.memory_allocated() / BYTES_IN_MEGABYTES, 3),
        step_num
    )

    writer.add_scalar(
        'memory/memory_reserved, MB',
        round(torch.cuda.memory_reserved() / BYTES_IN_MEGABYTES, 3),
        step_num
    )


class TensorboardTimer:

    def __init__(self, scope, step_num, writer):
        self._scope = scope
        self._step_num = step_num
        self._writer = writer

    def __enter__(self):
        self.start = time.time() * 1000.0
        return self

    def __exit__(self, *args):
        self.end = time.time() * 1000.0
        interval = self.end - self.start
        self._writer.add_scalar(
            'time/{} ms'.format(self._scope),
            round(interval, 3),
            self._step_num
        )
