from lightning.pytorch.loggers import TensorBoardLogger


class FedstellarLogger(TensorBoardLogger):

    def __init__(self, *args, **kwargs):
        self.local_step = 0
        self.global_step = 0
        super().__init__(*args, **kwargs)

    def log_metrics(self, metrics, step=None):
        # Any custom code to log metrics
        # FL round information
        self.local_step = step
        step = self.global_step + self.local_step
        # logging.info(f'(statisticslogger.py) log_metrics: step={step}, metrics={metrics}')
        if "epoch" in metrics:
            metrics.pop("epoch")
        super().log_metrics(metrics, step)  # Call the original log_metrics
