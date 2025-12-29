import logging

from pytorch_lightning.callbacks import Callback

logger = logging.getLogger()

class ProgressLogger(Callback):
    def __init__(self, metric_monitor, precision = 4):
        # Metric to monitor
        self.metric_monitor = metric_monitor
        self.precision = precision

    def on_train_start(self, trainer, pl_module, **kwargs):
        logger.info("Training started")

    def on_train_end(self, trainer, pl_module, **kwargs):
        logger.info("Training ended")

    def on_validation_epoch_end(self, trainer, pl_module, **kwargs):
        if trainer.sanity_checking:
            logger.info("Sanity checking ok.")

    def on_train_epoch_end(self, trainer, pl_module, **kwargs):
        metric_format = f"{{:.{self.precision}e}}"
        line = f"Epoch:{trainer.current_epoch}"
 
        metrics_str = []

        losses_dict = trainer.callback_metrics
        for metric, dict_metric in self.metric_monitor.items():
            if dict_metric in losses_dict:
                metric_item = losses_dict[dict_metric].item()
                metric_item = metric_format.format(metric_item)
                metric_item = f"{metric}:{metric_item}"
                metrics_str.append(metric_item)

        if len(metrics_str) == 0:
            return

        line = line + ": " + "   ".join(metrics_str)
        logger.info(line)