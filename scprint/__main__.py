"""Entry point for scprint."""

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.cli import ArgsType, SaveConfigCallback
from lightning.pytorch.loggers import WandbLogger
from scdataloader import DataModule

from scprint import scPrint
from scprint.cli import MyCLI


class MySaveConfig(SaveConfigCallback):
    """
    MySaveConfig is a subclass of SaveConfigCallback to parametrize the wandb logger further in cli mode
    """

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if type(trainer.logger) is WandbLogger:
            if self.config.get("wandblog", "") != "":
                trainer.logger.watch(
                    pl_module,
                    log=self.config.get("wandblog", "all"),
                    log_freq=self.config.get("wandblog_freq", 1000),
                    log_graph=False,
                )
                trainer.logger.log_hyperparams({"datamodule": trainer.datamodule})
            if trainer.is_global_zero:
                print(trainer.datamodule)
                print(trainer.callbacks)
                # print(pl_module.hparams)
        return super().setup(trainer, pl_module, stage)


def main(args: ArgsType = None):
    cli = MyCLI(  # noqa: F841
        scPrint,
        DataModule,
        args=args,
        parser_kwargs={"parser_mode": "omegaconf"},
        save_config_kwargs={"overwrite": True},
        save_config_callback=MySaveConfig,
    )

if __name__ == "__main__":  # pragma: no cover
    main()
