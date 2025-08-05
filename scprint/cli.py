import torch
from jsonargparse import class_from_function
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    StochasticWeightAveraging,
)
from lightning.pytorch.cli import LightningCLI, _get_short_description
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.trainer import Trainer

from scprint.tasks import Denoiser, Embedder, GNInfer

from .trainer import TrainingMode

TASKS = [("embed", Embedder), ("gninfer", GNInfer), ("denoise", Denoiser)]


class MyCLI(LightningCLI):
    """
    MyCLI is a subclass of LightningCLI to add some missing params and create bindings between params of the model and the data.

    Used to allow calling denoise / embed / gninfer from the command line.
    Also to add more parameters and link parameters between the scdataloader and the scPRINT model.
    """

    def add_arguments_to_parser(self, parser):
        parser.add_lightning_class_args(EarlyStopping, "scprint_early_stopping")
        parser.set_defaults(
            {
                "scprint_early_stopping.monitor": "val_loss",
                # patience is the number of consecutive epochs with no improvement after which learning rate will be reduced.
                "scprint_early_stopping.patience": 5,
            }
        )
        parser.add_lightning_class_args(
            LearningRateMonitor, "scprint_learning_rate_monitor"
        )
        parser.set_defaults({"scprint_learning_rate_monitor.logging_interval": "epoch"})
        parser.add_lightning_class_args(TrainingMode, "scprint_training")
        parser.link_arguments(
            "data.gene_pos", "model.gene_pos_enc", apply_on="instantiate"
        )
        parser.link_arguments("data.genes", "model.genes", apply_on="instantiate")
        parser.link_arguments(
            "data.decoders", "model.label_decoders", apply_on="instantiate"
        )
        parser.link_arguments(
            "data.labels_hierarchy", "model.labels_hierarchy", apply_on="instantiate"
        )
        parser.link_arguments(
            "data.metacell_mode", "model.use_metacell_token", apply_on="instantiate"
        )
        parser.link_arguments("data.classes", "model.classes", apply_on="instantiate")
        parser.link_arguments(
            "data.gene_embeddings", "model.precpt_gene_emb", apply_on="parse"
        )
        parser.link_arguments(
            "data.organisms", "model.organisms", apply_on="instantiate"
        )
        parser.link_arguments(
            "data.num_datasets", "model.num_batch_labels", apply_on="instantiate"
        )
        parser.add_argument("--set_float32_matmul_precision", type=bool, default=False)
        parser.add_argument("--wandblog", type=str, default="")
        parser.add_argument("--log_freq", type=int, default=500)
        parser.add_argument("--log_graph", type=bool, default=False)
        parser.add_argument("--project", type=str)

    def _add_subcommands(self, parser, **kwargs) -> None:
        """Adds subcommands to the input parser."""
        self._subcommand_parsers = {}
        parser_subcommands = parser.add_subcommands()
        # the user might have passed a builder function
        trainer_class = (
            self.trainer_class
            if isinstance(self.trainer_class, type)
            else class_from_function(self.trainer_class)
        )
        # register all subcommands in separate subcommand parsers under the main parser
        for subcommand in self.subcommands():
            fn = getattr(trainer_class, subcommand)
            # extract the first line description in the docstring for the subcommand help message
            description = _get_short_description(fn)
            subparser_kwargs = kwargs.get(subcommand, {})
            subparser_kwargs.setdefault("description", description)
            subcommand_parser = self._prepare_subcommand_parser(
                trainer_class, subcommand, **subparser_kwargs
            )
            self._subcommand_parsers[subcommand] = subcommand_parser
            parser_subcommands.add_subcommand(
                subcommand, subcommand_parser, help=description
            )
        for subcommand in TASKS:
            fn = getattr(subcommand[1], "__init__")
            description = _get_short_description(fn)
            subparser_kwargs = {}
            subparser_kwargs.setdefault("description", description)
            parser = self.init_parser(**subparser_kwargs)
            parser.add_argument(
                "--ckpt_path",
                type=str,
                help=("Path to the checkpoint to load."),
                required=True,
            )
            parser.add_argument(
                "--output_filename",
                type=str,
                help=("Path to the output file(s)."),
                required=True,
            )
            parser.add_argument(
                "--adata", type=str, help=("Path to the anndata file."), required=True
            )
            if subcommand[0] == "gninfer":
                parser.add_argument(
                    "--cell_type",
                    type=str,
                    help=("The cell type to infer the gene regulatory network."),
                    required=True,
                )
            parser.add_argument("--seed_everything", type=int, default=42)
            parser.add_argument(
                "--species",
                type=str,
                help=(
                    "If not included in the anndata under 'organism_ontology_term_id', the species of the dataset."
                ),
                required=True,
            )
            parser.add_class_arguments(subcommand[1])
            added = parser.add_method_arguments(
                subcommand[1],
                "__call__",
                skip=set(["model", "adata", "cell_type"]),
            )
            
            self._subcommand_method_arguments[subcommand] = added
            self._subcommand_parsers[subcommand[0]] = parser
            parser_subcommands.add_subcommand(subcommand[0], parser, help=description)

    def _run_subcommand(self, subcommand: str) -> None:
        """Run the chosen subcommand."""
        before_fn = getattr(self, f"before_{subcommand}", None)
        if callable(before_fn):
            before_fn()
        if subcommand in self.subcommands():
            klass = self.trainer
            default = getattr(klass, subcommand)
            fn = getattr(self, subcommand, default)
            fn_kwargs = self._prepare_subcommand_kwargs(subcommand)
            fn(**fn_kwargs)
            after_fn = getattr(self, f"after_{subcommand}", None)
            if callable(after_fn):
                after_fn()
        else:
            import numpy as np
            import scanpy as sc
            from scdataloader import Preprocessor
            from scdataloader.utils import load_genes 
            from scprint import scPrint
            import numpy as np

            adata = sc.read_h5ad(self.config_init[subcommand]["adata"])
            adata.obs.drop(columns="is_primary_data", inplace=True, errors="ignore")
            adata.obs["organism_ontology_term_id"] = self.config_init[subcommand][
                "species"
            ]
            preprocessor = Preprocessor(
                do_postp=False,
                force_preprocess=True,
            )
            adata = preprocessor(adata)
            conf = dict(self.config_init[subcommand])
            model_checkpoint_file = self.config_init[subcommand]["ckpt_path"]
            try:
                m = torch.load(model_checkpoint_file)
            except RuntimeError: 
                m = torch.load(model_checkpoint_file, map_location=torch.device('cpu'))
            transformer = "flash" if torch.cuda.is_available() else "normal"
            if "prenorm" in m['hyper_parameters']:
                m['hyper_parameters'].pop("prenorm")
                torch.save(m, model_checkpoint_file)
            if "label_counts" in m['hyper_parameters']:
                model = scPrint.load_from_checkpoint(model_checkpoint_file, precpt_gene_emb=None, classes=m['hyper_parameters']['label_counts'], transformer=transformer)
            else:
                model = scPrint.load_from_checkpoint(model_checkpoint_file, precpt_gene_emb=None, transformer=transformer)
            missing = set(model.genes) - set(load_genes(model.organisms).index)
            if len(missing) > 0:
                print(
                    "Warning: some genes missmatch exist between model and ontology: solving...",
                )
                model._rm_genes(missing)
            if not torch.cuda.is_available():
                model = model.to(torch.float32)
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            conf["dtype"] = dtype
            
            model = model.to("cuda" if torch.cuda.is_available() else "cpu")
            for key in [
                "seed_everything",
                "config",
                "species",
                "cache",
                "ckpt_path",
                "adata",
                "output_filename",
                "cell_type",
            ]:
                conf.pop(key, None)

            if subcommand == "embed":
                emb = Embedder(**conf)
                print("embedding...")
                adata, metrics = emb(
                    model=model,
                    adata=adata,
                )
                # save the anndata
                print("metrics:")
                print(metrics)
                print()
                print(
                    "saving the file under the path: ",
                    self.config_init[subcommand]["output_filename"],
                )
                adata.write(
                    self.config_init[subcommand]["output_filename"] + "_embedded.h5ad"
                )

            if subcommand == "gninfer":
                gn = GNInfer(**conf)
                adata = gn(
                    model=model,
                    adata=adata,
                    cell_type=self.config_init[subcommand]["cell_type"],
                )
                print(
                    "saving the file under the path: ",
                    self.config_init[subcommand]["output_filename"],
                )
                adata.write(
                    self.config_init[subcommand]["output_filename"]
                    + "_"
                    + self.config_init[subcommand]["cell_type"]
                    + "_grn.h5ad"
                )

            if subcommand == "denoise":
                dn = Denoiser(**conf)
                metrics, random_indices, expr_pred = dn(
                    model=model,
                    adata=adata,
                )
                print("metrics:")
                print(metrics)
                print()
                # now what we are doing here it to complete the expression profile with the denoised values. this is not done by default for now
                print(
                    "saving the file under the path: ",
                    self.config_init[subcommand]["output_filename"],
                )
                expr_pred.write(
                    self.config_init[subcommand]["output_filename"] + "_denoised.h5ad"
                )

    def before_instantiate_classes(self):
        for k, v in self.config.items():
            if "set_float32_matmul_precision" in k:
                if v:
                    torch.set_float32_matmul_precision("medium")
        if "fit" in self.config and self.config["fit"]["trainer"]["strategy"] in [
            "ddp",
            "ddp_find_unused_parameters_true",
        ]:
            import os

            os.environ["NCCL_TIMEOUT"] = str(7000)  # 2 hours in seconds
            os.environ["TORCH_DISTRIBUTED_TIMEOUT"] = str(7000)  # 2 hours in seconds
            os.environ["PL_TRAINER_STRATEGY_TIMEOUT"] = str(7000)

            print("setting global pytorch distributed timeout to 10000s")

    def instantiate_trainer(self, **kwargs) -> Trainer:
        """Override to customize trainer instantiation"""
        # Modify strategy if it's DDP
        trainer = super().instantiate_trainer(**kwargs)
        if "fit" in self.config and self.config["fit"]["trainer"]["strategy"] in [
            "ddp",
            "ddp_find_unused_parameters_true",
        ]:
            # Create DDPStrategy with custom timeout
            from datetime import timedelta

            # Update the config
            print("updating the config")
            trainer.strategy._timeout = timedelta(seconds=7000)  # 2hours in second
            trainer.strategy.setup_distributed()
        # Call parent method to create trainer
        return trainer
