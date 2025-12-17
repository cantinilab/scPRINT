from typing import List, Optional

from lightning.pytorch.callbacks import Callback


class TrainingMode(Callback):
    def __init__(
        self,
        noise: List[float] = [0.6],
        cce_temp: float = 0.3,  # .6
        cce_scale: float = 0.2,  # .01
        ecs_threshold: float = 0.4,
        class_embd_diss_scale: float = 0.3,
        ecs_scale: float = 0.2,  # .1
        mvc_scale: float = 0.0,
        do_next_tp: bool = False,
        do_generate: bool = True,
        class_scale: float = 1,
        mask_ratio: List[float | str] = [],  # 0.3
        test_every: int = 5,
        randsamp: bool = True,
        warmup_duration: int = 500,
        fused_adam: bool = False,
        adv_class_scale: float = 1.0,
        lr_reduce_patience: int = 2,
        lr_reduce_factor: float = 0.6,
        lr_reduce_monitor: str = "val_loss",
        run_full_forward: bool = False,
        lr: float = 0.0001,
        dropout: float = 0.1,
        optim: str = "adamW",
        weight_decay: float = 0.01,
        zinb_and_mse: bool = False,
        var_context_length: bool = False,
        vae_kl_warmup_steps: int = 80_000,
        vae_kl_scale: float = 0.001,
        name: str = "",
        set_step: Optional[int] = None,
        mask_zeros: bool = False,
    ):
        """
        TrainingMode a callback to set the training specific info to the model.

        This is because lightning is unfortunately setup this way. the model should be separated from training
        but at the same time it has training specific methods... so we have to do this.

        Args:
            noise (List[float]): List of noise levels to apply if denoising is enabled. Defaults to [0.6], meaning only one forward path with 60% of the counts being dropped will happen.
            cce_temp (float): Similarity threshold for CCE. Defaults to 0.5.
            cce_scale (float): Scaling factor for CCE loss. Defaults to 0.002.
            ecs_threshold (float): Threshold for ECS. Defaults to 0.3.
            ecs_scale (float): Scaling factor for ECS loss. Defaults to 0.05.
            mvc_scale (float): Scaling factor for MVC loss. Defaults to 1.0.
            do_generate (bool): Whether to do the bottleneck learning task. Defaults to True.
            class_scale (float): Scaling factor for classification loss. Defaults to 1.5.
            mask_ratio (List[float]): List of mask ratios to apply during training. Defaults to [], meaning no masking is applied during pretraining.
            warmup_duration (int): Number of warmup steps for learning rate scheduling. Defaults to 500.
            fused_adam (bool): Whether to use fused Adam optimizer. Defaults to True.
            adv_class_scale (float): Scaling factor for adversarial classification loss. Defaults to 0.1.
            lr_reduce_patience (int): Number of epochs with no improvement after which learning rate will be reduced. Defaults to 1.
            lr_reduce_factor (float): Factor by which the learning rate will be reduced. Defaults to 0.6.
            lr_reduce_monitor (str): Quantity to be monitored for learning rate reduction. Defaults to "val_loss".
            run_full_forward (bool): Whether to run a second forward pass without masking or denoising for the bottleneck learning / MVC case. Defaults to False.
            lr (float): Initial learning rate. Defaults to 0.001.
            optim (str): Optimizer to use during training. Defaults to "adamW".
            weight_decay (float): Weight decay to apply during optimization. Defaults to 0.01.
            name (str): Name of the training mode. Defaults to an empty string. should be an ID for the model
            test_every (int): Number of epochs between testing. Defaults to 1.
            class_embd_diss_scale (float): Scaling factor for the class embedding dissimilarity loss. Defaults to 0.1.
            zinb_and_mse (bool): Whether to use ZINB and MSE loss. Defaults to False.
            var_context_length (bool): Whether to use variable context length. Defaults to False.
            dropout (float): Dropout rate for the model. Defaults to 0.1.
            set_step (int, optional): Set the global step for the model. Defaults to None.
            vae_kl_scale (float): Scaling factor for the VAE KL loss. Defaults to 0.3.
            randsamp (bool): Whether to use random sampling for the noise amount at each training step. Defaults to True.
            vae_kl_warmup_steps (int): Number of warmup steps for the VAE KL loss. Defaults to 20_000.
            mask_zeros (bool): Whether to mask zeros in the expression matrix. Defaults to False.
        """
        super().__init__()
        self.noise = noise
        self.cce_temp = cce_temp
        self.cce_scale = cce_scale
        self.ecs_threshold = ecs_threshold
        self.ecs_scale = ecs_scale
        self.vae_kl_scale = vae_kl_scale
        self.do_next_tp = do_next_tp
        self.do_generate = do_generate
        self.class_scale = class_scale
        self.mask_ratio = mask_ratio
        self.warmup_duration = warmup_duration
        self.fused_adam = fused_adam
        self.mvc_scale = mvc_scale
        self.adv_class_scale = adv_class_scale
        self.lr_reduce_patience = lr_reduce_patience
        self.lr_reduce_factor = lr_reduce_factor
        self.lr_reduce_monitor = lr_reduce_monitor
        self.lr = lr
        self.optim = optim
        self.weight_decay = weight_decay
        self.run_full_forward = run_full_forward
        self.name = name
        self.test_every = test_every
        self.class_embd_diss_scale = class_embd_diss_scale
        self.zinb_and_mse = zinb_and_mse
        self.var_context_length = var_context_length
        self.dropout = dropout
        self.set_step = set_step
        self.randsamp = randsamp
        self.mask_zeros = mask_zeros
        self.vae_kl_warmup_steps = vae_kl_warmup_steps

    def __repr__(self):
        return (
            f"TrainingMode("
            f"noise={self.noise}, "
            f"cce_temp={self.cce_temp}, "
            f"cce_scale={self.cce_scale}, "
            f"ecs_threshold={self.ecs_threshold}, "
            f"ecs_scale={self.ecs_scale}, "
            f"lr={self.lr},"
            f"optim={self.optim},"
            f"weight_decay={self.weight_decay},"
            f"vae_kl_scale={self.vae_kl_scale},"
            f"adv_class_scale={self.adv_class_scale}, "
            f"do_next_tp={self.do_next_tp}, "
            f"class_scale={self.class_scale}, "
            f"mask_ratio={self.mask_ratio}, "
            f"warmup_duration={self.warmup_duration}, "
            f"fused_adam={self.fused_adam}, "
            f"lr_reduce_patience={self.lr_reduce_patience}, "
            f"lr_reduce_factor={self.lr_reduce_factor}, "
            f"lr_reduce_monitor={self.lr_reduce_monitor}, "
            f"mvc_scale={self.mvc_scale}, "
            f"run_full_forward={self.run_full_forward}), "
            f"name={self.name}, "
            f"test_every={self.test_every}, "
            f"class_embd_diss_scale={self.class_embd_diss_scale}, "
            f"zinb_and_mse={self.zinb_and_mse}, "
            f"var_context_length={self.var_context_length}, "
            f"dropout={self.dropout}, "
            f"set_step={self.set_step}, "
            f"randsamp={self.randsamp}, "
            f"mask_zeros={self.mask_zeros}, "
            f"vae_kl_warmup_steps={self.vae_kl_warmup_steps})"
        )

    def setup(self, trainer, model, stage=None):
        # do something with all training_step outputs, for example:
        model.noise = self.noise
        model.cce_temp = self.cce_temp
        model.cce_scale = self.cce_scale
        model.ecs_threshold = self.ecs_threshold
        model.ecs_scale = self.ecs_scale
        model.class_embd_diss_scale = self.class_embd_diss_scale
        model.mvc_scale = self.mvc_scale
        model.do_generate = self.do_generate
        model.adv_class_scale = self.adv_class_scale
        model.do_next_tp = self.do_next_tp
        model.class_scale = self.class_scale
        model.vae_kl_scale = self.vae_kl_scale
        model.mask_ratio = self.mask_ratio
        model.warmup_duration = self.warmup_duration
        model.fused_adam = self.fused_adam
        model.lr_reduce_patience = self.lr_reduce_patience
        model.lr_reduce_factor = self.lr_reduce_factor
        model.lr_reduce_monitor = self.lr_reduce_monitor
        model.run_full_forward = self.run_full_forward
        model.lr = self.lr
        model.optim = self.optim
        model.weight_decay = self.weight_decay
        model.name = self.name
        model.test_every = self.test_every
        model.zinb_and_mse = self.zinb_and_mse
        model.var_context_length = self.var_context_length
        model.dropout = self.dropout
        model.set_step = self.set_step
        model.randsamp = self.randsamp
        model.mask_zeros = self.mask_zeros
        model.vae_kl_warmup_steps = self.vae_kl_warmup_steps
        # model.configure_optimizers()
