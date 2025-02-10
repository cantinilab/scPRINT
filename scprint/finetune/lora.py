import torch
from .utils import mark_only_lora_as_trainable
from torch.optim import AdamW
from torch.optim.lr_scheduler import get_lr_scheduler


def lora(model, datamodule, r, lora_alpha, lora_dropout, lr, weight_decay, precision):
    mark_only_lora_as_trainable(model)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = get_lr_scheduler(
        optimizer, warmup_steps=train.lr_warmup_steps, max_steps=lr_max_steps
    )

    # strict=False because missing keys due to LoRA weights not contained in state dict
    load_checkpoint(model, checkpoint_path, strict=False)

    train_time = time.perf_counter()
    token_counts = fit(
        model,
        optimizer,
        scheduler,
        datamodule,
        devices,
        out_dir,
    )

    training_time = time.perf_counter() - train_time
    output = create_finetuning_performance_report(
        training_time, token_counts, model.device.type
    )
    print(output)

    # Final evaluation
    if eval.final_validation:
        val_loss = validate(
            model,
            val_dataloader,
            dataclasses.replace(eval, max_iters=len(val_dataloader)),
        )
        metrics = {"val_loss": val_loss, "val_ppl": math.exp(val_loss)}
        trainer.log_dict(metrics)
        print(
            f"Final evaluation | val loss: {val_loss.item():.3f} | val ppl: {math.exp(val_loss):.3f}"
        )

    # Save the final LoRA checkpoint at the end of training
    save_path = out_dir / "final" / "lit_model.pth.lora"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_lora_checkpoint(model, save_path)
    if trainer.global_rank == 0:
        # Copy checkpoint files from original checkpoint dir
        copy_config_files(checkpoint_dir, save_path.parent)
        save_hyperparameters(setup, save_path.parent)
        save_prompt_style(data.prompt_style, save_path.parent)
        merge_lora(checkpoint_dir=save_path.parent)


def fit(
    model,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    datamodule,
    devices: int,
    checkpoint_dir: Path,
    out_dir: Path,
) -> dict:
    if eval.initial_validation:
        val_loss = validate(
            model,
            val_dataloader,
            dataclasses.replace(eval, max_iters=len(val_dataloader)),
        )
        val_loss = f"{val_loss:.3f}"
    else:
        print("Verifying settings ...")
        validate(
            model,
            val_dataloader,
            dataclasses.replace(eval, max_iters=2),
        )  # sanity check
        val_loss = "n/a"

    train_iterator = CycleIterator(train_dataloader)
    throughput = ThroughputMonitor(fabric, window_size=50)
    running_loss = RunningMean(
        window=train.gradient_accumulation_iters(devices), sync_on_compute=False
    ).to(model.device)
    max_steps = train.max_steps or float("inf")
    step_count = 0
    iter_num = 0
    total_lengths = 0
    total_t0 = time.perf_counter()

    token_counts = {
        "raw_tokens": torch.tensor(0, device=model.device, dtype=torch.long),
        "raw_tokens_plus_prompt_template": torch.tensor(
            0, device=model.device, dtype=torch.long
        ),
        "raw_tokens_plus_prompt_template_and_padding": torch.tensor(
            0, device=model.device, dtype=torch.long
        ),
    }

    while step_count < max_steps:
        iter_num += 1
        iter_t0 = time.perf_counter()
        batch = next(train_iterator)
        if train_iterator.epoch >= train.epochs:
            break
        input_ids, targets = batch["input_ids"], batch["labels"]

        is_accumulating = iter_num % train.gradient_accumulation_iters(devices) != 0
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(input_ids, lm_head_chunk_size=128)
            # shift the targets such that output n predicts token n+1
            logits[-1] = logits[-1][..., :-1, :]
            loss = chunked_cross_entropy(logits, targets[..., 1:])
            fabric.backward(loss / train.gradient_accumulation_iters(devices))

        running_loss.update(loss.detach())

        if not is_accumulating:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            step_count += 1

        if iter_num % train.log_interval == 0:
            loss = (
                running_loss.compute().item()
            )  # expensive device-to-host synchronization
            t1 = time.perf_counter()

            metrics = {
                "loss": loss,
                "iter": iter_num,
                "step": step_count,
                "epoch": train_iterator.epoch,
                "iter_time": t1 - iter_t0,
                "learning_rate": scheduler.get_last_lr()[0],
            }
            if isinstance(val_loss, torch.Tensor):
                val_loss = f"{val_loss:.3f}"
            print(
                f"Epoch {metrics['epoch'] + 1} | iter {metrics['iter']} step {metrics['step']} |"
                f" loss train: {metrics['loss']:.3f},"
                f" val: {val_loss} |"
                f" iter time: {metrics['iter_time'] * 1000:.2f} ms"
                f"{' (step)' if not is_accumulating else ''}"
            )
            trainer.log_dict(metrics, step=iter_num)

        if not is_accumulating and step_count % eval.interval == 0:
            t0 = time.perf_counter()
            val_loss = validate(model, val_dataloader, eval)
            generate_example(model, tokenizer, eval, data)
            t1 = time.perf_counter() - t0
            print(
                f"iter {iter_num}: val loss {val_loss.item():.4f}, val time: {t1 * 1000:.2f} ms"
            )
            metrics = {"val_loss": val_loss, "val_ppl": math.exp(val_loss)}
            trainer.log_dict(metrics, step=iter_num)
            trainer.barrier()

        if (
            train.save_interval is not None
            and not is_accumulating
            and step_count % train.save_interval == 0
        ):
            checkpoint_file = out_dir / f"step-{step_count:06d}" / "lit_model.pth.lora"
            checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
            save_lora_checkpoint(model, checkpoint_file)
            if model.global_rank == 0:
                copy_config_files(checkpoint_dir, checkpoint_file.parent)
                save_hyperparameters(setup, checkpoint_file.parent)
                save_prompt_style(data.prompt_style, checkpoint_file.parent)

    total_token_counts = {}
    for key in token_counts:
        total = trainer.all_reduce(token_counts[key], reduce_op="sum")
        total_token_counts[key] = total.item()

    return total_token_counts


# FSDP has issues with `inference_mode`
@torch.no_grad()
def validate(
    model,
    val_dataloader: DataLoader,
) -> torch.Tensor:
    model.eval()
    losses = torch.zeros(min(len(val_dataloader), eval.max_iters))
    for k, batch in enumerate(val_dataloader):
        if k >= eval.max_iters:
            break
        input_ids, targets = batch["input_ids"], batch["labels"]
        logits = model(input_ids)
        losses[k] = chunked_cross_entropy(
            logits[..., :-1, :], targets[..., 1:], chunk_size=0
        )

    val_loss = losses.mean()
    model.train()
    return val_loss


def get_lr_scheduler(optimizer, warmup_steps: int, max_steps: int):
    # linear warmup followed by cosine annealing
    scheduler1 = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: step / warmup_steps
    )
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=(max_steps - warmup_steps)
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, [scheduler1, scheduler2], milestones=[warmup_steps]
    )
