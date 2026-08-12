import json
import logging
import os
import subprocess
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass, field
import datasets

import accelerate
import transformers
from transformers import (
    AutoConfig, AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback, TrainerCallback,
    HfArgumentParser
)


os.environ['TOKENIZERS_PARALLELISM'] = 'false'

logger_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
log_lvl = logging.INFO
logging.basicConfig(format=logger_fmt, level=log_lvl)
logger = logging.getLogger('')

logger.info(f"CUDA DEVICE COUNT: {torch.cuda.device_count()}")

def split_token_ids_into_segments(token_ids, tokens_per_segment=None):
    """v5: chunk a flat token-id list into segments of `tokens_per_segment` tokens.

    Setting tokens_per_segment=1 reduces to 1-symbol-per-segment (recovers
    token-level recurrence in the v5 ablations). None ⇒ single segment.
    """
    if tokens_per_segment is None or tokens_per_segment <= 0:
        return [token_ids]
    return [token_ids[i:i + tokens_per_segment]
            for i in range(0, len(token_ids), tokens_per_segment)]


def collate_fn(batch):
    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False)

    segments_batch = []
    for sample in batch:
        context = sample['context']

        perform_memory_task = torch.rand(1) < args.memory_task_freq
        if perform_memory_task and args.memory_task == "reconstruct":
            query = '!?'
            target = '!?' + context[2:-2]
        elif perform_memory_task and args.memory_task == "continue":
            query_start_ind = torch.randint(0, len(context) - args.memory_key_size - args.memory_value_size - 4, (1,))
            query = '!?' + context[query_start_ind:query_start_ind + args.memory_key_size]
            target = '!?' + context[query_start_ind + args.memory_key_size:query_start_ind + args.memory_key_size + args.memory_value_size]
        else:
            query = sample['query']
            target = sample['target']

        query_ids  = encode(query)
        target_ids = encode(target)
        qt_ids     = query_ids + target_ids
        context_ids_full = encode(context)
        # Drop trailing '|' segment-terminator so context length is a multiple
        # of tokens_per_segment (v5p1 requires uniform context segments). The
        # '|' is re-attached as the leading token of the query.
        sep_id = encode('|')
        if len(sep_id) == 1 and context_ids_full and context_ids_full[-1] == sep_id[0]:
            context_ids_full = context_ids_full[:-1]
            qt_ids = sep_id + qt_ids
        context_chunks = split_token_ids_into_segments(
            context_ids_full, tokens_per_segment=args.tokens_per_segment
        )

        segments = []
        for chunk_ids in context_chunks:
            seg = {
                'input_ids':      torch.tensor(chunk_ids, dtype=torch.long),
                'attention_mask': torch.ones(len(chunk_ids), dtype=torch.long),
                'labels':         torch.full((len(chunk_ids),), -100, dtype=torch.long),
                'labels_mask':    torch.zeros(len(chunk_ids), dtype=torch.bool),
            }
            segments.append(seg)

        qt_input_ids      = torch.tensor(qt_ids, dtype=torch.long)
        qt_attention_mask = torch.ones(len(qt_ids), dtype=torch.long)
        labels = torch.full((len(qt_ids),), -100, dtype=torch.long)
        if len(target_ids) > 0:
            labels[-len(target_ids):] = torch.tensor(target_ids, dtype=torch.long)
            labels_mask = torch.zeros(len(qt_ids), dtype=torch.bool)
            labels_mask[-len(target_ids) - 1:] = True
        else:
            labels_mask = torch.zeros(len(qt_ids), dtype=torch.bool)
        segments.append({
            'input_ids':      qt_input_ids,
            'attention_mask': qt_attention_mask,
            'labels':         labels,
            'labels_mask':    labels_mask,
        })
        segments_batch.append(segments)

    # Pad segments across the batch
    batch_segments = []
    num_segments   = len(segments_batch[0])
    id_pad_value   = tokenizer.pad_token_id if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None else 0
    for i in range(num_segments):
        input_ids    = pad_sequence([s[i]['input_ids']    for s in segments_batch], batch_first=True, padding_value=id_pad_value)
        attention_mask = pad_sequence([s[i]['attention_mask'] for s in segments_batch], batch_first=True, padding_value=0)
        labels       = pad_sequence([s[i]['labels']       for s in segments_batch], batch_first=True, padding_value=-100)
        labels_mask  = pad_sequence([s[i]['labels_mask']  for s in segments_batch], batch_first=True, padding_value=False)
        batch_segments.append({
            'input_ids':      input_ids,
            'attention_mask': attention_mask,
            'labels':         labels,
            'labels_mask':    labels_mask,
        })

    full_labels = torch.cat([s['labels'] for s in batch_segments], dim=1)
    return {"segments": batch_segments, "labels": full_labels}


def compute_metrics_fn(eval_pred, ignore_token_ids, tokenizer):
    predictions, labels, inputs = eval_pred.predictions, eval_pred.label_ids, eval_pred.inputs
    logits = predictions[..., :-1, :]
    labels = labels[..., 1:]
    preds  = np.argmax(logits, axis=-1)

    mask = (labels != -100)
    for t_id in ignore_token_ids:
        mask &= (labels != t_id)

    accuracy = (preds[mask] == labels[mask]).mean()

    decoded_labels   = [tokenizer.decode(label[label != -100], skip_special_tokens=True).replace(' ', '') for label in labels]
    memory_task_mask = [decoded_labels[i][:2] == '!?' for i in range(len(decoded_labels))]

    exact_match_memory_task = np.mean([
        np.all(preds[i][mask[i]] == labels[i][mask[i]])
        for i in range(len(preds))
        if np.any(mask[i]) and memory_task_mask[i]
    ]) if any(memory_task_mask) else float('nan')

    exact_match_base = np.mean([
        np.all(preds[i][mask[i]] == labels[i][mask[i]])
        for i in range(len(preds))
        if np.any(mask[i]) and not memory_task_mask[i]
    ]) if any(not m for m in memory_task_mask) else float('nan')

    n_samples = 5
    for pred, label, inp in zip(preds[:n_samples], labels[:n_samples], inputs[:n_samples]):
        m = (label != -100)
        inp[inp == -100] = 0
        label[label == -100] = 0
        print('i:', tokenizer.decode(inp, skip_special_tokens=True).replace(' ', ''))
        print('p:', tokenizer.decode(pred[m], skip_special_tokens=True).replace(' ', ''))
        print('t:', tokenizer.decode(label[m], skip_special_tokens=True).replace(' ', ''))
        print('-' * 50)

    res = {
        "token_accuracy": float(accuracy),
        "exact_match":    float(exact_match_base),
    }
    res[f"exact_match_{args.memory_task}"] = float(exact_match_memory_task)
    return res


class StopOnMetricValue(TrainerCallback):
    def __init__(self, metric_name, value, higher_is_better=True):
        self.metric_name      = metric_name
        self.value            = value
        self.higher_is_better = higher_is_better

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        metric_to_check = self.metric_name if self.metric_name.startswith("eval_") else f"eval_{self.metric_name}"
        metric_value = metrics.get(metric_to_check)
        if metric_value is None:
            return
        op = np.greater_equal if self.higher_is_better else np.less_equal
        if op(metric_value, self.value):
            control.should_training_stop = True
            logger.info(f'metric {self.metric_name}={metric_value:.4f} >= {self.value:.4f}, stopping training.')


class CustomTrainer(Trainer):
    def create_scheduler(self, num_training_steps, optimizer=None):
        num_training_steps = int(num_training_steps / 0.9)
        return super().create_scheduler(num_training_steps, optimizer)

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        for cb in self.callback_handler.callbacks:
            if isinstance(cb, EarlyStoppingCallback):
                logs['patience'] = cb.early_stopping_patience_counter
                break
        return super().log(logs, start_time)


class ClearMLMetricsCallback(TrainerCallback):
    """Report every numeric Trainer log without creating tasks on worker ranks."""

    def __init__(self, task=None):
        self.clearml_logger = task.get_logger() if task is not None else None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.clearml_logger is None or not state.is_world_process_zero or not logs:
            return
        for name, value in logs.items():
            if not isinstance(value, (int, float, np.number)) or not np.isfinite(value):
                continue
            if "_" in name:
                title, series = name.split("_", 1)
            else:
                title, series = "train", name
            self.clearml_logger.report_scalar(
                title=title, series=series, value=float(value),
                iteration=int(state.global_step),
            )


def _git_value(*args):
    try:
        result = subprocess.run(
            ["git", *args], check=True, capture_output=True, text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def init_clearml(experiment_args, accelerator):
    """Create one ClearML task for the whole distributed training run."""
    if not experiment_args.clearml_project or not accelerator.is_main_process:
        return None
    try:
        from clearml import Task
    except ImportError as exc:
        raise RuntimeError(
            "ClearML logging was requested, but the 'clearml' package is not installed. "
            "Install it with: pip install clearml"
        ) from exc

    task_name = experiment_args.clearml_task_name or Path(experiment_args.exp_path).name
    task = Task.init(
        project_name=experiment_args.clearml_project,
        task_name=task_name,
        output_uri=experiment_args.clearml_output_uri,
        reuse_last_task_id=False,
        auto_connect_arg_parser=False,
        auto_connect_frameworks=True,
        auto_resource_monitoring=True,
        auto_connect_streams=True,
    )
    task.connect(dict(vars(experiment_args)), name="Arguments")
    commit = _git_value("rev-parse", "HEAD")
    branch = _git_value("branch", "--show-current") or _git_value(
        "rev-parse", "--abbrev-ref", "HEAD"
    )
    status = _git_value("status", "--porcelain")
    repository = {
        "commit": commit or "unknown",
        "branch": branch or "unknown",
        "version": _git_value("describe", "--always", "--tags", "--dirty") or "unknown",
        "is_dirty": bool(status) if status is not None else "unknown",
    }
    task.connect(repository, name="Repository")
    if experiment_args.clearml_tags:
        task.set_tags([tag.strip() for tag in experiment_args.clearml_tags.split(",") if tag.strip()])
    task.get_logger().report_text(
        "Repository: branch={branch}, commit={commit}, version={version}, "
        "dirty={is_dirty}".format(**repository)
    )
    return task


@dataclass
class ExperimentArgs:
    exp_path:                 str            = field()
    per_device_batch_size:    int            = field()
    data_path:                str            = field(default='./data/N2-K4V4-S4(32-64)_1M')
    tokenizer_path:           str            = field(default='./tokenizers/kv_alphabet_62/')
    gradient_accumulation_steps: Optional[int]   = field(default=1)
    total_batch_size:         Optional[int]  = field(default=None)
    metric_for_best_model:    Optional[str]  = field(default='token_accuracy')
    warmup_steps:             Optional[int]  = field(default=1000)
    max_steps:                Optional[int]  = field(default=50000)
    logging_steps:            Optional[int]  = field(default=100)
    eval_steps:               Optional[int]  = field(default=100)
    weight_decay:             Optional[float]= field(default=0.0)
    learning_rate:            Optional[float]= field(default=1e-04)
    lr_scheduler_type:        Optional[str]  = field(default='constant_with_warmup')
    early_stopping_patience:  Optional[int]  = field(default=50)
    seed:                     Optional[int]  = field(default=142)
    base_model:               Optional[str]  = field(default='gpt2')
    n_layer:                  Optional[int]  = field(default=4)
    n_head:                   Optional[int]  = field(default=1)
    n_embd:                   Optional[int]  = field(default=128)
    # GDN / FLA parameters
    fla_layer:                Optional[str]  = field(default='GatedDeltaNet')
    state_size:               Optional[int]  = field(default=32)    # num_heads * head_dim
    expand_v:                 Optional[float]= field(default=2.0)
    conv_kernel:              Optional[int]  = field(default=4)
    # v5 memory-path parameters
    num_memory_vectors:       Optional[int]  = field(default=1)             # M
    write_mode:               Optional[str]  = field(default='cross_attn')  # 'identity'|'pool'|'cross_attn'
    read_mode:                Optional[str]  = field(default='cross_attn')  # 'identity'|'unpool'|'cross_attn'|'gdn_readout'
    write_value_dim:          Optional[int]  = field(default=None)          # None = model hidden_size
    num_memory_heads:         Optional[int]  = field(default=1)
    use_parallel_prefill:     Optional[bool] = field(default=True)          # v5p3: parallel-prefill context pass (equivalent to recurrent)
    # Dataset / task
    memory_task_freq:         Optional[float]= field(default=0.0)
    memory_task:              Optional[str]  = field(default=None)
    memory_key_size:          Optional[int]  = field(default=4)
    memory_value_size:        Optional[int]  = field(default=4)
    model_cpt:                Optional[str]  = field(default=None)
    tokens_per_segment:       Optional[int]  = field(default=None)          # v5: replaces pairs_per_segment; 1 ⇒ token-level recurrence
    n_pairs:                  Optional[int]  = field(default=None)
    n_keys:                   Optional[int]  = field(default=None)
    n_values:                 Optional[int]  = field(default=None)
    # ClearML (empty project disables integration)
    clearml_project:          Optional[str]  = field(default=None)
    clearml_task_name:        Optional[str]  = field(default=None)
    clearml_tags:             Optional[str]  = field(default=None)  # comma-separated
    clearml_output_uri:       Optional[str]  = field(default=None)


if __name__ == '__main__':
    parser = HfArgumentParser(ExperimentArgs)
    args = parser.parse_args_into_dataclasses()[0]

    accel = accelerate.Accelerator()
    from accelerate.logging import get_logger
    logger = get_logger('')
    transformers.utils.logging.set_verbosity(log_lvl)

    logger.info(f'num processes: {accel.num_processes}')
    logger.info(f'mixed precision: {accel.mixed_precision}')

    clearml_task = init_clearml(args, accel)

    if accel.is_main_process:
        Path(args.exp_path).mkdir(parents=True, exist_ok=True)
        json.dump({'cli_args': dict(vars(args))},
                  open(os.path.join(args.exp_path, 'config.json'), 'w'), indent=4)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    # Build base model config
    if args.base_model == 'gpt2':
        config = AutoConfig.from_pretrained('gpt2')
        config.n_layer = args.n_layer
        config.n_head  = args.n_head
        config.n_embd  = args.n_embd
    elif args.base_model == 'pythia':
        config = AutoConfig.from_pretrained('EleutherAI/pythia-160m')
        config.num_hidden_layers  = args.n_layer
        config.num_attention_heads = args.n_head
        config.hidden_size         = args.n_embd
        config.intermediate_size   = args.n_embd * 4
    elif args.base_model == 'llama':
        config = AutoConfig.from_pretrained('NousResearch/Llama-3.2-1B')
        config.num_hidden_layers     = args.n_layer
        config.num_attention_heads   = args.n_head
        config.num_key_value_heads   = args.n_head
        config.hidden_size           = args.n_embd
        config.head_dim              = args.n_embd // args.n_head
        config.intermediate_size     = args.n_embd * 4
    else:
        raise ValueError(f'Unsupported base_model: {args.base_model}')

    config.torch_dtype   = "float32"
    config.vocab_size    = tokenizer.vocab_size
    config.pad_token_id  = tokenizer.convert_tokens_to_ids('[PAD]')
    config.bos_token_id  = tokenizer.convert_tokens_to_ids('[BOS]')
    config.eos_token_id  = tokenizer.convert_tokens_to_ids('[EOS]')

    from modeling_rmt.huggingface_rmm_v5p4 import RecurrentMemoryBase, RecurrentMemoryConfig

    head_dim = args.state_size // args.n_head

    rmm_config = RecurrentMemoryConfig(
        base_model_config  = config,
        fla_layer_name     = args.fla_layer,
        num_heads          = args.n_head,
        head_dim           = head_dim,
        expand_v           = args.expand_v,
        conv_size          = args.conv_kernel,
        # v5 memory path
        num_memory_vectors = args.num_memory_vectors,
        write_mode         = args.write_mode,
        read_mode          = args.read_mode,
        write_value_dim    = args.write_value_dim,
        num_memory_heads   = args.num_memory_heads,
        use_parallel_prefill = args.use_parallel_prefill,
        max_n_segments     = 10,
        think_token_id     = tokenizer.convert_tokens_to_ids('[THINK]'),
        answer_token_id    = tokenizer.convert_tokens_to_ids('[ANSWER]'),
        bos_token_id       = tokenizer.convert_tokens_to_ids('[BOS]'),
        eos_token_id       = tokenizer.convert_tokens_to_ids('[EOS]'),
    )
    model = RecurrentMemoryBase(rmm_config)
    model.main_input_name = 'labels'

    if args.model_cpt and args.model_cpt != 'None':
        import os as _os
        model_cpt_path   = args.model_cpt
        use_safetensors  = False
        if _os.path.isdir(model_cpt_path):
            dir_files = _os.listdir(model_cpt_path)
            if "model_best" in dir_files:
                candidate = _os.path.join(model_cpt_path, "model_best", "pytorch_model.bin")
                if _os.path.exists(candidate):
                    model_cpt_path = candidate
                else:
                    candidate_st = _os.path.join(model_cpt_path, "model_best", "model.safetensors")
                    if _os.path.exists(candidate_st):
                        model_cpt_path = candidate_st
                        use_safetensors = True
            else:
                checkpoints = sorted([el for el in dir_files if el.startswith("checkpoint-")])
                if checkpoints:
                    ckpt_dir  = _os.path.join(model_cpt_path, checkpoints[-1])
                    candidate = _os.path.join(ckpt_dir, "pytorch_model.bin")
                    if _os.path.exists(candidate):
                        model_cpt_path = candidate
                    else:
                        model_cpt_path = _os.path.join(ckpt_dir, "model.safetensors")
                        use_safetensors = True
        if use_safetensors:
            from safetensors.torch import load_model
            load_model(model, model_cpt_path, device="cuda:0")
        else:
            model.load_state_dict(torch.load(model_cpt_path, map_location='cpu'), strict=False)
        print(f"Loaded checkpoint: {model_cpt_path}")

    logger.info(f'model: {model}')
    total_parameters = sum(p.numel() for p in model.parameters())
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"parameters: {total_parameters:,} ({trainable_parameters:,} trainable)")
    if clearml_task is not None:
        clearml_task.connect({
            "class": type(model).__name__,
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "configuration": rmm_config.to_dict(),
        }, name="Model")

    # Load or generate dataset
    try:
        logger.info(f'Loading dataset from: {args.data_path}')
        dataset = datasets.load_from_disk(args.data_path)
    except Exception as e:
        logger.info(f'Generating dataset: {e}')
        from kv_dataset_utils import generate_sequence
        raw_samples = [
            generate_sequence(num_kv_pairs=args.n_pairs, n_segments=1,
                              min_segment_len=0, max_segment_len=0,
                              k_length=args.n_keys, v_length=args.n_values)
            for _ in range(1_005_000)
        ]
        import datasets as ds
        dataset = ds.Dataset.from_dict({
            'context': [s['context'] for s in raw_samples],
            'query':   [s['query']   for s in raw_samples],
            'target':  [s['target']  for s in raw_samples],
        })
        dataset = dataset.train_test_split(test_size=5_000, seed=args.seed)
        dataset = datasets.DatasetDict({"train": dataset["train"], "valid": dataset["test"]})
        dataset.save_to_disk(args.data_path)

    if clearml_task is not None:
        clearml_task.connect({
            "path": args.data_path,
            "train_samples": len(dataset["train"]),
            "validation_samples": len(dataset["valid"]),
        }, name="Dataset")

    ignore_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in ['!', '|']]

    def compute_metrics(eval_preds):
        return compute_metrics_fn(eval_preds, ignore_token_ids, tokenizer)

    if args.total_batch_size is None:
        args.total_batch_size = args.per_device_batch_size * accel.num_processes * args.gradient_accumulation_steps
    else:
        assert args.total_batch_size == args.per_device_batch_size * accel.num_processes * args.gradient_accumulation_steps

    training_args = TrainingArguments(
        output_dir  = str(args.exp_path),
        logging_dir = str(args.exp_path),
        max_steps   = args.max_steps,
        per_device_train_batch_size = args.per_device_batch_size,
        per_device_eval_batch_size  = args.per_device_batch_size,
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        warmup_steps        = args.warmup_steps,
        weight_decay        = args.weight_decay,
        learning_rate       = args.learning_rate,
        lr_scheduler_type   = args.lr_scheduler_type,
        eval_strategy       = 'steps',
        save_strategy       = 'steps',
        save_steps          = args.eval_steps,
        eval_steps          = args.eval_steps,
        logging_steps       = args.logging_steps,
        report_to           = 'tensorboard',
        metric_for_best_model   = args.metric_for_best_model,
        load_best_model_at_end  = True,
        eval_on_start           = True,
        greater_is_better       = True,
        remove_unused_columns   = False,
        include_num_input_tokens_seen = False,
        include_for_metrics     = ['inputs'],
        save_total_limit        = 1,
        dataloader_num_workers  = 4,
        dataloader_pin_memory   = True,
        seed                    = args.seed,
    )

    trainer = CustomTrainer(
        model           = model,
        args            = training_args,
        train_dataset   = dataset['train'],
        eval_dataset    = dataset['valid'],
        data_collator   = collate_fn,
        compute_metrics = compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience),
            StopOnMetricValue('exact_match',      0.99, higher_is_better=True),
            StopOnMetricValue('exact_match_None', 0.99, higher_is_better=True),
            ClearMLMetricsCallback(clearml_task),
        ],
    )
    trainer.train()
    logger.info('training done. running final evaluation...')
    metrics = trainer.evaluate(dataset['valid'])
    logger.info(f'{metrics}')
    trainer.save_metrics(split='all', metrics=metrics)
    if clearml_task is not None:
        config_path = os.path.join(args.exp_path, 'config.json')
        metrics_path = os.path.join(args.exp_path, 'all_results.json')
        if os.path.isfile(config_path):
            clearml_task.upload_artifact('experiment_config', artifact_object=config_path)
        if os.path.isfile(metrics_path):
            clearml_task.upload_artifact('final_metrics', artifact_object=metrics_path)
        clearml_task.close()
