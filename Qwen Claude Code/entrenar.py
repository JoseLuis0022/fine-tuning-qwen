"""
Wrapper de entrenamiento LoRA para Qwen3-8B con barra de progreso en vivo.
Llama directamente a las funciones internas de mlx_lm.lora (sin pasar por
run(), que en esta version ignora el training_callback que se le pasa)
para poder conectar un callback propio que actualice una barra tqdm.
"""
import sys
import types

from tqdm import tqdm

from mlx_lm.lora import CONFIG_DEFAULTS, build_parser, load_dataset, train_model
from mlx_lm.tuner.callbacks import TrainingCallback
from mlx_lm.utils import load


class ProgressBarCallback(TrainingCallback):
    def __init__(self, total_iters, log_path):
        self.bar = tqdm(total=total_iters, unit="it", dynamic_ncols=True)
        self.log_file = open(log_path, "a")

    def on_train_loss_report(self, info):
        self.bar.n = info["iteration"]
        self.bar.set_postfix(
            train_loss=f"{info['train_loss']:.3f}",
            lr=f"{info['learning_rate']:.1e}",
            it_s=f"{info['iterations_per_second']:.2f}",
            mem_gb=f"{info['peak_memory']:.1f}",
            refresh=True,
        )
        self.bar.refresh()
        linea = (
            f"Iter {info['iteration']}: Train loss {info['train_loss']:.3f}, "
            f"LR {info['learning_rate']:.1e}, It/sec {info['iterations_per_second']:.2f}, "
            f"Tokens/sec {info['tokens_per_second']:.1f}, Peak mem {info['peak_memory']:.2f} GB\n"
        )
        self.log_file.write(linea)
        self.log_file.flush()

    def on_val_loss_report(self, info):
        self.bar.write(
            f"  -> Iter {info['iteration']}: Val loss {info['val_loss']:.3f} "
            f"(val took {info['val_time']:.1f}s)"
        )
        self.log_file.write(
            f"Iter {info['iteration']}: Val loss {info['val_loss']:.3f}\n"
        )
        self.log_file.flush()

    def close(self):
        self.bar.n = self.bar.total
        self.bar.refresh()
        self.bar.close()
        self.log_file.close()


def main():
    overrides = {
        "model": "./modelo-base",
        "train": True,
        "data": "./data",
        "fine_tune_type": "lora",
        "num_layers": 16,
        "batch_size": 2,
        "iters": 300,
        "val_batches": 25,
        "learning_rate": 1e-4,
        "steps_per_report": 10,
        "steps_per_eval": 50,
        "adapter_path": "./adaptadores",
        "save_every": 100,
    }

    parser = build_parser()
    args = vars(parser.parse_args([]))
    for k, v in overrides.items():
        args[k] = v
    for k, v in CONFIG_DEFAULTS.items():
        if args.get(k, None) is None:
            args[k] = v
    args = types.SimpleNamespace(**args)

    print(f"Cargando modelo desde {args.model} ...")
    model, tokenizer = load(args.model, tokenizer_config={"trust_remote_code": True})

    print("Cargando datasets ...")
    train_set, valid_set, _ = load_dataset(args, tokenizer)
    print(f"Train: {len(train_set)} ejemplos | Valid: {len(valid_set)} ejemplos")

    callback = ProgressBarCallback(args.iters, "resultados/entrenamiento.log")
    try:
        train_model(args, model, train_set, valid_set, training_callback=callback)
    finally:
        callback.close()

    print("\nEntrenamiento terminado. Adaptadores guardados en ./adaptadores/")


if __name__ == "__main__":
    sys.exit(main())
