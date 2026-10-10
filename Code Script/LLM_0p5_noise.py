"""
Standalone Python conversion of infantCryLLM1.ipynb.

Architecture:
    WAV audio -> frozen Wav2Vec2 -> audio projection -> TinyLlama + LoRA
    -> five-class classification head

This file contains Python code only. Run environment creation and pip commands
in the terminal, not inside this file.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import librosa
import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Wav2Vec2Model,
    Wav2Vec2Processor,
)


DEFAULT_DATASET_PATH = (
    "/home/falak/rock-sam-vit/sheraz/"
    "babycry_audiodatset_6000samples"
)
DEFAULT_OUTPUT_DIR = (
    "/home/falak/rock-sam-vit/sheraz/"
    "outputs/infant_cry_llm_0p5"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Wav2Vec2 + TinyLlama infant-cry classifier."
    )
    parser.add_argument(
        "--dataset-path",
        default=DEFAULT_DATASET_PATH,
        help="Folder containing one subfolder per infant-cry class.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory used to save the trained model and results.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--test-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--noise-percent",
        type=float,
        default=0.5,
        help="Gaussian training-noise RMS as a percentage of waveform RMS.",
    )
    return parser.parse_args()


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"Random seed: {seed}")


def select_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"CUDA memory: "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )
    else:
        print(
            "Warning: TinyLlama training on CPU will be extremely slow. "
            "A CUDA GPU is strongly recommended."
        )

    return device


def scan_dataset(dataset_path: str) -> Tuple[List[str], List[int], List[str]]:
    dataset_root = Path(dataset_path).expanduser().resolve()

    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset folder was not found:\n{dataset_root}"
        )

    classes = sorted(
        folder.name
        for folder in dataset_root.iterdir()
        if folder.is_dir()
    )

    if len(classes) != 5:
        raise ValueError(
            f"Expected exactly 5 class folders, but found {len(classes)}: "
            f"{classes}"
        )

    audio_files: List[str] = []
    labels: List[int] = []

    for label_id, class_name in enumerate(classes):
        class_folder = dataset_root / class_name

        class_files = sorted(
            file_path
            for file_path in class_folder.rglob("*")
            if file_path.is_file()
            and file_path.suffix.lower() == ".wav"
        )

        if not class_files:
            raise ValueError(
                f"No .wav files were found in class folder: {class_folder}"
            )

        audio_files.extend(str(path) for path in class_files)
        labels.extend([label_id] * len(class_files))

    print(f"\nClasses: {classes}")
    print(f"Number of classes: {len(classes)}")
    print(f"Total audio files: {len(audio_files)}")

    class_counts = Counter(labels)
    print("\nDataset balance")
    print("-" * 42)
    for class_id, class_name in enumerate(classes):
        print(f"{class_id}: {class_name:20s} {class_counts[class_id]:5d}")
    print("-" * 42)

    if len(audio_files) != 6000:
        print(
            f"Warning: expected 6000 audio files, but found "
            f"{len(audio_files)}."
        )

    return audio_files, labels, classes



def add_rms_noise(audio: np.ndarray, noise_percent: float, seed: int | None = None) -> np.ndarray:
    """Add zero-mean Gaussian noise at a percentage of the waveform RMS."""
    if noise_percent <= 0:
        return audio

    rng = np.random.default_rng(seed)
    rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))
    if rms <= 0:
        return audio

    noise = rng.standard_normal(audio.shape).astype(np.float32)
    noise_rms = float(np.sqrt(np.mean(np.square(noise), dtype=np.float64)))
    if noise_rms <= 0:
        return audio

    target_noise_rms = rms * (noise_percent / 100.0)
    noise = noise * (target_noise_rms / noise_rms)
    return audio + noise


class BabyCryDataset(Dataset):
    def __init__(
        self,
        files: Sequence[str],
        labels: Sequence[int],
        wav_processor: Wav2Vec2Processor,
        sample_rate: int = 16_000,
        duration: float = 5.0,
        noise_percent: float = 0.0,
        augment_noise: bool = False,
    ) -> None:
        self.files = list(files)
        self.labels = list(labels)
        self.wav_processor = wav_processor
        self.sample_rate = sample_rate
        self.target_length = int(sample_rate * duration)
        self.noise_percent = float(noise_percent)
        self.augment_noise = bool(augment_noise)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        audio_path = self.files[index]

        try:
            audio, _ = librosa.load(
                audio_path,
                sr=self.sample_rate,
                mono=True,
            )
        except Exception as error:
            raise RuntimeError(
                f"Could not load audio file: {audio_path}"
            ) from error

        if len(audio) < self.target_length:
            audio = np.pad(
                audio,
                (0, self.target_length - len(audio)),
                mode="constant",
            )
        else:
            audio = audio[: self.target_length]

        if self.augment_noise and self.noise_percent > 0:
            audio = add_rms_noise(audio, self.noise_percent)

        input_values = self.wav_processor(
            audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        ).input_values.squeeze(0)

        return {
            "audio": input_values,
            "label": torch.tensor(
                self.labels[index],
                dtype=torch.long,
            ),
        }


class AudioProjection(nn.Module):
    def __init__(self, audio_dim: int, llm_dim: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(audio_dim, llm_dim),
            nn.LayerNorm(llm_dim),
        )

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        return self.projection(audio_features)


class ClassificationHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(512, num_classes),
        )

    def forward(self, representation: torch.Tensor) -> torch.Tensor:
        return self.classifier(representation)


class AudioLLMClassifier(nn.Module):
    def __init__(
        self,
        wav_encoder: Wav2Vec2Model,
        projector: AudioProjection,
        llm: nn.Module,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.wav_encoder = wav_encoder
        self.projector = projector
        self.llm = llm
        self.classifier = ClassificationHead(
            hidden_dim=llm.config.hidden_size,
            num_classes=num_classes,
        )

    def get_decoder(self) -> nn.Module:
        """
        Return TinyLlama's decoder without its vocabulary-output head.

        This reduces memory because classification does not require
        language-model vocabulary logits.
        """
        base_model = (
            self.llm.get_base_model()
            if hasattr(self.llm, "get_base_model")
            else self.llm
        )

        if not hasattr(base_model, "model"):
            raise AttributeError(
                "The selected LLM does not expose its decoder as `.model`."
            )

        return base_model.model

    def forward(
        self,
        audio: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Wav2Vec2 is frozen.
        with torch.no_grad():
            audio_features = self.wav_encoder(
                audio
            ).last_hidden_state

        # Convert the Wav2Vec2 sequence into one audio token.
        audio_features = audio_features.mean(dim=1)
        audio_token = self.projector(audio_features)

        embedding_layer = self.llm.get_input_embeddings()
        llm_dtype = embedding_layer.weight.dtype

        audio_token = audio_token.to(llm_dtype).unsqueeze(1)
        text_embeddings = embedding_layer(input_ids)

        inputs_embeds = torch.cat(
            [audio_token, text_embeddings],
            dim=1,
        )

        audio_attention_mask = torch.ones(
            (attention_mask.size(0), 1),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        combined_attention_mask = torch.cat(
            [audio_attention_mask, attention_mask],
            dim=1,
        )

        decoder_outputs = self.get_decoder()(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_attention_mask,
            use_cache=False,
            return_dict=True,
        )

        hidden_states = decoder_outputs.last_hidden_state

        # TinyLlama is causal. The final prompt token can attend to both the
        # preceding projected audio token and the complete prompt.
        final_token_indices = combined_attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(
            hidden_states.size(0),
            device=hidden_states.device,
        )
        representation = hidden_states[
            batch_indices,
            final_token_indices,
        ]

        logits = self.classifier(representation.float())
        return logits


def load_models(
    device: torch.device,
    num_classes: int,
) -> Tuple[
    AudioLLMClassifier,
    Wav2Vec2Processor,
    AutoTokenizer,
]:
    wav2vec_name = "facebook/wav2vec2-base"
    llm_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    print(f"\nLoading Wav2Vec2 processor: {wav2vec_name}")
    wav_processor = Wav2Vec2Processor.from_pretrained(
        wav2vec_name
    )

    print(f"Loading Wav2Vec2 encoder: {wav2vec_name}")
    wav_encoder = Wav2Vec2Model.from_pretrained(
        wav2vec_name
    )

    for parameter in wav_encoder.parameters():
        parameter.requires_grad = False

    wav_encoder.eval()

    print(f"Loading tokenizer: {llm_name}")
    tokenizer = AutoTokenizer.from_pretrained(llm_name)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    if device.type == "cuda":
        llm_dtype = (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )
    else:
        llm_dtype = torch.float32

    print(f"Loading TinyLlama: {llm_name}")
    print(f"TinyLlama dtype: {llm_dtype}")

    llm = AutoModelForCausalLM.from_pretrained(
        llm_name,
        torch_dtype=llm_dtype,
        low_cpu_mem_usage=True,
    )
    llm.config.use_cache = False

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    llm = get_peft_model(llm, lora_config)

    audio_projector = AudioProjection(
        audio_dim=wav_encoder.config.hidden_size,
        llm_dim=llm.config.hidden_size,
    )

    model = AudioLLMClassifier(
        wav_encoder=wav_encoder,
        projector=audio_projector,
        llm=llm,
        num_classes=num_classes,
    )

    model.to(device)
    model.wav_encoder.eval()

    print("\nLoRA parameters")
    llm.print_trainable_parameters()

    return model, wav_processor, tokenizer


def create_data_loaders(
    train_files: Sequence[str],
    test_files: Sequence[str],
    train_labels: Sequence[int],
    test_labels: Sequence[int],
    wav_processor: Wav2Vec2Processor,
    sample_rate: int,
    duration: float,
    batch_size: int,
    test_batch_size: int,
    num_workers: int,
    device: torch.device,
    noise_percent: float = 0.0,
) -> Tuple[DataLoader, DataLoader]:
    train_dataset = BabyCryDataset(
        train_files,
        train_labels,
        wav_processor=wav_processor,
        sample_rate=sample_rate,
        duration=duration,
        noise_percent=noise_percent,
        augment_noise=noise_percent > 0,
    )

    test_dataset = BabyCryDataset(
        test_files,
        test_labels,
        wav_processor=wav_processor,
        sample_rate=sample_rate,
        duration=duration,
    )

    loader_options = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **loader_options,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        **loader_options,
    )

    print(f"\nTraining samples: {len(train_dataset)}")
    print(f"Testing samples:  {len(test_dataset)}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Testing batches:  {len(test_loader)}")

    return train_loader, test_loader


def prepare_prompt(
    tokenizer: AutoTokenizer,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    prompt = "Classify this infant cry into one of the five categories."

    encoded_prompt = tokenizer(
        prompt,
        return_tensors="pt",
        padding=False,
        truncation=True,
    )

    return (
        encoded_prompt["input_ids"].to(device),
        encoded_prompt["attention_mask"].to(device),
    )


def repeat_prompt(
    prompt_input_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return (
        prompt_input_ids.expand(batch_size, -1),
        prompt_attention_mask.expand(batch_size, -1),
    )


def count_parameters(model: nn.Module) -> None:
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(f"\nTrainable parameters: {trainable:,}")
    print(f"Total parameters:     {total:,}")
    print(f"Trainable percentage: {100 * trainable / total:.4f}%")


def verify_forward_pass(
    model: AudioLLMClassifier,
    train_loader: DataLoader,
    prompt_input_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    device: torch.device,
) -> None:
    model.eval()
    batch = next(iter(train_loader))

    audio = batch["audio"].to(device)
    input_ids, attention_mask = repeat_prompt(
        prompt_input_ids,
        prompt_attention_mask,
        batch_size=audio.size(0),
    )

    with torch.no_grad():
        outputs = model(
            audio,
            input_ids,
            attention_mask,
        )

    print(f"\nForward output shape: {tuple(outputs.shape)}")
    print(f"Contains NaN: {torch.isnan(outputs).any().item()}")
    print(f"Contains Inf: {torch.isinf(outputs).any().item()}")
    print(f"Maximum logit: {outputs.max().item():.6f}")
    print(f"Minimum logit: {outputs.min().item():.6f}")


def train_model(
    model: AudioLLMClassifier,
    train_loader: DataLoader,
    tokenizer: AutoTokenizer,
    prompt_input_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    output_dir: Path,
    classes: Sequence[str],
) -> List[Dict[str, float]]:
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        (
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    history: List[Dict[str, float]] = []
    start_epoch = 1

    checkpoint_path = output_dir / "latest_checkpoint.pt"
    adapter_path = (
        output_dir
        / "llm_lora_adapter"
        / "adapter_model.safetensors"
    )
    history_path = output_dir / "training_history.json"

    if checkpoint_path.exists() and adapter_path.exists():
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        print(f"Resuming from checkpoint: {checkpoint_path}")

        adapter_state = load_file(str(adapter_path), device="cpu")
        set_peft_model_state_dict(
            model.llm,
            adapter_state,
            adapter_name="default",
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )

        saved_classes = checkpoint.get("classes")
        if (
            saved_classes is not None
            and list(saved_classes) != list(classes)
        ):
            raise ValueError(
                "Checkpoint classes do not match the current dataset."
            )

        model.projector.load_state_dict(
            checkpoint["projector_state_dict"]
        )
        model.classifier.load_state_dict(
            checkpoint["classifier_state_dict"]
        )
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        for optimizer_state in optimizer.state.values():
            for key, value in optimizer_state.items():
                if torch.is_tensor(value):
                    optimizer_state[key] = value.to(device)

        start_epoch = int(checkpoint["epoch"]) + 1

        if history_path.exists():
            with history_path.open("r", encoding="utf-8") as file:
                history = json.load(file)

        print(
            f"Checkpoint loaded. Continuing from epoch "
            f"{start_epoch}/{epochs}."
        )
    else:
        print("No complete checkpoint found. Starting from epoch 1.")

    if start_epoch > epochs:
        print("Requested training epochs are already complete.")
        return history

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        model.wav_encoder.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{epochs}",
        )

        for batch in progress_bar:
            audio = batch["audio"].to(
                device,
                non_blocking=True,
            )
            labels = batch["label"].to(
                device,
                non_blocking=True,
            )

            input_ids, attention_mask = repeat_prompt(
                prompt_input_ids,
                prompt_attention_mask,
                batch_size=audio.size(0),
            )

            optimizer.zero_grad(set_to_none=True)

            outputs = model(
                audio,
                input_ids,
                attention_mask,
            )
            loss = criterion(outputs, labels)

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss detected: {loss.item()}"
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                (
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ),
                max_norm=1.0,
            )

            optimizer.step()

            predictions = outputs.argmax(dim=1)
            batch_size = labels.size(0)

            running_loss += loss.item()
            total += batch_size
            correct += (
                predictions == labels
            ).sum().item()

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                accuracy=f"{100 * correct / total:.2f}%",
            )

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total

        history.append(
            {
                "epoch": epoch,
                "train_loss": epoch_loss,
                "train_accuracy": epoch_accuracy,
            }
        )

        print("-" * 60)
        print(f"Epoch {epoch}/{epochs}")
        print(f"Training loss:     {epoch_loss:.4f}")
        print(f"Training accuracy: {epoch_accuracy:.2f}%")

        output_dir.mkdir(parents=True, exist_ok=True)

        with (output_dir / "training_history.json").open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(history, file, indent=2)

        # Save a resumable checkpoint after every epoch.
        torch.save(
            {
                "epoch": epoch,
                "projector_state_dict": model.projector.state_dict(),
                "classifier_state_dict": model.classifier.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "classes": list(classes),
            },
            output_dir / "latest_checkpoint.pt",
        )

        model.llm.save_pretrained(
            output_dir / "llm_lora_adapter"
        )
        tokenizer.save_pretrained(
            output_dir / "tokenizer"
        )

    return history


@torch.no_grad()
def evaluate_model(
    model: AudioLLMClassifier,
    test_loader: DataLoader,
    prompt_input_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    device: torch.device,
    classes: Sequence[str],
    output_dir: Path,
) -> None:
    model.eval()

    all_labels: List[int] = []
    all_predictions: List[int] = []

    for batch in tqdm(test_loader, desc="Testing"):
        audio = batch["audio"].to(
            device,
            non_blocking=True,
        )
        labels = batch["label"].to(
            device,
            non_blocking=True,
        )

        input_ids, attention_mask = repeat_prompt(
            prompt_input_ids,
            prompt_attention_mask,
            batch_size=audio.size(0),
        )

        outputs = model(
            audio,
            input_ids,
            attention_mask,
        )

        predictions = outputs.argmax(dim=1)

        all_labels.extend(labels.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())

    test_accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    report_text = classification_report(
        all_labels,
        all_predictions,
        target_names=list(classes),
        zero_division=0,
    )

    report_dictionary = classification_report(
        all_labels,
        all_predictions,
        target_names=list(classes),
        zero_division=0,
        output_dict=True,
    )

    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(len(classes))),
    )

    print(f"\nTest accuracy: {test_accuracy * 100:.2f}%")
    print("\nClassification report")
    print(report_text)
    print("Confusion matrix")
    print(matrix)

    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "classification_report.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(report_dictionary, file, indent=2)

    np.savetxt(
        output_dir / "confusion_matrix.csv",
        matrix,
        fmt="%d",
        delimiter=",",
    )

    with (output_dir / "test_summary.txt").open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            f"Test accuracy: {test_accuracy * 100:.2f}%\n\n"
        )
        file.write(report_text)
        file.write("\nConfusion matrix\n")
        file.write(np.array2string(matrix))


def main() -> None:
    args = parse_arguments()
    set_random_seed(args.seed)
    device = select_device()

    audio_files, labels, classes = scan_dataset(
        args.dataset_path
    )

    train_files, test_files, train_labels, test_labels = (
        train_test_split(
            audio_files,
            labels,
            test_size=args.test_size,
            random_state=args.seed,
            stratify=labels,
        )
    )

    model, wav_processor, tokenizer = load_models(
        device=device,
        num_classes=len(classes),
    )

    train_loader, test_loader = create_data_loaders(
        train_files=train_files,
        test_files=test_files,
        train_labels=train_labels,
        test_labels=test_labels,
        wav_processor=wav_processor,
        sample_rate=args.sample_rate,
        duration=args.duration,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        device=device,
        noise_percent=args.noise_percent,
    )

    print(
        f"Training noise augmentation: {args.noise_percent:.3f}% of waveform RMS"
    )

    prompt_input_ids, prompt_attention_mask = prepare_prompt(
        tokenizer=tokenizer,
        device=device,
    )

    count_parameters(model)

    verify_forward_pass(
        model=model,
        train_loader=train_loader,
        prompt_input_ids=prompt_input_ids,
        prompt_attention_mask=prompt_attention_mask,
        device=device,
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "classes.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                str(index): class_name
                for index, class_name in enumerate(classes)
            },
            file,
            indent=2,
        )

    wav_processor.save_pretrained(
        output_dir / "wav2vec_processor"
    )

    train_model(
        model=model,
        train_loader=train_loader,
        tokenizer=tokenizer,
        prompt_input_ids=prompt_input_ids,
        prompt_attention_mask=prompt_attention_mask,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        output_dir=output_dir,
        classes=classes,
    )

    evaluate_model(
        model=model,
        test_loader=test_loader,
        prompt_input_ids=prompt_input_ids,
        prompt_attention_mask=prompt_attention_mask,
        device=device,
        classes=classes,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    # Required for safe DataLoader behavior when run as a .py program.
    main()
