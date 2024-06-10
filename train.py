#!/usr/bin/env python3
"""Recipe for pretraining BestRQ (TODO CITATION).
See config file for model definition.
See the readme of the recipe for advice on the pretraining that may appear
a bit challenging depending on your available resources.

To run this recipe call python train_best_rq.py best_rq.yaml --find_unused_parameters --max_grad_norm 0.0

Authors
    * Ryan Whetten 2023
"""

import os
import logging
import sys
import time
from functools import partial

import speechbrain as sb
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from hyperpyyaml import load_hyperpyyaml

from speechbrain import Stage
from speechbrain.utils.distributed import run_on_main
from speechbrain.dataio.dataloader import SaveableDataLoader
from speechbrain.dataio.sampler import DynamicBatchSampler
from mask import brq_mask_collate_fn
import wandb


logger = logging.getLogger(__name__)

class BestRQBrain(sb.core.Brain):
    
    def compute_forward(self, batch, stage):
        """Computes forward pass through BestRQ model and returns encoded and
        target embeddings as well as other metrics of interest.
        """
        # get batch and mask
        wavs, wav_lens, mask = batch
        wavs, wav_lens, mask = (
            wavs.to(self.device),
            wav_lens.to(self.device),
            mask.to(self.device),
        )

        ############### START ##############
        ### get fbanks and normalize
        feats = self.hparams.compute_features(wavs)
        current_epoch = self.hparams.epoch_counter.current
        feats = self.modules.normalize(feats, wav_lens, epoch=current_epoch)

        B, T, C = feats.shape
        divis_by = self.hparams.pad_to_divisible_by


        #### pad features
        current_dim_size = T
        dim_to_pad = 1  # Pad along the second dimension (i.e. time)

        # Calculate the amount of padding needed to make the tensor divisible by 4
        current_dim_size = feats.shape[dim_to_pad]
        padding_needed = (4 - (current_dim_size % 4)) % 4  # Ensure positive padding

        # Define the padding
        padding = [0, 0, 0, 0, 0, 0]  # Initialize padding for all dimensions
        padding[dim_to_pad * 2] = padding_needed  # Set padding for the chosen dimension

        # add in padding to features and mask
        feats = torch.nn.functional.pad(feats, padding)
        # print('features w/padding: ', feats.shape)

        # get targets from quantizer
        targets = self.modules.Quantizer(feats.view(B, feats.shape[1]//divis_by, -1))
        # print('targets: ', targets.shape)

        # generate random noise
        noise = torch.normal(
            mean=self.hparams.noise_mean, 
            std=self.hparams.noise_std, 
            size=(B, mask.shape[0], C), 
            device=self.device
        )
        # replace with random noise
        feats[:,mask,:] = noise

        #### convolutions
        src = self.modules.CNN(feats)
        # print('after cnn: ', src.shape)

        ##### transformer
        enc_out = self.modules.wrapper(src, wav_lens) # only use encoder
        # print('enc out: ', enc_out.shape)

        ##### linear
        logits = self.modules.linear(enc_out)
        # print('linear layer out: ', logits.shape)

        mask_idx = mask[::divis_by] // divis_by
        logits[:,mask_idx,:]
        targets[:,mask_idx].shape

        if  not torch.isfinite(logits).all():
            print('feats: ', torch.isfinite(feats).all())
            print('src: ', torch.isfinite(src).all())
            print('enc: ', torch.isfinite(enc_out).all())
            print('logits: ', torch.isfinite(logits).all())
            print('targets: ', torch.isfinite(targets).all())
            # print('scaler: ', self.scaler.state_dict())
        ##### get masked region
        logits = logits[:,mask_idx,:]
        targets = targets[:,mask_idx]
        B, T, C = logits.shape
        return logits.view(B * T, C), targets.view(B*T)
    

    def compute_objectives(self, predictions, batch, stage):
        pred, targets = predictions
        if stage != sb.Stage.TRAIN and sb.utils.distributed.if_main_process():
            predicted_classes = torch.argmax(pred, dim=-1)
            correct_predictions = (predicted_classes == targets)
            accuracy = correct_predictions.sum().item() / len(correct_predictions)
            self.acc_metric.append(accuracy)
        loss = F.cross_entropy(pred, targets)

        if self.step % 100 == 0 and stage == sb.Stage.TRAIN:
            param_norms =[
                param.detach().flatten()
                for param in self.modules.parameters()
                if param is not None
            ]
            param_norms = torch.cat(param_norms).norm()

            grads = [
                param.grad.detach().flatten()
                for param in self.modules.parameters()
                if param.grad is not None
            ]
            cb_usage = targets.unique().shape[0] / self.hparams.cb_vocab
            norm = torch.cat(grads).norm()
            
            if sb.utils.distributed.if_main_process():
                wandb.log({
                    'loss': loss,
                    'param_norm':param_norms,
                    'grad_norm':norm,
                    'cb_usage':cb_usage,
                    **self.scaler.state_dict()
                })
  
        return loss

    def on_fit_batch_end(self, batch, outputs, loss, should_step):
        """ Called after fit_batch(), updates learning rate and does per-step logging. """
        if should_step:
            self.hparams.noam_annealing(self.optimizer)

        # Perform step-wise logging
        if (
            hasattr(self.hparams, "log_interval")
            and self.optimizer_step % self.hparams.log_interval == 0
        ):

            # Create a dictionary and fill it with everything we
            # want to log such as contrastive loss, diversity loss,
            # learning rate etc.
            log_dct = {}
            # log_dct = {
            #     k: (v.item() if isinstance(v, torch.Tensor) else v)
            #     for k, v in objectives.items()
            # }
            current_lr = self.optimizer.param_groups[0]["lr"]
            log_dct["steps"] = self.optimizer_step
            log_dct["lr"] = current_lr
            log_dct["avg_loss"] = self.avg_train_loss

            if hasattr(self, "time_last_log"):
                run_time_since_last_log = time.time() - self.time_last_log
                log_dct["run_time"] = run_time_since_last_log
            self.time_last_log = time.time()

            if sb.utils.distributed.if_main_process():
                self.hparams.train_steps_logger.log_stats(stats_meta=log_dct,)

    # def evaluate_batch(self, batch, stage):
    #     """ Returns accuracy on contrastive objective. """
    #     out = self.compute_forward(batch, stage=stage)
    #     objectives = self.compute_objectives(out, batch, stage=stage)
    #     return objectives["backprop_loss"].detach().cpu()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.acc_metric = []

    def on_stage_end(self, stage, stage_loss, epoch=None):

        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats

        if stage == sb.Stage.VALID:
            if self.acc_metric:
                # print('acc_metric', self.acc_metric)
                stage_stats["accuracy"] = sum(self.acc_metric) / len(
                    self.acc_metric
                )

            self.hparams.train_stage_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    "steps": self.optimizer_step,
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "VRAM": torch.cuda.max_memory_allocated(), 
                },
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )

            self.checkpointer.save_and_keep_only(
                end_of_epoch=True,
                num_to_keep=5,
                meta={"valid_loss": stage_loss},
            )


def dataio_prepare(hparams):
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"], replacements={"data_root": data_folder},
    )

    # We remove longer and shorter files from the train.
    train_data = train_data.filtered_sorted(
        sort_key="duration",
        key_max_value={"duration": hparams["avoid_if_longer_than"]},
        key_min_value={"duration": hparams["avoid_if_shorter_than"]},
    )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"], replacements={"data_root": data_folder},
    )

    datasets = [train_data, valid_data]

    def get_output_lengths(input_lengths):
        """ Function to get the output length of the feature extractor this is
            necessery to compute the masks of BestRQ.
        """
        sr = hparams["sample_rate"]
        hop_length = hparams["hop_length"]

        return (input_lengths // (sr*hop_length / 1000) + 1).to(torch.long)

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig"])

    # We create the DynamicBatch Sampler
    train_sampler = DynamicBatchSampler(
        train_data,
        hparams["seconds_per_batch"],
        num_buckets=hparams["train_num_buckets"],
        length_func=lambda x: x["duration"],
        batch_ordering="random",
        shuffle=True,
    )

    # We define the custom collation function that is necessary for best-rq to
    # generate masks.
    brq_mask_collate_fn_partial = partial(
        brq_mask_collate_fn,
        get_out_len_fn=get_output_lengths,
        mask_prob=hparams["mask_prob"],
        mask_length=hparams["mask_length"],
        n_mels=hparams["n_mels"],
    )

    train_loader_kwargs = {
        "batch_sampler": train_sampler,
        "collate_fn": brq_mask_collate_fn_partial,
        "num_workers": hparams["train_dataloader_options"]["num_workers"],
        "pin_memory": True,
    }

    valid_loader = SaveableDataLoader(
        valid_data,
        collate_fn=brq_mask_collate_fn_partial,
        num_workers=hparams["test_dataloader_options"]["num_workers"],
        batch_size=hparams["test_dataloader_options"]["batch_size"],
        pin_memory=True,
    )
    
    if 'wandb_offline' in hparams and hparams['wandb_offline']:
        os.environ["WANDB_MODE"] = "offline"
    
    wandb.init(
        # Set the project where this run will be logged
        project=hparams["project_name"],
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        # name=f"exp_0", 
        name=hparams["experiment_name"],
        # Track hyperparameters and run metadata
        config={
        "learning_rate": hparams["lr"],
        "architecture": hparams["architecture"],
        "dataset": "Libri",
        "number_of_epochs": hparams["number_of_epochs"],
        "precision": hparams["precision"],
        "max_grad_norm": hparams["max_grad_norm"],
        "grad_accumulation_factor": hparams["grad_accumulation_factor"],
        "seconds_per_batch": hparams["seconds_per_batch"], # Fits in a 32GB GPUs (V100)
        "train_num_buckets": hparams["train_num_buckets"],
        "d_model": hparams["d_model"],
        "d_ffn": hparams["d_ffn"],
        "nhead": hparams["nhead"],
        "num_encoder_layers": hparams["num_encoder_layers"],
        "seed": hparams["seed"],
        "cb_vocab": hparams["cb_vocab"],
        "p_input": hparams["p_input"],
        "cb_dim": hparams["cb_dim"],
        "mask_prob": hparams["mask_prob"],
    })

    return train_data, valid_loader, train_loader_kwargs

def main():
    logger.setLevel(logging.INFO)
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    hparams.update(run_opts)

    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    from librispeech_prepare import prepare_librispeech

    run_on_main(
        prepare_librispeech,
        kwargs={
            "data_folder": hparams["data_folder"],
            "tr_splits": hparams["train_splits"],
            "dev_splits": hparams["dev_splits"],
            "te_splits": hparams["test_splits"],
            "save_folder": hparams["output_folder"],
            "merge_lst": hparams["train_splits"],
            "merge_name": "train.csv",
            "skip_prep": hparams["skip_prep"],
        },
    )

    # Part that matters starts here.
    train_dataset, valid_loader, train_loader_kwargs = dataio_prepare(hparams)

    brain = BestRQBrain(
        modules=hparams["modules"],
        opt_class=hparams["optimizer"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    # with torch.autograd.detect_anomaly():
    brain.fit(
        brain.hparams.epoch_counter,
        train_dataset,
        valid_loader,
        train_loader_kwargs=train_loader_kwargs,
        progressbar=True,
    )
    wandb.finish()


if __name__ == "__main__":
    main()
