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

import time 

logger = logging.getLogger(__name__)

WARMUP_STEPS = 10
WARMUP_MAX_DURATION = 90
EVAL_AVG_STEPS = 10
EVAL_MAX_DURATION = 90
BATCH_SIZE = 6


def compute_benchmark(batch, hparams, device="cuda"):
    # get batch and mask
    wavs, wav_lens, mask = batch
    wavs, wav_lens, mask = (
        wavs.to(device),
        wav_lens.to(device),
        mask.to(device),
    )

    feats = hparams["compute_features"](wavs)
    current_epoch = hparams["epoch_counter"].current    
    feats = hparams["modules"]["normalize"](feats, wav_lens, epoch=current_epoch)

    B, T, C = feats.shape
    divis_by = hparams["pad_to_divisible_by"]

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

    # get targets from quantizer
    targets = hparams["modules"]["Quantizer"](feats.view(B, feats.shape[1]//divis_by, -1))

    # generate random noise
    noise = torch.normal(
        mean=hparams["noise_mean"], 
        std=hparams["noise_std"], 
        size=(B, mask.shape[0], C), 
        device=device
    )

    # replace with random noise
    feats[:,mask,:] = noise


    torch.cuda.synchronize()
    start_time = time.time()    
    #### convolutions
    src = hparams["modules"]["CNN"](feats)

    ##### transformer
    enc_out = hparams["modules"]["wrapper"](src, wav_lens) # only use encoder

    # ##### linear
    logits = hparams["modules"]["linear"](enc_out)

    torch.cuda.synchronize()
    end_time = time.time()

    # mask_idx = mask[::divis_by] // divis_by
    # logits[:,mask_idx,:]
    # targets[:,mask_idx].shape

    # ##### get masked region
    # logits = logits[:,mask_idx,:]
    # targets = targets[:,mask_idx]
    # B, T, C = logits.shape
    return end_time - start_time, torch.cuda.max_memory_allocated()


def main():
    logger.setLevel(logging.INFO)
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    hparams.update(run_opts)

    hparams["model"].to("cuda")
    hparams["normalize"].to("cuda")
    hparams["compute_features"].to("cuda")

    print(f"starting experiment name: {hparams['experiment_name']}")

    def get_output_lengths(input_lengths):
        """ Function to get the output length of the feature extractor this is
            necessery to compute the masks of BestRQ.
        """
        sr = hparams["sample_rate"]
        hop_length = hparams["hop_length"]

        return (input_lengths // (sr*hop_length / 1000) + 1).to(torch.long)

    # We define the custom collation function that is necessary for best-rq to
    # generate masks.
    brq_mask_collate_fn_partial = partial(
        brq_mask_collate_fn,
        get_out_len_fn=get_output_lengths,
        mask_prob=hparams["mask_prob"],
        mask_length=hparams["mask_length"],
        n_mels=hparams["n_mels"],
    )

    # warmup
    for _ in range(WARMUP_STEPS):
        for sim_test_time in range(10, WARMUP_MAX_DURATION, 10):
            x = torch.rand(sim_test_time * 16000)
            mask = brq_mask_collate_fn_partial(
                [
                    {
                        "sig":x,
                        "id": "0"
                    }
                ]
            )
            
            with torch.cuda.amp.autocast():
                compute_benchmark(mask, hparams)

    save_results = []
    save_runs = []
    for sim_test_time in range(10, EVAL_MAX_DURATION, 10):
        avg_duration = 0.0
        std_duration = 0.0
        min_duration = 1000000.0
        max_duration = 0.0
        min_mem = float("inf")
        max_mem = float("-inf")
        avg_mem = 0.0
        std_mem = 0.0
        for n_iter in range(EVAL_AVG_STEPS):
            x = torch.rand(sim_test_time * 16000)

            mask = brq_mask_collate_fn_partial(
                [
                    {
                        "sig":x,
                        "id": f"{i}"
                    } for i in range(BATCH_SIZE)
                ]
            )

            with torch.cuda.amp.autocast():
                duration, mem = compute_benchmark(mask, hparams)
            torch.cuda.reset_peak_memory_stats()

            avg_duration += duration / EVAL_AVG_STEPS
            std_duration += duration ** 2 / EVAL_AVG_STEPS
            avg_mem += mem / EVAL_AVG_STEPS
            std_mem += mem ** 2 / EVAL_AVG_STEPS

            min_duration = min(min_duration, duration)
            max_duration = max(max_duration, duration)

            min_mem = min(min_mem, mem)
            max_mem = max(max_mem, mem)

            save_runs.append([sim_test_time, duration, mem])

        # convert in GiB
        avg_mem_gib = avg_mem / 1024 / 1024 / 1024
        min_mem_gib = min_mem / 1024 / 1024 / 1024
        max_mem_gib = max_mem / 1024 / 1024 / 1024

        print(f"Duration for {sim_test_time} seconds: {avg_duration}; Memory: {avg_mem}; Memory in GiB: {avg_mem_gib}; Min Memory: {min_mem_gib}; Max Memory: {max_mem_gib}")

        save_results.append((sim_test_time, avg_duration, avg_mem, avg_mem_gib, std_duration, std_mem, min_duration, max_duration, min_mem_gib, max_mem_gib))

    # save in csv file at location hparams["output_folder"]
    import os
    os.makedirs(os.path.join("memory_results", hparams["experiment_name"]), exist_ok=True)
    save_file = os.path.join("memory_results", hparams["experiment_name"], "memory_test.csv")
    with open(save_file, "w") as f:
        f.write("Time,Duration,Memory,Memory (GiB), Std Duration, Std Memory, Min Duration, Max Duration, Min Memory (GiB), Max Memory (GiB)\n")
        for time, duration, mem, mem_gib, std_duration, std_mem, min_duration, max_duration, min_mem, max_mem in save_results:
            f.write(f"{time},{duration},{mem},{mem_gib},{std_duration},{std_mem}, {min_duration}, {max_duration}, {min_mem}, {max_mem}\n")

    save_file = os.path.join("memory_results", hparams["experiment_name"], "memory_runs.csv")
    with open(save_file, "w") as f:
        f.write("Time,Duration,Memory\n")
        for time, duration, mem in save_runs:
            f.write(f"{time},{duration},{mem}\n")
        

if __name__ == "__main__":
    main()
