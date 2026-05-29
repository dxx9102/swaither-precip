# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
from datasets.base import DownscalingDataset
from datasets.dataset import init_dataset_from_config
from physicsnemo.utils.diffusion import convert_datetime_to_cftime


import numpy as np 

##-------- MY CHANGES --------------
def _to_cftime(t):
    """Accept str / datetime / cftime and return cftime."""
    if isinstance(t, str):
        dt = datetime.datetime.strptime(t, "%Y-%m-%dT%H:%M:%S")
        return convert_datetime_to_cftime(dt)
    if isinstance(t, datetime.datetime):
        return convert_datetime_to_cftime(t)
    # assume already cftime-like
    return t


def get_dataset_and_sampler(dataset_cfg, times, has_lead_time=False, *, hori_idx=None):
    """
    Returns (dataset, sampler_indices).

    - If hori_idx is None: returns *all* (time, lead) samples that match `times`.
    - If hori_idx is an int: returns only the samples matching that time AND that lead index.
    """
    dataset, _ = init_dataset_from_config(dataset_cfg, batch_size=1)

    plot_times = [_to_cftime(t) for t in times]
    all_times = dataset.time()

    # indices matching the requested times (includes all leads if time is repeated)
    time_indices = [i for i, t in enumerate(all_times) if t in plot_times]

    if len(time_indices) == 0:
        raise ValueError(
            f"None of the requested times were found in dataset.time().\n"
            f"Requested (cftime): {plot_times[:5]}...\n"
            f"Dataset first times: {all_times[:5]}...\n"
            f"Dataset last times : {all_times[-5:]}...\n"
        )

    # --- NEW: horizon filtering ---
    if hori_idx is not None:
        # Fast path if dataset exposes samples_idx with (ti_low, ti_high, li)
        if hasattr(dataset, "samples_idx") and len(dataset.samples_idx) > 0 and len(dataset.samples_idx[0]) >= 3:
            time_indices = [i for i in time_indices if dataset.samples_idx[i][2] == hori_idx]
        else:
            # Fallback: peek lead_label from __getitem__ (safe, but slower if many indices)
            filtered = []
            for i in time_indices:
                item = dataset[i]
                # expecting (y, x, lead_label, time_id)
                if len(item) >= 3:
                    lead_label = item[2]
                    li = int(lead_label.item()) if hasattr(lead_label, "item") else int(lead_label)
                    if li == hori_idx:
                        filtered.append(i)
            time_indices = filtered

        if len(time_indices) == 0:
            raise ValueError(
                f"Requested times exist, but none matched hori_idx={hori_idx}.\n"
                f"Tip: check that lead_label is 0..L-1 and ordering matches your horizons."
            )

    return dataset, time_indices
    
# def get_dataset_and_sampler(dataset_cfg, times, has_lead_time=False):
#     """
#     Get a dataset and sampler for generation.
#     """
#     (dataset, _) = init_dataset_from_config(dataset_cfg, batch_size=1)
#     if has_lead_time:
#         plot_times = times
#     else:
#         plot_times = [
#             convert_datetime_to_cftime(
#                 datetime.datetime.strptime(time, "%Y-%m-%dT%H:%M:%S")
#             )
#             for time in times
#         ]
#     all_times = dataset.time()
#     time_indices = [all_times.index(t) for t in plot_times]
#     sampler = time_indices

#     return dataset, sampler


def save_images(
    writer,
    dataset: DownscalingDataset,
    times,
    image_out,
    image_tar,
    image_lr,
    time_index,
    t_index,
    has_lead_time,
):
    """
    Saves inferencing result along with the baseline

    Parameters
    ----------

    writer (NetCDFWriter): Where the data is being written
    in_channels (List): List of the input channels being used
    input_channel_info (Dict): Description of the input channels
    out_channels (List): List of the output channels being used
    output_channel_info (Dict): Description of the output channels
    input_norm (Tuple): Normalization data for input
    target_norm (Tuple): Normalization data for the target
    image_out (torch.Tensor): Generated output data
    image_tar (torch.Tensor): Ground truth data
    image_lr (torch.Tensor): Low resolution input data
    time_index (int): Epoch number
    t_index (int): index where times are located
    """
    # weather sub-plot
    image_lr2 = image_lr[0].unsqueeze(0)
    image_lr2 = image_lr2.cpu().numpy()
    image_lr2 = dataset.denormalize_input(image_lr2)

    image_tar2 = image_tar[0].unsqueeze(0)
    image_tar2 = image_tar2.cpu().numpy()
    image_tar2 = dataset.denormalize_output(image_tar2)

    # some runtime assertions
    if image_tar2.ndim != 4:
        raise ValueError("image_tar2 must be 4-dimensional")

    for idx in range(image_out.shape[0]):
        image_out2 = image_out[idx].unsqueeze(0)
        if image_out2.ndim != 4:
            raise ValueError("image_out2 must be 4-dimensional")

        # Denormalize the input and outputs
        image_out2 = image_out2.cpu().numpy()
        image_out2 = dataset.denormalize_output(image_out2)

        time = times[t_index]
        if hasattr(time, "strftime"):
            time_str = time.strftime("%Y-%m-%dT%H:%M:%S")
        else:
            time_str = str(time)

        writer.write_time(time_index, np.string_(time_str))  # important

        for channel_idx in range(image_out2.shape[1]):
            info = dataset.output_channels()[channel_idx]
            channel_name = info.name + info.level
            truth = image_tar2[0, channel_idx]

            writer.write_truth(channel_name, time_index, truth)
            writer.write_prediction(
                channel_name, time_index, idx, image_out2[0, channel_idx]
            )

        input_channel_info = dataset.input_channels()
        for channel_idx in range(len(input_channel_info)):
            info = input_channel_info[channel_idx]
            channel_name = info.name + info.level
            writer.write_input(channel_name, time_index, image_lr2[0, channel_idx])
            if channel_idx == image_lr2.shape[1] - 1:
                break
