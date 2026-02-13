from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, List, Optional

import torch
import triton
import triton.language as tl
from huggingface_hub import snapshot_download

from sglang.srt.constrained.base_grammar_backend import BaseGrammarObject
from sglang.srt.distributed.parallel_state import (
    GroupCoordinator,
    patch_tensor_parallel_group,
)
from sglang.srt.environ import envs
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.common import get_last_loc
from sglang.srt.server_args import ServerArgs, get_global_server_args
from sglang.srt.utils import is_cuda, is_hip, is_npu, next_power_of_2

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_npu = is_npu()

if TYPE_CHECKING:
    from sglang.srt.speculative.eagle_info import EagleVerifyInput


if _is_cuda:
    from sgl_kernel import fast_topk
elif _is_hip:
    from sgl_kernel import fast_topk
else:
    from sglang.srt.utils.common import fast_topk


logger = logging.getLogger(__name__)


# Simulate acceptance length for benchmarking purposes
SIMULATE_ACC_LEN = envs.SGLANG_SIMULATE_ACC_LEN.get()  # turn off if < 0
SIMULATE_ACC_METHOD = envs.SGLANG_SIMULATE_ACC_METHOD.get()

TREE_TRAVERSE_TIME_THRESHOLD = 1  # TODO: set this properly
TREE_SPEC_KERNEL_AVAILABLE = _is_cuda  # This kernel is only available for CUDA now


def spec_need_hidden_states(server_args: Optional[ServerArgs] = None) -> bool:
    if server_args is None:
        server_args = get_global_server_args()

    # TODO(lsyin): also skip when 1) step = 1 or 2) standalone draft model
    return not server_args.enable_multi_layer_eagle


def create_extend_after_decode_spec_info(
    verified_id,
    seq_lens,
    accept_lens,
    positions,
    new_verified_id,
):
    """
    PyTorch implementation of the Triton kernel create_extend_after_decode_spec_info
    """
    bs_upper = seq_lens.shape[0]
    batch_size = seq_lens.shape[0]

    for pid in range(batch_size):
        seq_length = seq_lens[pid].item()
        accept_length = accept_lens[pid].item()

        accept_len_cumsum = torch.sum(accept_lens[:pid]).item()
        positions_ptr = positions[accept_len_cumsum:]
        for offset in range(min(bs_upper, accept_length)):
            positions_ptr[offset] = seq_length - accept_length + offset

        accept_len_cumsum += accept_length - 1
        verified_id_data = verified_id[accept_len_cumsum]
        new_verified_id[pid] = verified_id_data


def assign_req_to_token_pool(
    req_pool_indices,
    req_to_token,
    start_offset,
    end_offset,
    out_cache_loc,
    pool_len,
    bs_upper,
):
    """
    PyTorch implementation of the Triton kernel assign_req_to_token_pool
    """
    batch_size = req_pool_indices.shape[0]
    BLOCK_SIZE = 32

    for pid in range(batch_size):
        kv_start = start_offset[pid].item()
        kv_end = end_offset[pid].item()
        token_pool_idx = req_pool_indices[pid].item()
        token_pool = req_to_token[token_pool_idx]

        # Calculate out_offset
        out_offset = 0
        for length_pid in range(pid):
            start_val = start_offset[length_pid].item()
            end_val = end_offset[length_pid].item()
            out_offset += (end_val - start_val)

        out_cache_ptr = out_cache_loc[out_offset:]

        save_offset = kv_start
        load_offset = 0

        num_loop = (kv_end - kv_start + BLOCK_SIZE - 1) // BLOCK_SIZE
        for loop_idx in range(num_loop):
            block_size = min(BLOCK_SIZE, kv_end - save_offset)
            mask = torch.arange(block_size) < block_size
            if mask.any():
                data = out_cache_ptr[load_offset:load_offset + block_size]
                token_pool[save_offset:save_offset + block_size] = data
            save_offset += BLOCK_SIZE
            load_offset += BLOCK_SIZE


def assign_req_to_token_pool_func(
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    start_offset: torch.Tensor,
    end_offset: torch.Tensor,
    out_cache_loc: torch.Tensor,
    batch_size: int,
):
    """
    Wrapper function to call assign_req_to_token_pool with appropriate parameters
    """
    pool_len = req_to_token.shape[1]
    bs_upper = next_power_of_2(batch_size)
   
    assign_req_to_token_pool(
        req_pool_indices,
        req_to_token,
        start_offset,
        end_offset,
        out_cache_loc,
        pool_len,
        bs_upper,
    )


def assign_draft_cache_locs(
    req_pool_indices,
    req_to_token,
    seq_lens,
    extend_lens,
    num_new_pages_per_topk,
    out_cache_loc,
    source_cache_loc,
    target_cache_loc,
    last_page_lens_cumsum,
    duplicate_cache_len: int,
    pool_len: int,
    topk: int,
    speculative_num_steps: int,
    page_size: int,
    bs_upper: int,
    iter_upper: int,
):
    """
    PyTorch implementation of the Triton kernel assign_draft_cache_locs
    """
    BLOCK_SIZE = 128
    batch_size = req_pool_indices.shape[0]

    for pid in range(batch_size):
        if page_size == 1 or topk == 1:
            copy_len = topk * speculative_num_steps
            out_cache_ptr = out_cache_loc[pid * topk * speculative_num_steps:]
        else:
            copy_len = extend_lens[pid].item()
            cum_copy_len = torch.sum(extend_lens[:pid]).item()
            out_cache_ptr = out_cache_loc[cum_copy_len:]

        # Part 1: Copy from out_cache_loc to req_to_token
        kv_start = seq_lens[pid].item()
        token_pool_idx = req_pool_indices[pid].item()
        token_pool = req_to_token[token_pool_idx]

        num_loop = (copy_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        for i in range(num_loop):
            copy_start = i * BLOCK_SIZE
            copy_end = min(copy_start + BLOCK_SIZE, copy_len)
            block_size = copy_end - copy_start
            if block_size > 0:
                data = out_cache_ptr[copy_start:copy_end]
                token_pool[kv_start + copy_start:kv_start + copy_end] = data

        if page_size != 1 and topk != 1 and duplicate_cache_len > 0:
            # Part 2: Copy indices into source_cache_loc and target_cache_loc
            prefix_len = seq_lens[pid].item()
            last_page_len = prefix_len % page_size
            if last_page_len > 0:
                num_new_pages_per_topk_ = num_new_pages_per_topk[pid].item()
                prefix_base = prefix_len - last_page_len
                src_indices = token_pool[prefix_base:prefix_base + last_page_len]
               
                last_page_lens_cumsum_ = last_page_lens_cumsum[pid].item()
               
                for topk_id in range(1, topk):
                    src_start = (topk - 1) * (last_page_lens_cumsum_ - last_page_len) + (topk_id - 1) * last_page_len
                    source_cache_loc[src_start:src_start + last_page_len] = src_indices
                   
                    tgt_indices = token_pool[prefix_base + topk_id * num_new_pages_per_topk_ * page_size:prefix_base + topk_id * num_new_pages_per_topk_ * page_size + last_page_len]
                    target_cache_loc[src_start:src_start + last_page_len] = tgt_indices

            # Part 3: Copy and remove the used indices for duplication
            iter_start = pid * speculative_num_steps * topk
            for topk_id in range(topk):
                prefix_base = seq_lens[pid].item() // page_size * page_size
                prefix_len = seq_lens[pid].item()
                last_page_len = prefix_len % page_size
                num_new_pages_per_topk_ = num_new_pages_per_topk[pid].item()
                start = prefix_base + topk_id * num_new_pages_per_topk_ * page_size + last_page_len
               
                for iter_offset in range(speculative_num_steps):
                    if last_page_len <= iter_offset < (speculative_num_steps + last_page_len):
                        idx = start + iter_offset - last_page_len
                        out_idx = iter_start + topk_id * speculative_num_steps - last_page_len + iter_offset
                        if 0 <= out_idx < out_cache_loc.size(0) and 0 <= idx < token_pool.size(0):
                            out_cache_loc[out_idx] = token_pool[idx]


def generate_draft_decode_kv_indices(
    req_pool_indices,
    req_to_token,
    paged_kernel_lens,
    kv_indices,
    kv_indptr,
    positions,
    pool_len: int,
    kv_indices_stride: int,
    kv_indptr_stride: int,
    bs_upper: int,
    iter_upper: int,
    num_tokens_upper: int,
    page_size: int,
):
    """
    PyTorch implementation of the Triton kernel generate_draft_decode_kv_indices
    """
    num_steps = kv_indices.shape[0] // kv_indices_stride
    num_seqs = req_pool_indices.shape[0]
    topk = kv_indptr.shape[0] // kv_indptr_stride
   
    for iters in range(num_steps):
        for bid in range(num_seqs):
            for topk_id in range(topk):
                kv_indices_ptr = kv_indices[iters * kv_indices_stride:]
                kv_indptr_ptr = kv_indptr[iters * kv_indptr_stride:]
               
                seq_len = paged_kernel_lens[bid].item()
                cum_seq_len = torch.sum(paged_kernel_lens[:bid]).item()

                # Update kv_indices
                kv_offset = cum_seq_len * topk + bid * (iters + 1) * topk + topk_id * (seq_len + iters + 1)
                kv_ptr = kv_indices_ptr[kv_offset:]
                token_pool_idx = req_pool_indices[bid].item()
                token_pool_ptr = req_to_token[token_pool_idx]

                # Copy prefix tokens
                for block_start in range(0, seq_len, 128):
                    block_end = min(block_start + 128, seq_len)
                    block_size = block_end - block_start
                    if block_size > 0:
                        data = token_pool_ptr[block_start:block_end]
                        kv_ptr[block_start:block_end] = data

                # Copy extend tokens
                if page_size == 1 or topk == 1:
                    extend_start = seq_len + topk_id * num_steps
                    extend_end = extend_start + min(iter_upper, iters + 1)
                    extend_data = token_pool_ptr[extend_start:extend_end]
                else:
                    prefix_len = seq_len
                    last_page_len = prefix_len % page_size
                    num_new_pages_per_topk = (last_page_len + num_steps + page_size - 1) // page_size
                    prefix_base = seq_len // page_size * page_size
                    start = prefix_base + topk_id * num_new_pages_per_topk * page_size + last_page_len
                    extend_end = min(iter_upper, iters + 1)
                    extend_data = token_pool_ptr[start:start + extend_end]
               
                kv_ptr[seq_len:seq_len + extend_end] = extend_data

                # Update kv_indptr
                zid = bid * topk + topk_id
                if zid == 0:
                    zid = num_seqs * topk
                if zid < len(positions):
                    base = torch.sum(positions[:zid]).item()
                    kv_indptr_ptr[zid] = base + zid * (iters + 1)


def align_evict_mask_to_page_size(
    seq_lens,
    evict_mask,
    page_size: int,
    num_draft_tokens: int,
):
    """
    PyTorch implementation of the Triton kernel align_evict_mask_to_page_size
    """
    batch_size = seq_lens.shape[0]
    BLOCK_SIZE = 128
   
    for bid in range(batch_size):
        seq_len = seq_lens[bid].item()
        mask_row = evict_mask[bid, :num_draft_tokens]
       
        num_trues = torch.sum(mask_row).item()
        num_false = num_draft_tokens - num_trues
       
        start = ((seq_len + num_false - 1) // page_size * page_size - seq_len)
        start = max(start, 0)
        end = min(start + page_size, num_draft_tokens)
       
        if start < end:
            evict_mask[bid, start:end] = False


def get_target_cache_loc(
    tgt_cache_loc,
    to_free_slots,
    accept_length,
    to_free_num_slots,
    out_cache_loc,
    num_verify_tokens: int,
    num_verify_tokens_upper: int,
    bs_upper: int,
):
    """
    PyTorch implementation of the Triton kernel get_target_cache_loc
    """
    batch_size = accept_length.shape[0]
   
    for bid in range(batch_size):
        # Write the first part to tgt_cache_loc
        accept_len_all = torch.sum(accept_length[:bid]).item()
        tgt_cache_loc_start = accept_len_all + bid
        copy_len = accept_length[bid].item() + 1
        out_cache_loc_row = out_cache_loc[bid, :copy_len]
        tgt_cache_loc[tgt_cache_loc_start:tgt_cache_loc_start + copy_len] = out_cache_loc_row

        # Write the second part to to_free_num_pages
        to_free_num_slots_all = torch.sum(to_free_num_slots[:bid]).item()
        to_free_num_slots_cur = to_free_num_slots[bid].item()
        out_cache_loc_start = num_verify_tokens - to_free_num_slots_cur
        to_free_slots_start = torch.sum(to_free_num_slots[:bid]).item()

        copy_len = to_free_num_slots_cur
        if copy_len > 0:
            out_cache_loc_row = out_cache_loc[bid, out_cache_loc_start:out_cache_loc_start + copy_len]
            to_free_slots[to_free_slots_start:to_free_slots_start + copy_len] = out_cache_loc_row


def filter_finished_cache_loc_kernel(
    out_cache_loc,
    tgt_cache_loc,
    accept_length,
    accept_length_filter,
    bs_upper: int,
    num_verify_tokens_upper: int,
):
    """
    PyTorch implementation of the Triton kernel filter_finished_cache_loc_kernel
    """
    batch_size = accept_length.shape[0]
   
    for bid in range(batch_size):
        accept_length_all = torch.sum(accept_length[:bid]).item()
        old_start = accept_length_all + bid

        accept_length_filter_all = torch.sum(accept_length_filter[:bid]).item()
        new_start = accept_length_filter_all

        copy_len = accept_length_filter[bid].item()
        value = tgt_cache_loc[old_start:old_start + copy_len]
        out_cache_loc[new_start:new_start + copy_len] = value


@torch.compile(dynamic=True, disable=_is_npu)
def get_src_tgt_cache_loc(
    seq_lens: torch.Tensor,
    out_cache_loc: torch.Tensor,
    accept_index: torch.Tensor,
    accept_length: torch.Tensor,
    draft_token_num: int,
    page_size: int,
):
    src_cache_loc = out_cache_loc[accept_index]
    tgt_cache_loc = torch.empty_like(src_cache_loc)
    extended_len = seq_lens + draft_token_num
    keep_len = torch.minimum(
        (seq_lens + accept_length + 1 + page_size - 1) // page_size * page_size,
        extended_len,
    )
    to_free_num_slots = extended_len - keep_len
    return src_cache_loc, tgt_cache_loc, to_free_num_slots


@torch.compile(dynamic=True, disable=_is_npu)
def create_accept_length_filter(
    accept_length: torch.Tensor,
    unfinished_index_device: torch.Tensor,
    seq_lens: torch.Tensor,
):
    accept_length_filter = torch.zeros_like(accept_length)
    accept_length_filter[unfinished_index_device] = (
        accept_length[unfinished_index_device] + 1
    )
    seq_lens.add_(accept_length + 1)
    return accept_length_filter


@torch.compile(dynamic=True, disable=_is_npu)
def select_top_k_tokens(
    i: int,
    topk_p: torch.Tensor,
    topk_index: torch.Tensor,
    hidden_states: torch.Tensor,
    scores: torch.Tensor,
    topk: int,
):
    if i == 0:
        # The first step after extend
        input_ids = topk_index.flatten()
        if hidden_states is not None:
            hidden_states = hidden_states.repeat_interleave(topk, dim=0)
        scores = topk_p  # shape: (b, topk)

        tree_info = (
            topk_p.unsqueeze(1),  # shape: (b, 1, topk)
            topk_index,  # shape: (b, topk)
            torch.arange(-1, topk, dtype=torch.long, device=input_ids.device)
            .unsqueeze(0)
            .repeat(topk_p.shape[0], 1),  # shape: (b, topk + 1)
        )
    else:
        # The later decode steps
        expand_scores = torch.mul(
            scores.unsqueeze(2), topk_p.reshape(-1, topk, topk)
        )  # (b, topk, 1) x (b, topk ,topk) -> (b, topk, topk)
        topk_cs_p, topk_cs_index = fast_topk(
            expand_scores.flatten(start_dim=1), topk, dim=-1
        )  # (b, topk)
        scores = topk_cs_p  # shape: (b, topk)

        topk_index = topk_index.reshape(-1, topk**2)
        input_ids = torch.gather(topk_index, index=topk_cs_index, dim=1).flatten()

        if hidden_states.shape[0] > 0:
            selected_input_index = topk_cs_index.flatten() // topk + torch.arange(
                0, hidden_states.shape[0], step=topk, device=topk_index.device
            ).repeat_interleave(topk)
            hidden_states = hidden_states[selected_input_index, :]

        tree_info = (
            expand_scores,  # shape: (b, topk, topk)
            topk_index,  # shape: (b, topk * topk)
            topk_cs_index + (topk**2 * (i - 1) + topk),  # shape: (b, topk)
        )

    return input_ids, hidden_states, scores, tree_info


def generate_simulated_accept_index(
    accept_index,
    predict,
    accept_length,
    bs,
    spec_steps,
    simulate_acc_len: float = SIMULATE_ACC_LEN,
    simulate_acc_method: str = SIMULATE_ACC_METHOD,
):
    assert simulate_acc_len > 0.0

    if simulate_acc_method == "multinomial":
        simulated_values = torch.normal(
            mean=simulate_acc_len,
            std=1.0,
            size=(1,),
            device="cpu",
        )
        # clamp simulated values to be between 1 and self.spec_steps
        simulated_values = torch.clamp(simulated_values, min=1.0, max=spec_steps + 1)
        simulate_acc_len = int(simulated_values.round().item())
    elif simulate_acc_method == "match-expected":
        # multinomial sampling does not match the expected length
        # we keep it for the sake of compatibility of existing tests
        # but it's better to use "match-expected" for the cases that need to
        # match the expected length, One caveat is that this will only sample
        # either round down or round up of the expected length
        simulate_acc_len = max(1.0, min(spec_steps + 1, simulate_acc_len))
        lower = int(simulate_acc_len // 1)
        upper = lower + 1 if lower < spec_steps + 1 else lower
        if lower == upper:
            simulate_acc_len = lower
        else:
            weight_upper = simulate_acc_len - lower
            weight_lower = 1.0 - weight_upper
            probs = torch.tensor([weight_lower, weight_upper], device="cpu")
            sampled_index = torch.multinomial(probs, num_samples=1)
            simulate_acc_len = lower if sampled_index == 0 else upper
    else:
        raise ValueError(f"Invalid simulate_acc_method: {SIMULATE_ACC_METHOD}")

    accept_indx_first_col = accept_index[:, 0].view(-1, 1)
    sim_accept_index = torch.full(
        (bs, spec_steps + 1), -1, dtype=torch.int32, device="cpu"
    )
    sim_accept_index[:, :simulate_acc_len] = accept_indx_first_col + torch.arange(
        simulate_acc_len, device=accept_index.device
    )
    accept_length.fill_(simulate_acc_len - 1)
    predict.fill_(100)  # some legit token id
    return sim_accept_index


def traverse_tree(
    retrieve_next_token: torch.Tensor,
    retrieve_next_sibling: torch.Tensor,
    draft_tokens: torch.Tensor,
    grammar: BaseGrammarObject,
    allocate_token_bitmask: torch.Tensor,
    vocab_size: Optional[int] = None,
):
    """
    Traverse the tree constructed by the draft model to generate the logits mask.
    """
    assert (
        retrieve_next_token.shape == retrieve_next_sibling.shape == draft_tokens.shape
    )

    def dfs(
        curr: int,
        retrieve_next_token: torch.Tensor,
        retrieve_next_sibling: torch.Tensor,
        parent_pos: int,
    ):
        if curr == 0:
            # the first token generated by the target model, and thus it is always
            # accepted from the previous iteration
            accepted = True
        else:
            parent_bitmask = allocate_token_bitmask[parent_pos]
            curr_token_id = draft_tokens[curr]
            if vocab_size and curr_token_id >= vocab_size:
                accepted = False
            else:
                # 32 boolean bitmask values are packed into 32-bit integers
                accepted = (
                    parent_bitmask[curr_token_id // 32] & (1 << (curr_token_id % 32))
                ) != 0

        if accepted:
            if curr != 0:
                # Accept the current token
                grammar.accept_token(draft_tokens[curr])
            if not grammar.is_terminated():
                # Generate the bitmask for the current token
                grammar.fill_vocab_mask(allocate_token_bitmask, curr)
                if retrieve_next_token[curr] != -1:
                    # Visit the child node
                    dfs(
                        retrieve_next_token[curr],
                        retrieve_next_token,
                        retrieve_next_sibling,
                        curr,
                    )

            if curr != 0:
                # Rollback the current token
                grammar.rollback(1)

        if retrieve_next_sibling[curr] != -1:
            # Visit the sibling node
            dfs(
                retrieve_next_sibling[curr],
                retrieve_next_token,
                retrieve_next_sibling,
                parent_pos,
            )

    dfs(0, retrieve_next_token, retrieve_next_sibling, -1)


def generate_token_bitmask(
    reqs: List[Req],
    verify_input: EagleVerifyInput,
    retrieve_next_token_cpu: torch.Tensor,
    retrieve_next_sibling_cpu: torch.Tensor,
    draft_tokens_cpu: torch.Tensor,
    vocab_size: int,
):
    """
    Generate the logit mask for structured output.
    Draft model's token can be either valid or invalid with respect to the grammar.
    We need to perform DFS to
    1. figure out which tokens are accepted by the grammar.
    2. if so, what is the corresponding logit mask.
    """

    num_draft_tokens = draft_tokens_cpu.shape[-1]

    allocate_token_bitmask = None
    assert len(reqs) == retrieve_next_token_cpu.shape[0]
    grammar = None
    for i, req in enumerate(reqs):
        if req.grammar is not None:
            if allocate_token_bitmask is None:
                allocate_token_bitmask = req.grammar.allocate_vocab_mask(
                    vocab_size=vocab_size,
                    batch_size=draft_tokens_cpu.numel(),
                    device="cpu",
                )
            grammar = req.grammar
            s = time.perf_counter()
            traverse_tree(
                retrieve_next_token_cpu[i],
                retrieve_next_sibling_cpu[i],
                draft_tokens_cpu[i],
                req.grammar,
                allocate_token_bitmask[
                    i * num_draft_tokens : (i + 1) * num_draft_tokens
                ],
                vocab_size=vocab_size,
            )
            tree_traverse_time = time.perf_counter() - s
            if tree_traverse_time > TREE_TRAVERSE_TIME_THRESHOLD:
                logger.warning(
                    f"Bit mask generation took {tree_traverse_time} seconds with "
                    f"grammar: {req.grammar}"
                )

    verify_input.grammar = grammar
    return allocate_token_bitmask


def load_token_map(token_map_path: str) -> List[int]:
    if not os.path.exists(token_map_path):
        cache_dir = snapshot_download(
            os.path.dirname(token_map_path),
            ignore_patterns=["*.bin", "*.safetensors"],
        )
        token_map_path = os.path.join(cache_dir, os.path.basename(token_map_path))
    hot_token_id = torch.load(token_map_path, weights_only=True)
    return torch.tensor(hot_token_id, dtype=torch.int64)


@contextmanager
def draft_tp_context(tp_group: GroupCoordinator):
    # Draft model doesn't use dp and has its own tp group.
    # We disable mscclpp now because it doesn't support 2 comm groups.
    with patch_tensor_parallel_group(tp_group):
        yield


def detect_nan(logits_output: LogitsProcessorOutput):
    logits = logits_output.next_token_logits
    if torch.any(torch.isnan(logits)):
        logger.error("Detected errors during sampling! NaN in the logits.")
        raise ValueError("Detected errors during sampling! NaN in the logits.")


# Disable torch.compile for this function because it will be
# even slower.
# @torch.compile(dynamic=True)
def get_last_loc_large_page_size_large_top_k(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    speculative_num_steps: int,
    topk: int,
    page_size: int,
):
    prefix_lens = seq_lens
    last_page_lens = prefix_lens % page_size
    num_new_pages_per_topk = (
        last_page_lens + speculative_num_steps + page_size - 1
    ) // page_size
    seq_lens = prefix_lens // page_size * page_size + num_new_pages_per_topk * (
        page_size * topk
    )
    extend_lens = seq_lens - prefix_lens
    last_loc = get_last_loc(
        req_to_token,
        req_pool_indices,
        prefix_lens,
    )

    return (
        prefix_lens,
        seq_lens,
        last_loc,
        num_new_pages_per_topk,
        extend_lens,
        last_page_lens,
    )
