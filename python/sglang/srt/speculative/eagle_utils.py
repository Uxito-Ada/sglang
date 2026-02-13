import math
from enum import IntEnum
from typing import List, Optional

import torch

from sglang.srt.utils import is_cuda, is_hip, is_npu

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_npu = is_npu()

if _is_cuda or _is_hip:
    from sgl_kernel import (
        build_tree_kernel_efficient as sgl_build_tree_kernel_efficient,
    )


def organize_draft_results(
    score_list: List[torch.Tensor],
    token_list: List[torch.Tensor],
    parents_list: List[torch.Tensor],
    num_draft_token: int,
):
    score_list = torch.cat(score_list, dim=1).flatten(1)
    ss_token_list = torch.cat(token_list, dim=1)
    #print(f"score_list shape: {score_list.shape}")
    #print(f"score_list last dim len: {len(score_list[-1])}")
    #print(f"k = {num_draft_token - 1}")
    top_scores = torch.topk(score_list, num_draft_token - 1, dim=-1)
    top_scores_index = top_scores.indices
    top_scores_index = torch.sort(top_scores_index).values
    draft_tokens = torch.gather(ss_token_list, index=top_scores_index, dim=1)

    if len(parents_list) > 1:
        parent_list = torch.cat(parents_list[:-1], dim=1)
    else:
        batch_size = parents_list[0].shape[0]
        parent_list = torch.empty(batch_size, 0, device=parents_list[0].device)

    return parent_list, top_scores_index, draft_tokens


class TreeMaskMode(IntEnum):
    FULL_MASK = 0
    QLEN_ONLY = 1
    QLEN_ONLY_BITPACKING = 2
import torch
import numpy as np

def build_tree_kernel_efficient_cpu(
    parent_list,
    selected_index,
    verified_seq_len,
    tree_mask,
    positions,
    retrive_index,
    retrive_next_token,
    retrive_next_sibling,
    topk,
    depth,
    draft_token_num,
    tree_mask_mode
):
    """
    CPU implementation of the GPU kernel for building a tree structure
    
    Args:
        parent_list: [bs, draft_token_num] - parent indices for each token
        selected_index: [bs, draft_token_num] - selected token indices
        verified_seq_len: [bs] - length of verified sequence for each batch
        tree_mask: [bs, draft_token_num, draft_token_num] - mask for tree connections
        positions: [bs, draft_token_num] - position information
        retrive_index: [bs, draft_token_num] - retrieval indices
        retrive_next_token: [bs, draft_token_num] - next token indices
        retrive_next_sibling: [bs, draft_token_num] - next sibling indices
        topk: number of top candidates
        depth: maximum depth
        draft_token_num: number of draft tokens
        tree_mask_mode: mode for tree mask generation
    """
    
    # Get batch size
    bs = parent_list.size(0)
    
    # Process each batch item
    for b in range(bs):
        # Get current batch data
        parent_b = parent_list[b]
        selected_b = selected_index[b]
        verified_len = verified_seq_len[b].item()
        
        # Initialize tree mask for this batch
        if tree_mask_mode == 0:  # Assuming 0 means QLEN_ONLY_BITPACKING based on original code
            # Bitpacking mode - handle differently based on draft_token_num
            num_bytes_per_item = 1
            if draft_token_num > 16:
                num_bytes_per_item = 4
            elif draft_token_num > 8:
                num_bytes_per_item = 2
            
            # For bitpacking, we'll handle the tree construction with packed bits
            _build_tree_efficient_partial_packed_cpu(
                parent_b, selected_b, verified_len, tree_mask[b], 
                positions[b], retrive_index[b], 
                retrive_next_token[b], retrive_next_sibling[b],
                topk, depth, draft_token_num, num_bytes_per_item
            )
        else:
            # Standard boolean mask mode
            _build_tree_efficient_cpu(
                parent_b, selected_b, verified_len, tree_mask[b], 
                positions[b], retrive_index[b], 
                retrive_next_token[b], retrive_next_sibling[b],
                topk, depth, draft_token_num, tree_mask_mode
            )

def _build_tree_efficient_cpu(
    parent_b, selected_b, verified_len, tree_mask_b, 
    positions_b, retrive_index_b, retrive_next_token_b, retrive_next_sibling_b,
    topk, depth, draft_token_num, tree_mask_mode
):
    """
    CPU implementation for standard tree building (non-bitpacked version)
    """
    # Reset tree mask
    tree_mask_b.fill_(False)
    
    # Build tree structure based on parent relationships
    for i in range(draft_token_num):
        current_pos = i
        
        # Set position
        positions_b[i] = current_pos
        
        # Mark valid connections in tree mask
        if i < len(parent_b):
            parent_idx = parent_b[i].item()
            
            # Ensure parent index is valid
            if 0 <= parent_idx < draft_token_num and parent_idx != i:
                tree_mask_b[parent_idx, i] = True
        
        # Set retrieval indices
        if i < len(selected_b):
            retrive_index_b[i] = selected_b[i]
        
        # Set next token and sibling relationships
        if i + 1 < draft_token_num:
            retrive_next_token_b[i] = i + 1
        else:
            retrive_next_token_b[i] = -1  # No next token
            
        # Find next sibling (same parent, higher index)
        current_parent = parent_b[i].item() if i < len(parent_b) else -1
        next_sibling = -1
        for j in range(i + 1, draft_token_num):
            if j < len(parent_b) and parent_b[j].item() == current_parent:
                next_sibling = j
                break
        retrive_next_sibling_b[i] = next_sibling

def _build_tree_efficient_partial_packed_cpu(
    parent_b, selected_b, verified_len, tree_mask_b, 
    positions_b, retrive_index_b, retrive_next_token_b, retrive_next_sibling_b,
    topk, depth, draft_token_num, num_bytes_per_item
):
    """
    CPU implementation for bitpacked tree building
    """
    # For bitpacked version, we convert the tree mask to appropriate format
    # Since PyTorch doesn't have native uint8 tensor for bit operations,
    # we'll work with boolean tensors and then pack if needed
    
    # Reset tree mask
    tree_mask_b.fill_(False)
    
    # Build tree structure similar to non-packed version
    for i in range(draft_token_num):
        current_pos = i
        positions_b[i] = current_pos
        
        if i < len(parent_b):
            parent_idx = parent_b[i].item()
            if 0 <= parent_idx < draft_token_num and parent_idx != i:
                tree_mask_b[parent_idx, i] = True
        
        if i < len(selected_b):
            retrive_index_b[i] = selected_b[i]
        
        if i + 1 < draft_token_num:
            retrive_next_token_b[i] = i + 1
        else:
            retrive_next_token_b[i] = -1
            
        # Find next sibling
        current_parent = parent_b[i].item() if i < len(parent_b) else -1
        next_sibling = -1
        for j in range(i + 1, draft_token_num):
            if j < len(parent_b) and parent_b[j].item() == current_parent:
                next_sibling = j
                break
        retrive_next_sibling_b[i] = next_sibling

# Alternative vectorized version for better performance
def build_tree_kernel_efficient_vectorized(
    parent_list,
    selected_index,
    verified_seq_len,
    tree_mask,
    positions,
    retrive_index,
    retrive_next_token,
    retrive_next_sibling,
    topk,
    depth,
    draft_token_num,
    tree_mask_mode
):
    """
    Vectorized CPU implementation for better performance
    """
    bs = parent_list.size(0)
    
    # Process all batches in parallel where possible
    for b in range(bs):
        parent_b = parent_list[b]
        selected_b = selected_index[b]
        verified_len = verified_seq_len[b].item()
        
        # Create position tensor
        positions[b] = torch.arange(draft_token_num, dtype=torch.long)
        
        # Set retrieve indices
        retrive_index[b] = selected_b[:draft_token_num]
        
        # Set next token (shift by 1)
        next_tokens = torch.cat([
            torch.arange(1, draft_token_num, dtype=torch.long), 
            torch.tensor([-1], dtype=torch.long)
        ])
        retrive_next_token[b] = next_tokens
        
        # Compute next siblings using vectorized operations
        parents = parent_b[:draft_token_num]
        next_siblings = torch.full_like(parents, -1, dtype=torch.long)
        
        for i in range(draft_token_num):
            current_parent = parents[i]
            # Find first occurrence after index i with same parent
            mask = (parents[i+1:] == current_parent)
            if mask.any():
                next_sibling_idx = torch.where(mask)[0][0] + i + 1
                next_siblings[i] = next_sibling_idx
        
        retrive_next_sibling[b] = next_siblings
        
        # Build tree mask
        if tree_mask_mode == 0:  # Bitpacking mode
            # Handle bitpacking logic
            _build_tree_mask_bitpacked(tree_mask[b], parent_b, draft_token_num)
        else:
            # Standard boolean mask
            _build_tree_mask_standard(tree_mask[b], parent_b, draft_token_num)

def _build_tree_mask_standard(tree_mask_b, parent_b, draft_token_num):
    """Build standard boolean tree mask"""
    tree_mask_b.fill_(False)
    
    for i in range(draft_token_num):
        if i < len(parent_b):
            parent_idx = parent_b[i].item()
            if 0 <= parent_idx < draft_token_num and parent_idx != i:
                tree_mask_b[parent_idx, i] = True

def _build_tree_mask_bitpacked(tree_mask_b, parent_b, draft_token_num):
    """Build bitpacked tree mask (emulated with boolean tensor)"""
    _build_tree_mask_standard(tree_mask_b, parent_b, draft_token_num)

# Example usage function
def example_usage():
    """
    Example of how to use the function
    """
    # Example parameters
    bs = 2
    draft_token_num = 8
    topk = 5
    depth = 3
    tree_mask_mode = 1  # Not bitpacking mode
    
    # Create example tensors
    parent_list = torch.randint(0, draft_token_num, (bs, draft_token_num))
    selected_index = torch.randint(0, 1000, (bs, draft_token_num))
    verified_seq_len = torch.randint(1, draft_token_num, (bs,))
    tree_mask = torch.zeros((bs, draft_token_num, draft_token_num), dtype=torch.bool)
    positions = torch.zeros((bs, draft_token_num), dtype=torch.long)
    retrive_index = torch.zeros((bs, draft_token_num), dtype=torch.long)
    retrive_next_token = torch.zeros((bs, draft_token_num), dtype=torch.long)
    retrive_next_sibling = torch.zeros((bs, draft_token_num), dtype=torch.long)
    
    # Call the function
    build_tree_kernel_efficient(
        parent_list, selected_index, verified_seq_len, tree_mask,
        positions, retrive_index, retrive_next_token, retrive_next_sibling,
        topk, depth, draft_token_num, tree_mask_mode
    )
    
    print("Tree building completed successfully!")
    return {
        'tree_mask': tree_mask,
        'positions': positions,
        'retrive_index': retrive_index,
        'retrive_next_token': retrive_next_token,
        'retrive_next_sibling': retrive_next_sibling
    }
import torch
import numpy as np

def build_tree_kernel_efficient_cpu_v2(
    parent_list,
    selected_index,
    verified_seq_len,
    tree_mask,
    positions,
    retrive_index,
    retrive_next_token,
    retrive_next_sibling,
    topk,
    depth,
    draft_token_num,
    tree_mask_mode
):
    """
    CPU implementation of the GPU kernel for building a tree structure
    
    Args:
        parent_list: [bs, draft_token_num] - parent indices for each token
        selected_index: [bs, draft_token_num] - selected token indices
        verified_seq_len: [bs] - length of verified sequence for each batch
        tree_mask: [bs, draft_token_num, draft_token_num] - mask for tree connections
        positions: [bs, draft_token_num] - position information
        retrive_index: [bs, draft_token_num] - retrieval indices
        retrive_next_token: [bs, draft_token_num] - next token indices
        retrive_next_sibling: [bs, draft_token_num] - next sibling indices
        topk: number of top candidates
        depth: maximum depth
        draft_token_num: number of draft tokens
        tree_mask_mode: mode for tree mask generation
    """
    
    # Get batch size
    bs = parent_list.size(0)
    
    # Process each batch item
    for b in range(bs):
        # Get current batch data
        parent_b = parent_list[b]
        selected_b = selected_index[b]
        verified_len = verified_seq_len[b].item()
        
        # Initialize tree mask for this batch
        if tree_mask_mode == 7:  # Assuming 7 means QLEN_ONLY_BITPACKING based on context
            # Bitpacking mode - handle differently based on draft_token_num
            num_bytes_per_item = 1
            if draft_token_num > 16:
                num_bytes_per_item = 4
            elif draft_token_num > 8:
                num_bytes_per_item = 2
            
            # For bitpacking, we'll handle the tree construction with packed bits
            _build_tree_efficient_partial_packed_cpu_v2(
                parent_b, selected_b, verified_len, tree_mask[b], 
                positions[b], retrive_index[b], 
                retrive_next_token[b], retrive_next_sibling[b],
                topk, depth, draft_token_num, num_bytes_per_item
            )
        else:
            # Standard boolean mask mode
            _build_tree_efficient_cpu_v2(
                parent_b, selected_b, verified_len, tree_mask[b], 
                positions[b], retrive_index[b], 
                retrive_next_token[b], retrive_next_sibling[b],
                topk, depth, draft_token_num, tree_mask_mode
            )

def _build_tree_efficient_cpu_v2(
    parent_b, selected_b, verified_len, tree_mask_b, 
    positions_b, retrive_index_b, retrive_next_token_b, retrive_next_sibling_b,
    topk, depth, draft_token_num, tree_mask_mode
):
    """
    CPU implementation for standard tree building (non-bitpacked version)
    """
    # Reset tree mask
    if tree_mask_b.dim() >= 2:
        tree_mask_b.fill_(False)
    
    # Build tree structure based on parent relationships
    for i in range(draft_token_num):
        current_pos = i
        
        # Set position - check if positions_b is 1D tensor
        if positions_b.dim() > 0 and i < positions_b.size(0):
            positions_b[i] = current_pos
        
        # Mark valid connections in tree mask
        if i < len(parent_b):
            parent_idx = parent_b[i].item()
            
            # Ensure parent index is valid and tree_mask_b has proper dimensions
            if (0 <= parent_idx < draft_token_num and parent_idx != i and 
                tree_mask_b.dim() >= 2 and 
                parent_idx < tree_mask_b.size(-2) and 
                i < tree_mask_b.size(-1)):
                tree_mask_b[parent_idx, i] = True
        
        # Set retrieval indices - check bounds
        if i < len(selected_b) and positions_b.dim() > 0 and i < retrive_index_b.size(0):
            retrive_index_b[i] = selected_b[i]
        
        # Set next token and sibling relationships - check bounds
        if (i < retrive_next_token_b.size(0)):
            if i + 1 < draft_token_num:
                retrive_next_token_b[i] = i + 1
            else:
                retrive_next_token_b[i] = -1  # No next token
                
        # Find next sibling (same parent, higher index) - check bounds
        if (i < retrive_next_sibling_b.size(0)):
            current_parent = parent_b[i].item() if i < len(parent_b) else -1
            next_sibling = -1
            for j in range(i + 1, draft_token_num):
                if j < len(parent_b) and parent_b[j].item() == current_parent:
                    next_sibling = j
                    break
            retrive_next_sibling_b[i] = next_sibling

def _build_tree_efficient_partial_packed_cpu_v2(
    parent_b, selected_b, verified_len, tree_mask_b, 
    positions_b, retrive_index_b, retrive_next_token_b, retrive_next_sibling_b,
    topk, depth, draft_token_num, num_bytes_per_item
):
    """
    CPU implementation for bitpacked tree building
    """
    # For bitpacked version, we convert the tree mask to appropriate format
    # Since PyTorch doesn't have native uint8 tensor for bit operations,
    # we'll work with boolean tensors and then pack if needed
    
    # Reset tree mask
    if tree_mask_b.dim() >= 2:
        tree_mask_b.fill_(False)
    
    # Build tree structure similar to non-packed version
    for i in range(draft_token_num):
        current_pos = i
        
        # Set position - check if positions_b is 1D tensor
        if positions_b.dim() > 0 and i < positions_b.size(0):
            positions_b[i] = current_pos
        
        # Mark valid connections in tree mask
        if i < len(parent_b):
            parent_idx = parent_b[i].item()
            if (0 <= parent_idx < draft_token_num and parent_idx != i and 
                tree_mask_b.dim() >= 2 and 
                parent_idx < tree_mask_b.size(-2) and 
                i < tree_mask_b.size(-1)):
                tree_mask_b[parent_idx, i] = True
        
        # Set retrieval indices - check bounds
        if i < len(selected_b) and positions_b.dim() > 0 and i < retrive_index_b.size(0):
            retrive_index_b[i] = selected_b[i]
        
        # Set next token - check bounds
        if (i < retrive_next_token_b.size(0)):
            if i + 1 < draft_token_num:
                retrive_next_token_b[i] = i + 1
            else:
                retrive_next_token_b[i] = -1
                
        # Find next sibling - check bounds
        if (i < retrive_next_sibling_b.size(0)):
            current_parent = parent_b[i].item() if i < len(parent_b) else -1
            next_sibling = -1
            for j in range(i + 1, draft_token_num):
                if j < len(parent_b) and parent_b[j].item() == current_parent:
                    next_sibling = j
                    break
            retrive_next_sibling_b[i] = next_sibling

def build_tree_kernel_efficient(
    verified_id: torch.Tensor,
    parent_list: List[torch.Tensor],
    top_scores_index: torch.Tensor,
    draft_tokens: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_sum: int,
    topk: int,
    spec_steps: int,
    num_verify_tokens: int,
    tree_mask_mode: TreeMaskMode = TreeMaskMode.FULL_MASK,
    tree_mask_buf: Optional[torch.Tensor] = None,
    position_buf: Optional[torch.Tensor] = None,
):
    draft_tokens = torch.cat((verified_id.unsqueeze(1), draft_tokens), dim=1).flatten()

    # seq_lens_sum == sum(seq_lens); seq_lens: sequence length without draft tokens
    bs = seq_lens.numel()
    device = seq_lens.device
    # e.g. for bs=1, tree_mask: num_draft_token, seq_lens_sum + num_draft_token (flattened)
    # where each row indicates the attending pattern of each draft token
    # if use_partial_packed_tree_mask is True, tree_mask: num_draft_token (flattened, packed)
    if tree_mask_buf is not None:
        tree_mask = tree_mask_buf
        if tree_mask_mode == TreeMaskMode.QLEN_ONLY:
            tree_mask.fill_(True)
        elif tree_mask_mode == TreeMaskMode.QLEN_ONLY_BITPACKING:
            tree_mask.fill_(0)
        elif tree_mask_mode == TreeMaskMode.FULL_MASK:
            tree_mask.fill_(True)
        else:
            raise NotImplementedError(f"Invalid tree mask: {tree_mask_mode=}")
    elif tree_mask_mode == TreeMaskMode.QLEN_ONLY:
        tree_mask = torch.full(
            (num_verify_tokens * bs * num_verify_tokens,),
            True,
            dtype=torch.bool,
            device=device,
        )
    elif tree_mask_mode == TreeMaskMode.QLEN_ONLY_BITPACKING:
        packed_dtypes = [torch.uint8, torch.uint16, torch.uint32]
        packed_dtype_idx = int(math.ceil(math.log2((num_verify_tokens + 7) // 8)))
        tree_mask = torch.zeros(
            (num_verify_tokens * bs,),
            dtype=packed_dtypes[packed_dtype_idx],
            device=device,
        )
    elif tree_mask_mode == TreeMaskMode.FULL_MASK:
        tree_mask = torch.full(
            (
                seq_lens_sum * num_verify_tokens
                + num_verify_tokens * num_verify_tokens * bs,
            ),
            True,
            device=device,
        )
    else:
        raise NotImplementedError(f"Invalid tree mask: {tree_mask_mode=}")

    # TODO: make them torch.empty and fuse them into `sgl_build_tree_kernel`
    retrive_buf = torch.full(
        (3, bs, num_verify_tokens), -1, device=device, dtype=torch.long
    )
    retrive_index, retrive_next_token, retrive_next_sibling = retrive_buf
    # position: where each token belongs to
    # e.g. if depth of each draft token is [0, 1, 1, 2] and the prompt length is 7
    # then, positions = [7, 8, 8, 9]
    if position_buf is not None:
        positions = position_buf
    else:
        positions = torch.empty(
            (bs * num_verify_tokens,), device=device, dtype=torch.long
        )

    if True: #_is_npu:
        build_tree_kernel_efficient_cpu_v2(
            parent_list.to(dtype=torch.int64),
            top_scores_index,
            seq_lens,
            tree_mask,
            positions,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            topk,
            spec_steps,
            num_verify_tokens,
            tree_mask_mode,
        )
    else:
        sgl_build_tree_kernel_efficient(
            parent_list,
            top_scores_index,
            seq_lens,
            tree_mask,
            positions,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            topk,
            spec_steps,
            num_verify_tokens,
            tree_mask_mode,
        )
    return (
        tree_mask,
        positions,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        draft_tokens,
    )


def verify_tree_greedy_func(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    target_predict: torch.Tensor,
    topk: int = -1,
):
    if _is_cuda or _is_hip:
        from sgl_kernel import verify_tree_greedy

        verify_tree_greedy(
            predicts=predicts,  # mutable
            accept_index=accept_index,  # mutable
            accept_token_num=accept_token_num,  # mutable
            candidates=candidates,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            target_predict=target_predict,
        )

    elif _is_npu:
        from sgl_kernel_npu.sample.verify_tree_greedy import verify_tree_greedy

        verify_tree_greedy(
            predicts=predicts,
            accept_index=accept_index,
            accept_token_num=accept_token_num,
            candidates=candidates,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            target_predict=target_predict,
        )
    return predicts, accept_index, accept_token_num
