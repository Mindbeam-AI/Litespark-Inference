# -*- coding: utf-8 -*-

import torch
from typing import List, Optional, Tuple


class RecurrentCache:
    """Cache for recurrent states in HGRN model"""

    def __init__(self):
        self.states: List[Optional[torch.Tensor]] = []

    @classmethod
    def from_legacy_cache(cls, past_key_values, seq_len: Optional[int] = None):
        """
        Convert legacy cache format to RecurrentCache.
        For compatibility with transformers API.
        """
        cache = cls()
        if past_key_values is not None:
            # If it's already a list of states, use them directly
            if isinstance(past_key_values, (list, tuple)):
                cache.states = list(past_key_values)
        return cache

    def to_legacy_cache(self):
        """
        Convert RecurrentCache to legacy cache format (list of states).
        For compatibility with transformers API.
        """
        return tuple(self.states) if self.states else None

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Get sequence length from cache"""
        if layer_idx < len(self.states) and self.states[layer_idx] is not None:
            return self.states[layer_idx].shape[-2]  # Assuming shape is [..., seq_len, hidden_size]
        return 0

    def get_max_length(self) -> Optional[int]:
        """Get maximum sequence length across all layers"""
        max_len = 0
        for state in self.states:
            if state is not None:
                max_len = max(max_len, state.shape[-2])
        return max_len if max_len > 0 else None

    def update(self, state: torch.Tensor, layer_idx: int, seq_len: Optional[int] = None) -> torch.Tensor:
        """Update cache with new state

        Args:
            state: The state tensor (can be a tuple)
            layer_idx: Index of the layer
            seq_len: Optional sequence length (for compatibility, not used)
        """
        # Extend states list if needed
        while len(self.states) <= layer_idx:
            self.states.append(None)

        # Handle tuple states (unpack first element)
        if isinstance(state, tuple):
            state = state[0]

        self.states[layer_idx] = state
        return state

    def get(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Get state for specific layer"""
        if layer_idx < len(self.states):
            return self.states[layer_idx]
        return None

    def clear(self):
        """Clear all cached states"""
        self.states.clear()

    def __getitem__(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Make cache subscriptable: cache[layer_idx]"""
        return self.get(layer_idx)

    def __setitem__(self, layer_idx: int, state: torch.Tensor):
        """Make cache subscriptable: cache[layer_idx] = state"""
        self.update(state, layer_idx)

    def __len__(self) -> int:
        """Return number of cached layers"""
        return len(self.states)