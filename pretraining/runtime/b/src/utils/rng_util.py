import random

import numpy as np
import torch


def fix_random_seeds(seed: int = 31):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


class RNGStateManager:
    """manager to save rng state once and reset to it multiple times."""
    
    def __init__(self):
        self._py_state = None
        self._np_state = None
        self._torch_state = None
        self._cuda_states = None
        self._saved = False
    
    def save(self):
        """save current rng states."""
        self._py_state = random.getstate()
        self._np_state = np.random.get_state()
        self._torch_state = torch.get_rng_state()
        self._cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        self._saved = True
    
    def reset(self):
        """reset to saved rng states (can be called multiple times)."""
        if not self._saved:
            raise RuntimeError("must call save() before reset()")
        random.setstate(self._py_state)
        np.random.set_state(self._np_state)
        torch.set_rng_state(self._torch_state)
        if self._cuda_states is not None:
            torch.cuda.set_rng_state_all(self._cuda_states)
    
    def snapshot(self) -> dict:
        """Return a copy of the current RNG states as a plain dict."""
        return {
            "py": random.getstate(),
            "np": np.random.get_state(),
            "torch": torch.get_rng_state().clone(),
            "cuda": [s.clone() for s in torch.cuda.get_rng_state_all()]
                     if torch.cuda.is_available() else None,
        }

    def load(self, state: dict):
        """Restore RNG states from a snapshot dict."""
        random.setstate(state["py"])
        np.random.set_state(state["np"])
        torch.set_rng_state(state["torch"])
        if state["cuda"] is not None:
            torch.cuda.set_rng_state_all(state["cuda"])
