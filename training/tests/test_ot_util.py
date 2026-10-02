from __future__ import annotations

import torch

from src.utils.ot_util import ClasswiseOTStream


def _stream(rank: int, **overrides) -> ClasswiseOTStream:
    values = dict(
        class_counts=[160 + (index % 7) for index in range(1000)],
        global_classes_per_step=16,
        samples_per_class=64,
        real_references_per_class=128,
        rank=rank,
        world_size=8,
        seed=7919,
        device="cpu",
    )
    values.update(overrides)
    return ClasswiseOTStream(**values)


def test_ot_stream_selects_16_global_classes_and_partitions_two_per_rank() -> None:
    batches = [_stream(rank).next_batch() for rank in range(8)]
    expected_global = batches[0].global_class_ids.tolist()
    assert len(expected_global) == 16
    assert len(set(expected_global)) == 16
    assert all(batch.global_class_ids.tolist() == expected_global for batch in batches)

    local_classes = []
    for rank, batch in enumerate(batches):
        assert batch.class_ids.shape == (2,)
        assert batch.generated_labels.shape == (128,)
        assert batch.target_indices.shape == (256,)
        assert batch.class_ids.tolist() == expected_global[rank::8]
        local_classes.extend(batch.class_ids.tolist())
        torch.testing.assert_close(
            batch.generated_labels, batch.class_ids.repeat_interleave(64)
        )
        torch.testing.assert_close(
            batch.target_labels, batch.class_ids.repeat_interleave(128)
        )
        for group, class_id in enumerate(batch.class_ids.tolist()):
            indices = batch.target_indices[group * 128 : (group + 1) * 128]
            assert len(set(indices.tolist())) == 128
            start = sum(_stream(rank).class_counts[:class_id].tolist())
            count = int(_stream(rank).class_counts[class_id])
            assert bool(torch.all((indices >= start) & (indices < start + count)))
    assert set(local_classes) == set(expected_global)


def test_ot_stream_keeps_global_class_rng_synchronized_across_ranks() -> None:
    streams = [_stream(rank) for rank in range(8)]
    for _ in range(3):
        batches = [stream.next_batch() for stream in streams]
        expected = batches[0].global_class_ids
        assert all(torch.equal(batch.global_class_ids, expected) for batch in batches)


def test_ot_stream_state_roundtrip_restores_exact_next_selection() -> None:
    stream = _stream(3)
    torch.manual_seed(123)
    global_state = torch.get_rng_state().clone()
    stream.next_batch()
    torch.testing.assert_close(torch.get_rng_state(), global_state)

    state = stream.state_dict()
    expected = stream.next_batch()
    expected_negative_noise = torch.randn(
        13, generator=stream.negative_noise_generator
    )
    restored = _stream(3)
    restored.load_state_dict(state)
    actual = restored.next_batch()
    for name in (
        "generated_labels",
        "target_indices",
        "target_labels",
        "class_ids",
        "global_class_ids",
    ):
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name))
    torch.testing.assert_close(
        torch.randn(13, generator=restored.negative_noise_generator),
        expected_negative_noise,
    )
