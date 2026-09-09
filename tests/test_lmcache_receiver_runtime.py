import threading

import pytest
import torch

from experiments.lossless_pd.lmcache_pd.receiver_runtime import (
    AnchorMailbox,
    inspect_anchor_message,
    parse_layers,
    parse_udp_endpoint,
)


class FakeKey:
    def __init__(self, value):
        self.value = value

    def to_string(self):
        return self.value


class FakeMeta:
    def __init__(self, address):
        self.address = address


class FakeObject:
    def __init__(self, size, address):
        self.size = size
        self.meta = FakeMeta(address)
        self.ref_count = 1
        self.data_ptr = address + 1000
        self.tensor = torch.empty((2, 36, 256, 4), dtype=torch.bfloat16)

    def get_size(self):
        return self.size

    def get_physical_size(self):
        return self.size

    def get_shape(self):
        return (2, 36, 256, 1024)

    def get_dtype(self):
        return "torch.bfloat16"

    def get_memory_format(self):
        return "MemoryFormat.KV_2LTD"

    def get_ref_count(self):
        return self.ref_count

    def ref_count_up(self):
        self.ref_count += 1

    def ref_count_down(self):
        self.ref_count -= 1


class FakeBackend:
    def __init__(self):
        self.data_lock = threading.Lock()
        self.data = {
            FakeKey("key-a"): FakeObject(10, 100),
            FakeKey("key-b"): FakeObject(20, 200),
        }
        self._req_allocated_keys = {"pd-7": ["key-a", "key-b"]}
        self.corrected_device = "cuda:0"


def notification():
    return {
        "event": "nixl_write",
        "phase": "anchor",
        "request_id": "external-7",
        "pd_request_id": "pd-7",
        "finished_ns": 9_000_000,
        "keys": ["key-a", "key-b"],
        "bytes": 30,
        "resident_bytes": 30,
        "remote_indexes": [100, 200],
        "chunk_indices": [3, 11],
        "token_ranges": [[768, 1024], [2816, 3072]],
        "prompt_tokens": 4096,
        "seed_record": {"seed_token_id": 42, "seed_sampled_ns": 1},
    }


def test_parse_udp_endpoint_validates_scheme_and_port():
    assert parse_udp_endpoint("udp://127.0.0.1:17600") == ("127.0.0.1", 17600)
    with pytest.raises(ValueError):
        parse_udp_endpoint("tcp://127.0.0.1:17600")
    with pytest.raises(ValueError):
        parse_udp_endpoint("udp://127.0.0.1:70000")


def test_parse_layers_requires_unique_non_negative_layers():
    assert parse_layers("1,9,17,25,33") == (1, 9, 17, 25, 33)
    with pytest.raises(ValueError):
        parse_layers("1,1")
    with pytest.raises(ValueError):
        parse_layers("-1")


def test_inspect_anchor_resolves_real_backend_objects_and_bytes():
    row = inspect_anchor_message(FakeBackend(), notification(), received_ns=10_000_000)
    assert row["complete"] is True
    assert row["control_plane_ms"] == 1.0
    assert row["found_keys"] == row["expected_keys"] == 2
    assert row["resolved_bytes"] == row["expected_resident_bytes"] == 30
    assert row["written_bytes"] == 30
    assert row["pd_tracked_keys"] == 2
    assert row["device"] == "cuda:0"
    assert row["token_ranges"] == [[768, 1024], [2816, 3072]]
    assert row["prompt_tokens"] == 4096
    assert [obj["address"] for obj in row["objects"]] == [100, 200]


def test_inspect_anchor_marks_missing_object_incomplete():
    message = notification()
    message["keys"].append("key-c")
    row = inspect_anchor_message(FakeBackend(), message, received_ns=10_000_000)
    assert row["complete"] is False
    assert row["missing_keys"] == ["key-c"]


def test_inspect_anchor_rejects_empty_readiness_claim():
    message = notification()
    message["keys"] = []
    message["bytes"] = message["resident_bytes"] = 0
    row = inspect_anchor_message(FakeBackend(), message, received_ns=10_000_000)
    assert row["complete"] is False


def test_inspect_anchor_accepts_receiver_deduplicated_keys():
    message = notification()
    message["bytes"] = 10
    row = inspect_anchor_message(FakeBackend(), message, received_ns=10_000_000)
    assert row["complete"] is True
    assert row["written_bytes"] == 10
    assert row["resolved_bytes"] == row["expected_resident_bytes"] == 30


def test_mailbox_claim_pins_and_release_unpins_objects():
    mailbox = AnchorMailbox.__new__(AnchorMailbox)
    mailbox.backend = FakeBackend()
    mailbox.trace_path = ""
    mailbox._condition = threading.Condition()
    mailbox._records = {}
    row = inspect_anchor_message(
        mailbox.backend, notification(), received_ns=10_000_000
    )
    mailbox.publish(row)
    assert mailbox.wait("external-7", timeout=0) is row
    objects = mailbox.claim_objects("pd-7")
    assert len(objects) == 2
    assert [obj.get_ref_count() for obj in objects] == [2, 2]
    mailbox.release_objects(objects)
    assert [obj.get_ref_count() for obj in objects] == [1, 1]


def test_mailbox_exposes_ordered_zero_copy_layer_views():
    mailbox = AnchorMailbox.__new__(AnchorMailbox)
    mailbox.backend = FakeBackend()
    mailbox.trace_path = ""
    mailbox._condition = threading.Condition()
    mailbox._records = {}
    row = inspect_anchor_message(
        mailbox.backend, notification(), received_ns=10_000_000
    )
    mailbox.publish(row)
    claim = mailbox.claim_layer_views("external-7", (1, 9, 17, 25, 33))
    assert claim is not None
    assert claim.chunk_indices == (3, 11)
    assert claim.token_ranges == ((768, 1024), (2816, 3072))
    assert claim.prompt_tokens == 4096
    assert set(claim.views) == {1, 9, 17, 25, 33}
    assert all(len(views) == 2 for views in claim.views.values())
    assert all(
        view.shape == (2, 1, 256, 4) for views in claim.views.values() for view in views
    )
    for views in claim.views.values():
        for view, owner in zip(views, claim.objects, strict=True):
            assert (
                view.untyped_storage().data_ptr()
                == owner.tensor.untyped_storage().data_ptr()
            )
    claim.release()
    assert all(obj.get_ref_count() == 1 for obj in claim.objects)
