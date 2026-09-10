import pytest

from experiments.lossless_pd.lmcache_pd.shape_invariant_vllm import (
    expand_decode_page_metadata,
)


def test_expand_decode_metadata_reuses_pages_with_progressive_lengths():
    expanded = expand_decode_page_metadata(
        seq_lens=(35,),
        query_lens=(4,),
        block_tables=((7, 11, 19),),
        page_size=16,
    )
    assert expanded.visible_lengths == (32, 33, 34, 35)
    assert expanded.indptr == (0, 2, 5, 8, 11)
    assert expanded.indices == (
        7,
        11,
        7,
        11,
        19,
        7,
        11,
        19,
        7,
        11,
        19,
    )
    assert expanded.last_page_len == (16, 1, 2, 3)


def test_expand_decode_metadata_supports_multiple_requests():
    expanded = expand_decode_page_metadata(
        seq_lens=(18, 34),
        query_lens=(2, 2),
        block_tables=((2, 3), (5, 8, 13)),
        page_size=16,
    )
    assert expanded.visible_lengths == (17, 18, 33, 34)
    assert expanded.last_page_len == (1, 2, 1, 2)
    assert expanded.indices == (2, 3, 2, 3, 5, 8, 13, 5, 8, 13)


@pytest.mark.parametrize(
    ("seq_lens", "query_lens", "tables", "page_size"),
    [
        ((), (), (), 16),
        ((8,), (0,), ((1,),), 16),
        ((8,), (9,), ((1,),), 16),
        ((33,), (1,), ((1, 2),), 16),
        ((8,), (1,), ((1,),), 0),
    ],
)
def test_expand_decode_metadata_rejects_invalid_shapes(
    seq_lens, query_lens, tables, page_size
):
    with pytest.raises(ValueError):
        expand_decode_page_metadata(seq_lens, query_lens, tables, page_size)
