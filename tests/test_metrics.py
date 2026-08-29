# SPDX-License-Identifier: Apache-2.0

from sparsecache.metrics import answer_em, answer_f1


def test_short_answer_metrics():
    assert answer_em("The Yanzhou!", ["Yanzhou"]) == 1.0
    assert answer_f1("Yanzhou District", ["Yanzhou"]) == 2 / 3
