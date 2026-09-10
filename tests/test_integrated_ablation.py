from experiments.lossless_pd.integrated_ablation import ARMS, balanced_arms


def test_balanced_arms_rotates_every_cell_through_first_position():
    orders = [balanced_arms(0, repeat) for repeat in range(4)]

    assert {order[0] for order in orders} == set(ARMS)
    assert all(set(order) == set(ARMS) for order in orders)


def test_request_index_also_rotates_arm_order():
    assert balanced_arms(1, 0) == balanced_arms(0, 1)
