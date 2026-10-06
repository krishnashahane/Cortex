from cortex.engine import initial_state, new_run_id


def test_state_is_bounded():
    state = initial_state(new_run_id(), max_iterations=999)
    assert state["max_iterations"] == 30


def test_run_id_is_unique_shape():
    run_id = new_run_id()
    assert run_id.startswith("run_")
    assert len(run_id) == 14
