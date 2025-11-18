"""Integration test exercising the chaos monkey in dry-run mode."""

from chaos.chaos_monkey import ChaosMonkey


def test_chaos_monkey_dry_run_behaviour():
    config = {
        "dry_run": True,
        "world_size": 4,
        "current_rank": 1,
        "kill_probability": 0.2,
        "slow_probability": 0.3,
        "oom_probability": 0.1,
        "check_interval": 1,
        "random_seed": 42,
    }

    monkey = ChaosMonkey(config)
    assert monkey.get_stats()["dry_run"] is True

    started = monkey.start()
    assert started

    # Inject a deterministic event
    event_result = monkey.inject_event()
    assert isinstance(event_result, bool)

    stopped = monkey.stop()
    assert stopped

    stats = monkey.get_stats()
    assert stats["running"] is False
    assert stats["config"]["check_interval"] == 1
