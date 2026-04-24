# Sweep temporal window sizes and record (accuracy, latency) to map the
# Pareto frontier.


WINDOW_SIZES = [8, 16, 32, 64, 96, 128]


def run_sweep(config_path: str) -> None:
    raise NotImplementedError
