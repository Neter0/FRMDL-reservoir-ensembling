import random


def _uniform_boundaries(num_steps, num_partitions):
    return [
        int(round(i * num_steps / num_partitions))
        for i in range(num_partitions + 1)
    ]


def _saccade_boundaries(num_steps, num_partitions, saccade_ms=(100, 200), frame_window_us=1000):
    if num_partitions != 3:
        return _uniform_boundaries(num_steps, num_partitions)

    scale = frame_window_us / 1000.0
    boundaries = [0]
    boundaries.extend(int(round(ms / scale)) for ms in saccade_ms)
    boundaries.append(num_steps)

    if any(boundary <= 0 or boundary >= num_steps for boundary in boundaries[1:-1]):
        return _uniform_boundaries(num_steps, num_partitions)
    if any(left >= right for left, right in zip(boundaries[:-1], boundaries[1:])):
        return _uniform_boundaries(num_steps, num_partitions)
    return boundaries


def _custom_boundaries(num_steps, num_partitions, custom_boundaries):
    if custom_boundaries is None:
        return _uniform_boundaries(num_steps, num_partitions)

    boundaries = [int(round(boundary)) for boundary in custom_boundaries]
    if len(boundaries) == num_partitions - 1:
        boundaries = [0] + boundaries + [num_steps]
    elif len(boundaries) != num_partitions + 1:
        raise ValueError("custom_boundaries must contain either internal boundaries or full boundaries")

    source_end = boundaries[-1]
    if source_end != num_steps and source_end > 0:
        boundaries = [
            int(round(boundary * num_steps / source_end))
            for boundary in boundaries
        ]
        boundaries[0] = 0
        boundaries[-1] = num_steps

    if any(boundary < 0 or boundary > num_steps for boundary in boundaries):
        raise ValueError("custom boundaries must fall within [0, num_steps]")
    if any(left >= right for left, right in zip(boundaries[:-1], boundaries[1:])):
        raise ValueError("custom boundaries must be strictly increasing")
    return boundaries


def _random_boundaries(num_steps, num_partitions, random_seed=0, min_fraction=0.15):
    if num_partitions == 1:
        return [0, num_steps]

    min_width = max(1, int(round((num_steps / num_partitions) * min_fraction)))
    available = num_steps - (num_partitions * min_width)
    if available < 0:
        return _uniform_boundaries(num_steps, num_partitions)

    rng = random.Random(random_seed)
    cuts = sorted(rng.sample(range(available + num_partitions - 1), num_partitions - 1))
    widths = []
    previous = -1
    for cut in cuts + [available + num_partitions - 1]:
        widths.append(cut - previous - 1 + min_width)
        previous = cut

    boundaries = [0]
    for width in widths:
        boundaries.append(boundaries[-1] + width)
    boundaries[-1] = num_steps
    return boundaries


def _build_base_windows(boundaries):
    return [
        (boundaries[part], boundaries[part + 1])
        for part in range(len(boundaries) - 1)
    ]


def _strict_part_for_step(step, boundaries):
    for part, (start, end) in enumerate(_build_base_windows(boundaries)):
        if start <= step < end:
            return part
    return len(boundaries) - 2


def _adjacent_overlap_windows(boundaries, overlap_steps, overlap_mode, num_steps):
    num_partitions = len(boundaries) - 1
    windows = _build_base_windows(boundaries)

    if overlap_steps == 0:
        return windows

    updated = []
    left_extension = overlap_steps // 2
    right_extension = overlap_steps - left_extension

    for part, (start, end) in enumerate(windows):
        if overlap_mode == "symmetric":
            if part > 0:
                start = max(0, start - left_extension)
            if part < num_partitions - 1:
                end = min(num_steps, end + right_extension)
        elif overlap_mode == "forward":
            if part > 0:
                start = max(0, start - overlap_steps)
        elif overlap_mode == "backward":
            if part < num_partitions - 1:
                end = min(num_steps, end + overlap_steps)
        else:
            raise ValueError("adjacent overlap mode must be symmetric, forward, or backward")
        updated.append((start, end))
    return updated


def build_temporal_partition_schedule(num_steps, num_partitions, overlap_fraction=0.0,
                                      schedule_mode="uniform", frame_window_us=1000,
                                      overlap_mode="symmetric", custom_boundaries=None,
                                      random_seed=0):
    """Return active partition indices for each time step.

    overlap_fraction is the fraction of a non-overlapping partition window that
    should be shared around each internal boundary. A value of 0.0 reproduces
    the original strict TEPRE schedule.
    """
    if num_partitions < 1:
        raise ValueError("num_partitions must be at least 1")
    if num_steps < 1:
        raise ValueError("num_steps must be at least 1")
    if overlap_fraction < 0.0 or overlap_fraction >= 1.0:
        raise ValueError("overlap_fraction must be in [0.0, 1.0)")

    if schedule_mode == "uniform":
        boundaries = _uniform_boundaries(num_steps, num_partitions)
    elif schedule_mode == "saccade":
        boundaries = _saccade_boundaries(
            num_steps,
            num_partitions,
            frame_window_us=frame_window_us,
        )
    elif schedule_mode == "event_density":
        boundaries = _custom_boundaries(num_steps, num_partitions, custom_boundaries)
    elif schedule_mode == "random_boundary":
        boundaries = _random_boundaries(num_steps, num_partitions, random_seed=random_seed)
    else:
        raise ValueError("schedule_mode must be uniform, saccade, event_density, or random_boundary")

    widths = [
        boundaries[i + 1] - boundaries[i]
        for i in range(num_partitions)
    ]
    base_width = max(1, min(widths))
    overlap_steps = int(round(base_width * overlap_fraction))
    if overlap_mode in ("symmetric", "forward", "backward"):
        windows = _adjacent_overlap_windows(boundaries, overlap_steps, overlap_mode, num_steps)
        schedule = []
        for step in range(num_steps):
            active_parts = [
                part
                for part, (start, end) in enumerate(windows)
                if start <= step < end
            ]
            if not active_parts:
                raise RuntimeError("invalid temporal schedule produced an empty step")
            schedule.append(active_parts)
    elif overlap_mode == "random":
        schedule = [[_strict_part_for_step(step, boundaries)] for step in range(num_steps)]
        duplicate_count = overlap_steps * max(0, num_partitions - 1)
        rng = random.Random(random_seed)
        candidate_steps = list(range(num_steps))
        rng.shuffle(candidate_steps)
        for step in candidate_steps[:duplicate_count]:
            base_part = schedule[step][0]
            extra_parts = [part for part in range(num_partitions) if part != base_part]
            schedule[step].append(rng.choice(extra_parts))
    else:
        raise ValueError("overlap_mode must be symmetric, forward, backward, or random")
    return schedule


def describe_temporal_partition_schedule(num_steps, num_partitions, overlap_fraction=0.0,
                                         schedule_mode="uniform", frame_window_us=1000,
                                         overlap_mode="symmetric", custom_boundaries=None,
                                         random_seed=0):
    schedule = build_temporal_partition_schedule(
        num_steps,
        num_partitions,
        overlap_fraction=overlap_fraction,
        schedule_mode=schedule_mode,
        frame_window_us=frame_window_us,
        overlap_mode=overlap_mode,
        custom_boundaries=custom_boundaries,
        random_seed=random_seed,
    )
    windows = []
    for part in range(num_partitions):
        active_steps = [
            step
            for step, active_parts in enumerate(schedule)
            if part in active_parts
        ]
        windows.append((active_steps[0], active_steps[-1] + 1))
    return windows
