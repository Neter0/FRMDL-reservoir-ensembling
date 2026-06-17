def build_temporal_partition_schedule(num_steps, num_partitions, overlap_fraction=0.0):
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

    boundaries = [
        int(round(i * num_steps / num_partitions))
        for i in range(num_partitions + 1)
    ]
    base_width = max(1, num_steps / num_partitions)
    overlap_steps = int(round(base_width * overlap_fraction))
    left_extension = overlap_steps // 2
    right_extension = overlap_steps - left_extension

    windows = []
    for part in range(num_partitions):
        start = boundaries[part]
        end = boundaries[part + 1]
        if part > 0:
            start = max(0, start - left_extension)
        if part < num_partitions - 1:
            end = min(num_steps, end + right_extension)
        windows.append((start, end))

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
    return schedule


def describe_temporal_partition_schedule(num_steps, num_partitions, overlap_fraction=0.0):
    schedule = build_temporal_partition_schedule(
        num_steps,
        num_partitions,
        overlap_fraction=overlap_fraction,
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
