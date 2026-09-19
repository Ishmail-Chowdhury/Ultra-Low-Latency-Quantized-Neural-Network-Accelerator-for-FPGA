"""Whole-calendar-day partitions with feature/label windows confined to sessions."""

from models.lob.data import examples_in_range


def day_ranges(rows):
    result = {}
    previous = None
    for index, row in enumerate(rows):
        day = row.session_id.split("/")[0]
        if day != previous:
            if day in result or (previous is not None and day <= previous):
                raise ValueError("Calendar days must be distinct chronological blocks")
            result[day] = [index, index + 1]
        result[day][1] = index + 1
        previous = day
    return result


def validate_plan(rows, split_days):
    ranges = day_ranges(rows)
    if set(split_days) != {"train", "validation", "test"} or any(not split_days[k] for k in split_days):
        raise ValueError("Nonempty train, validation and test day lists are required")
    assigned = [day for name in ("train", "validation", "test") for day in split_days[name]]
    if assigned != list(ranges) or len(assigned) != len(set(assigned)):
        raise ValueError("Day partitions must cover all data exactly once, in chronological order")
    return ranges


def split_for(rows, config, split_days, name):
    ranges = validate_plan(rows, split_days)
    days = split_days[name]
    return examples_in_range(rows, config, ranges[days[0]][0], ranges[days[-1]][1])


def splits(rows, config, split_days):
    return {name: split_for(rows, config, split_days, name) for name in ("train", "validation", "test")}
