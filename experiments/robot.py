"""Robot policy data: koch pick-place (real arm) and pusht (sim) frame tables.

Per-episode signals: length, action jerk, idle fraction, state path length, peak reward (pusht;
its `next_success` flag is never set in this release, so the best coverage reached stands in).
Selection: drop the shakiest / idlest 5% of episodes, the shortest 5%, and (pusht) the lowest-reward quarter.
"""

import curate
from common import DATA, save, timings

out = {}
for name, uri, extra in [
    ("koch", f"{DATA}/koch/frames.lance", {}),
    ("pusht", f"{DATA}/pusht/data/frames.lance", {"max_reward": curate.robot.peak("next_reward")}),
]:
    frames = curate.load(uri)
    eps = frames.group("episode_index", order="frame_index")
    print(name, frames, "episodes:", len(eps))
    eps.signal(
        ep_len=curate.robot.length(),
        jerk=curate.robot.jerk("action"),
        idle=curate.robot.idle_fraction("action", eps=1e-3),
        path=curate.robot.path_length("observation_state"),
        **extra,
    )
    st = eps.stats(["ep_len", "jerk", "idle", "path", *extra])
    print(st)
    rule = f"jerk <= {st['jerk']['p95']} AND idle <= {st['idle']['p95']} AND ep_len >= {st['ep_len']['p5']}"
    if extra:
        rule += f" AND max_reward >= {st['max_reward']['p25']}"
    good = frames.filter(rule)
    kept_eps = len(good.group("episode_index"))
    print(f"{name}: kept {kept_eps}/{len(eps)} episodes, {len(good):,}/{len(frames):,} frames  [{rule}]")
    good.tag(f"{name}-smooth-v1")
    out[name] = {"frames": len(frames), "episodes": len(eps), "kept_episodes": kept_eps, "kept_frames": len(good), "rule": rule, "stats": st, "timings": timings(good)}

save("robot", out)
