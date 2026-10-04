"""归因算法单元测试（仅依赖标准库）。

可用 pytest 运行，也可直接：python3 tests/test_attribution.py
"""

import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.attribution import (  # noqa: E402
    EventView,
    RulingView,
    attribute,
    attribute_segments,
    attribute_with_rulings,
    merge,
    subtract,
    summarize_segments,
    total_minutes,
)

UTC = timezone.utc
BASE = datetime(2026, 10, 1, 0, 0, tzinfo=timezone.utc)


def h(hours: float = 0, minutes: float = 0) -> datetime:
    return BASE + timedelta(hours=hours, minutes=minutes)


def ev(source: str, version: int, reason: str, start: datetime, end: datetime) -> EventView:
    return EventView(source_id=source, version=version, reason=reason, start=start, end=end)


# ---------------------------------------------------------------- 区间原语


def test_merge_overlapping_and_touching():
    assert merge([(h(1), h(2)), (h(2), h(3)), (h(1, 30), h(4))]) == [(h(1), h(4))]
    assert merge([]) == []
    assert merge([(h(1), h(1))]) == []  # 空区间被丢弃


def test_subtract_basic():
    assert subtract([(h(0), h(10))], [(h(3), h(5))]) == [(h(0), h(3)), (h(5), h(10))]


def test_subtract_full_cover_and_no_cover():
    assert subtract([(h(0), h(10))], [(h(0), h(10))]) == []
    assert subtract([(h(0), h(2))], [(h(5), h(6))]) == [(h(0), h(2))]


def test_subtract_multiple_covers():
    assert subtract([(h(0), h(10))], [(h(1), h(2)), (h(4), h(6)), (h(9), h(20))]) == [
        (h(0), h(1)),
        (h(2), h(4)),
        (h(6), h(9)),
    ]


def test_total_minutes():
    assert total_minutes([(h(0), h(1)), (h(2), h(2), )]) == 60
    assert total_minutes([(h(0), h(0, 30))]) == 30


# ---------------------------------------------------------------- 归因


def test_no_events():
    totals, used = attribute(h(8), h(16), [])
    assert totals == {"maintenance": 0, "failure": 0, "pending": 0}
    assert used == []


def test_half_open_interval():
    # [08:00, 10:00) 恰好 120 分钟，10:00 这一分钟不计入
    totals, _ = attribute(h(8), h(16), [ev("S1", 1, "failure", h(8), h(10))])
    assert totals["failure"] == 120
    # 相接事件不重叠、不漏分钟
    totals, _ = attribute(
        h(8),
        h(16),
        [ev("S1", 1, "failure", h(8), h(10)), ev("S2", 1, "failure", h(10), h(11))],
    )
    assert totals["failure"] == 180


def test_event_clipped_to_shift_window():
    totals, _ = attribute(h(8), h(16), [ev("S1", 1, "failure", h(7), h(9))])
    assert totals["failure"] == 60
    # 完全在班次外的事件不参与归因，也不出现在依据版本中
    totals, used = attribute(h(8), h(16), [ev("S1", 1, "failure", h(2), h(3))])
    assert totals["failure"] == 0
    assert used == []


def test_priority_maintenance_over_failure():
    totals, used = attribute(
        h(8),
        h(16),
        [
            ev("M1", 1, "maintenance", h(10), h(11)),
            ev("F1", 1, "failure", h(10, 30), h(12)),
        ],
    )
    assert totals == {"maintenance": 60, "failure": 60, "pending": 0}
    assert {u["source_id"] for u in used} == {"M1", "F1"}


def test_priority_failure_over_pending():
    totals, used = attribute(
        h(8),
        h(16),
        [
            ev("F1", 1, "failure", h(10), h(11)),
            ev("P1", 1, "pending", h(10, 30), h(12)),
        ],
    )
    assert totals == {"maintenance": 0, "failure": 60, "pending": 60}


def test_pending_subtracts_maintenance_and_failure():
    # 待判 [10:00, 12:00)，保养 [10:00,10:30)，故障 [11:00,11:30)
    totals, _ = attribute(
        h(8),
        h(16),
        [
            ev("M1", 1, "maintenance", h(10), h(10, 30)),
            ev("F1", 1, "failure", h(11), h(11, 30)),
            ev("P1", 1, "pending", h(10), h(12)),
        ],
    )
    assert totals == {"maintenance": 30, "failure": 30, "pending": 60}


def test_each_minute_attributed_once():
    # 三者完全重叠：全部归保养
    totals, used = attribute(
        h(8),
        h(16),
        [
            ev("M1", 1, "maintenance", h(10), h(11)),
            ev("F1", 1, "failure", h(10), h(11)),
            ev("P1", 1, "pending", h(10), h(11)),
        ],
    )
    assert totals == {"maintenance": 60, "failure": 0, "pending": 0}
    # 被高优先级完全覆盖的事件不出现在依据版本中
    assert [u["source_id"] for u in used] == ["M1"]


def test_same_reason_events_union():
    # 同原因重叠事件合并计算，不重复计分钟
    totals, _ = attribute(
        h(8),
        h(16),
        [
            ev("M1", 1, "maintenance", h(10), h(11)),
            ev("M2", 1, "maintenance", h(10, 30), h(11, 30)),
        ],
    )
    assert totals["maintenance"] == 90


def test_used_versions_reflect_contribution():
    totals, used = attribute(
        h(8),
        h(16),
        [
            ev("M1", 3, "maintenance", h(10), h(11)),
            ev("F1", 2, "failure", h(10), h(10, 30)),  # 完全被保养覆盖
            ev("F2", 1, "failure", h(12), h(13)),
        ],
    )
    assert totals == {"maintenance": 60, "failure": 60, "pending": 0}
    by_source = {u["source_id"]: u["version"] for u in used}
    assert by_source == {"M1": 3, "F2": 1}


# ---------------------------------------------------------------- 现场人工裁定


def rv(no: str, reason: str, start: datetime, end: datetime, seq: int) -> RulingView:
    return RulingView(ruling_no=no, reason=reason, start=start, end=end, seq=seq)


def test_ruling_overrides_all_event_priorities():
    # 保养 [10,12) 事件；裁定 [10:30,11:30) 为故障：裁定分钟覆盖保养
    totals, used = attribute_with_rulings(
        h(8),
        h(16),
        [ev("M1", 1, "maintenance", h(10), h(12))],
        [rv("R1", "failure", h(10, 30), h(11, 30), 1)],
    )
    assert totals == {"maintenance": 60, "failure": 60, "pending": 0}
    assert [u["source_id"] for u in used] == ["M1"]


def test_ruling_split_window_event_priority_kept_in_free_parts():
    # 故障事件 [9,13)；保养事件 [10,12)；裁定 [10:30,11:30) pending
    # 裁定外的剩余窗口中保养仍优先于故障：
    # [9,10:30) -> 保养 30(10-10:30) + 故障 60(9-10)
    # 裁定 [10:30,11:30) -> pending 60
    # [11:30,13) -> 保养 30(11:30-12) + 故障 60(12-13)
    totals, used = attribute_with_rulings(
        h(8),
        h(16),
        [
            ev("F1", 1, "failure", h(9), h(13)),
            ev("M1", 1, "maintenance", h(10), h(12)),
        ],
        [rv("R1", "pending", h(10, 30), h(11, 30), 1)],
    )
    assert totals == {"maintenance": 60, "failure": 120, "pending": 60}
    assert {u["source_id"] for u in used} == {"F1", "M1"}


def test_overlapping_rulings_latest_seq_wins():
    # 两条裁定重叠 [11,12)：seq 更大者生效，原因优先级不参与裁定间覆盖
    totals, used = attribute_with_rulings(
        h(8),
        h(16),
        [],
        [
            rv("R1", "maintenance", h(10), h(12), 1),
            rv("R2", "pending", h(11), h(13), 2),
        ],
    )
    assert totals == {"maintenance": 60, "failure": 0, "pending": 120}
    assert used == []


def test_no_rulings_equals_plain_attribute():
    events = [
        ev("M1", 1, "maintenance", h(10), h(11)),
        ev("F1", 1, "failure", h(10, 30), h(12)),
    ]
    t1, u1 = attribute(h(8), h(16), events)
    t2, u2 = attribute_with_rulings(h(8), h(16), events, [])
    assert t1 == t2 == {"maintenance": 60, "failure": 60, "pending": 0}
    assert u1 == u2


def test_rulings_clipped_to_shift_window():
    totals, _ = attribute_with_rulings(
        h(8),
        h(16),
        [],
        [rv("R1", "failure", h(7), h(8, 30), 1)],
    )
    assert totals == {"maintenance": 0, "failure": 30, "pending": 0}


def test_event_fully_covered_by_ruling_dropped_from_used():
    totals, used = attribute_with_rulings(
        h(8),
        h(16),
        [ev("F1", 1, "failure", h(10), h(11))],
        [rv("R1", "maintenance", h(10), h(11), 1)],
    )
    assert totals == {"maintenance": 60, "failure": 0, "pending": 0}
    assert used == []


def test_ruling_then_free_then_ruling_windows():
    # 裁定 [9,10) 与 [11,12)；故障事件 [9,12) 只在 [10,11) 生效
    totals, used = attribute_with_rulings(
        h(8),
        h(16),
        [ev("F1", 2, "failure", h(9), h(12))],
        [
            rv("R1", "maintenance", h(9), h(10), 1),
            rv("R2", "maintenance", h(11), h(12), 2),
        ],
    )
    assert totals == {"maintenance": 120, "failure": 60, "pending": 0}
    assert used == [{"source_id": "F1", "version": 2, "reason": "failure"}]


# ---------------------------------------------------------------- 归因区间明细


def _seg_tuples(segs):
    out = []
    for s in segs:
        a = (s.start - BASE).total_seconds() // 60
        b = (s.end - BASE).total_seconds() // 60
        if s.basis["type"] == "ruling":
            basis = ("ruling", s.basis["ruling_no"])
        else:
            basis = (
                "events",
                tuple(
                    sorted(
                        (e["source_id"], e["version"]) for e in s.basis["events"]
                    )
                ),
            )
        out.append((int(a), int(b), s.reason, basis))
    return out


def test_segments_basic_priority_split():
    segs = attribute_segments(
        h(8),
        h(16),
        [
            ev("M1", 1, "maintenance", h(10), h(11)),
            ev("F1", 1, "failure", h(10, 30), h(12)),
        ],
        [],
    )
    assert _seg_tuples(segs) == [
        (600, 660, "maintenance", ("events", (("M1", 1),))),
        (660, 720, "failure", ("events", (("F1", 1),))),
    ]
    totals, used = summarize_segments(segs)
    assert totals == {"maintenance": 60, "failure": 60, "pending": 0}
    assert {u["source_id"] for u in used} == {"M1", "F1"}


def test_segments_split_where_basis_changes_and_merge_same_basis():
    # 同原因（故障）两段相接但依据事件不同 → 在依据变化处拆段
    segs = attribute_segments(
        h(8),
        h(16),
        [
            ev("A", 1, "failure", h(10), h(11)),
            ev("B", 1, "failure", h(11), h(12)),
        ],
        [],
    )
    assert _seg_tuples(segs) == [
        (600, 660, "failure", ("events", (("A", 1),))),
        (660, 720, "failure", ("events", (("B", 1),))),
    ]
    # 同一依据覆盖连续区间（断点来自其他不影响原因/依据的区间）→ 合并
    segs2 = attribute_segments(
        h(8),
        h(16),
        [
            ev("A", 1, "failure", h(10), h(11, 30)),
            ev("B", 1, "failure", h(10, 30), h(12)),
        ],
        [],
    )
    t = _seg_tuples(segs2)
    # [10,10:30) 仅 A；[10:30,11:30) A+B；[11:30,12) 仅 B
    assert t == [
        (600, 630, "failure", ("events", (("A", 1),))),
        (630, 690, "failure", ("events", (("A", 1), ("B", 1)))),
        (690, 720, "failure", ("events", (("B", 1),))),
    ]


def test_segments_ruling_basis_overrides_and_free_falls_back():
    segs = attribute_segments(
        h(8),
        h(16),
        [ev("M", 1, "maintenance", h(10), h(12))],
        [rv("R1", "failure", h(10, 30), h(11, 30), 1)],
    )
    assert _seg_tuples(segs) == [
        (600, 630, "maintenance", ("events", (("M", 1),))),
        (630, 690, "failure", ("ruling", "R1")),
        (690, 720, "maintenance", ("events", (("M", 1),))),
    ]


def test_segments_overlapping_rulings_latest_seq_is_basis():
    segs = attribute_segments(
        h(8),
        h(16),
        [],
        [
            rv("R1", "maintenance", h(10), h(12), 1),
            rv("R2", "pending", h(11), h(13), 2),
        ],
    )
    assert _seg_tuples(segs) == [
        (600, 660, "maintenance", ("ruling", "R1")),
        (660, 780, "pending", ("ruling", "R2")),
    ]


def test_segments_no_non_downtime_minutes_and_sum_equals_total():
    # 事件 [10,11) 与 [12,13)，班次 8-16：只产出停机分钟，中间/两端空档不出现
    segs = attribute_segments(
        h(8),
        h(16),
        [
            ev("F1", 1, "failure", h(10), h(11)),
            ev("F2", 1, "failure", h(12), h(13)),
        ],
        [],
    )
    assert [x[:2] for x in _seg_tuples(segs)] == [(600, 660), (720, 780)]
    totals, _ = summarize_segments(segs)
    covered = sum(
        int((s.end - s.start).total_seconds()) // 60 for s in segs
    )
    assert covered == totals["maintenance"] + totals["failure"] + totals["pending"]
    assert covered == 120


def test_segments_sorted_non_overlapping_half_open_and_clipped():
    segs = attribute_segments(
        h(8),
        h(16),
        [ev("F1", 1, "failure", h(7), h(8, 30))],  # 越界部分裁剪
        [rv("R1", "maintenance", h(15, 30), h(17), 1)],
    )
    prev = None
    for s in segs:
        assert s.start < s.end
        assert h(8) <= s.start and s.end <= h(16)
        if prev is not None:
            assert s.start >= prev
        prev = s.end
    assert _seg_tuples(segs)[0][:2] == (480, 510)
    assert _seg_tuples(segs)[-1][:2] == (930, 960)


def test_segments_empty_when_nothing():
    assert attribute_segments(h(8), h(16), [], []) == []


def test_segments_match_totals_under_all_reasons_and_ruling():
    events = [
        ev("M1", 1, "maintenance", h(10), h(12)),
        ev("F1", 1, "failure", h(11), h(13)),
        ev("P1", 1, "pending", h(12, 30), h(14)),
    ]
    rulings = [rv("R1", "pending", h(10, 30), h(11, 30), 1)]
    t_ref, u_ref = attribute_with_rulings(h(8), h(16), events, rulings)
    segs = attribute_segments(h(8), h(16), events, rulings)
    t_seg, u_seg = summarize_segments(segs)
    assert t_seg == t_ref
    assert sorted((u["source_id"], u["version"], u["reason"]) for u in u_ref) == sorted(
        (u["source_id"], u["version"], u["reason"]) for u in u_seg
    )
    # 每原因区间分钟之和等于总数
    sums = {r: 0 for r in ("maintenance", "failure", "pending")}
    for s in segs:
        sums[s.reason] += int((s.end - s.start).total_seconds()) // 60
    assert sums == t_ref


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {name}: {exc!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
