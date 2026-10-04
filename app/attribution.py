"""停机归因核心算法。

规则：
- 事件与班次均为半开区间 [start, end)；
- 区间重叠时优先级：保养(maintenance) > 故障(failure) > 待判(pending)；
- 每分钟只归因一次（高优先级覆盖低优先级）；
- 现场人工裁定（ruling）覆盖自动事件归因：被裁定覆盖的分钟完全以生效顺序最新的
  未撤销裁定为准，事件只在未被任何裁定覆盖的剩余窗口内参与归因。

本模块只依赖标准库，可脱离 Web/数据库独立测试。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable

REASON_MAINTENANCE = "maintenance"  # 保养
REASON_FAILURE = "failure"  # 故障
REASON_PENDING = "pending"  # 待判
REASONS = (REASON_MAINTENANCE, REASON_FAILURE, REASON_PENDING)

# 数值越小优先级越高
PRIORITY = {REASON_MAINTENANCE: 0, REASON_FAILURE: 1, REASON_PENDING: 2}

Interval = tuple[datetime, datetime]


@dataclass(frozen=True)
class EventView:
    """归因计算所需的事件（某来源的某个版本）。"""

    source_id: str
    version: int
    reason: str
    start: datetime
    end: datetime


@dataclass(frozen=True)
class RulingView:
    """现场人工裁定：按生效顺序（seq 越大越新）覆盖自动事件归因。"""

    ruling_no: str
    reason: str
    start: datetime
    end: datetime
    seq: int  # 生效顺序（落库顺序），重叠分钟取 seq 最大的裁定


@dataclass(frozen=True)
class AttributionSegment:
    """一段停机归因区间明细（半开 [start, end)）。

    - reason   : 该区间最终归因原因；
    - basis    : 依据。裁定区间为 {"type": "ruling", "ruling_no": ...}；
                 事件区间为 {"type": "events", "events": [{"source_id",
                 "version", "reason"}, ...]}（该区间内决定此原因的全部
                 事件版本，即归因优先级下胜出原因的覆盖事件集合）。

    同一原因、同一依据的相邻区间在产出时已合并；未停机分钟不会产生段。
    """

    start: datetime
    end: datetime
    reason: str
    basis: dict


def merge(intervals: Iterable[Interval]) -> list[Interval]:
    """合并重叠或相接的半开区间，返回按起点排序的不重叠区间列表。"""
    out: list[list[datetime]] = []
    for s, e in sorted(intervals):
        if s >= e:
            continue
        if out and s <= out[-1][1]:
            if e > out[-1][1]:
                out[-1][1] = e
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def subtract(intervals: Iterable[Interval], covers: Iterable[Interval]) -> list[Interval]:
    """从 intervals 中扣除 covers 的并集（均为半开区间）。"""
    merged_covers = merge(covers)
    out: list[Interval] = []
    for s, e in sorted(intervals):
        if s >= e:
            continue
        cur = s
        for cs, ce in merged_covers:
            if ce <= cur:
                continue
            if cs >= e:
                break
            if cs > cur:
                out.append((cur, cs))
            cur = max(cur, ce)
            if cur >= e:
                break
        if cur < e:
            out.append((cur, e))
    return out


def total_minutes(intervals: Iterable[Interval]) -> int:
    """区间列表的总分钟数（输入均已对齐整分钟）。"""
    return sum(int((e - s).total_seconds()) // 60 for s, e in intervals)


def _clip(s: datetime, e: datetime, lo: datetime, hi: datetime) -> Interval | None:
    """将区间裁剪到 [lo, hi) 内，无交集返回 None。"""
    s2, e2 = max(s, lo), min(e, hi)
    return (s2, e2) if s2 < e2 else None


def attribute(
    shift_start: datetime,
    shift_end: datetime,
    events: Iterable[EventView],
) -> tuple[dict[str, int], list[dict]]:
    """计算班次窗口 [shift_start, shift_end) 内各原因的停机分钟数。

    返回 (totals, used_versions)：
      totals        : {"maintenance": m, "failure": f, "pending": p}
      used_versions : 实际贡献了分钟数的事件版本列表
                      [{"source_id": ..., "version": ..., "reason": ...}, ...]，
                      按优先级（保养→故障→待判）排列。
    """
    clipped: dict[str, list[tuple[Interval, EventView]]] = {r: [] for r in REASONS}
    for ev in events:
        if ev.reason not in PRIORITY:
            raise ValueError(f"unknown reason: {ev.reason!r}")
        iv = _clip(ev.start, ev.end, shift_start, shift_end)
        if iv is not None:
            clipped[ev.reason].append((iv, ev))

    m_raw = [iv for iv, _ in clipped[REASON_MAINTENANCE]]
    f_raw = [iv for iv, _ in clipped[REASON_FAILURE]]
    p_raw = [iv for iv, _ in clipped[REASON_PENDING]]

    # 保养全量生效；故障扣除保养覆盖；待判扣除保养与故障覆盖 —— 保证每分钟只归因一次
    m_eff = merge(m_raw)
    f_eff = merge(subtract(f_raw, m_raw))
    p_eff = merge(subtract(p_raw, m_raw + f_raw))

    totals = {
        REASON_MAINTENANCE: total_minutes(m_eff),
        REASON_FAILURE: total_minutes(f_eff),
        REASON_PENDING: total_minutes(p_eff),
    }

    effective = {REASON_MAINTENANCE: m_eff, REASON_FAILURE: f_eff, REASON_PENDING: p_eff}
    used: list[dict] = []
    for reason in REASONS:
        for iv, ev in clipped[reason]:
            if any(s < iv[1] and iv[0] < e for s, e in effective[reason]):
                used.append(
                    {"source_id": ev.source_id, "version": ev.version, "reason": reason}
                )
    return totals, used


def _attribute_window(
    lo: datetime,
    hi: datetime,
    events: Iterable[EventView],
) -> tuple[dict[str, int], list[dict]]:
    """在 [lo, hi) 窗口内按事件归因（保养 > 故障 > 待判，每分钟只归因一次）。"""
    clipped: dict[str, list[tuple[Interval, EventView]]] = {r: [] for r in REASONS}
    for ev in events:
        if ev.reason not in PRIORITY:
            raise ValueError(f"unknown reason: {ev.reason!r}")
        iv = _clip(ev.start, ev.end, lo, hi)
        if iv is not None:
            clipped[ev.reason].append((iv, ev))

    m_raw = [iv for iv, _ in clipped[REASON_MAINTENANCE]]
    f_raw = [iv for iv, _ in clipped[REASON_FAILURE]]
    p_raw = [iv for iv, _ in clipped[REASON_PENDING]]

    # 保养全量生效；故障扣除保养覆盖；待判扣除保养与故障覆盖 —— 保证每分钟只归因一次
    m_eff = merge(m_raw)
    f_eff = merge(subtract(f_raw, m_raw))
    p_eff = merge(subtract(p_raw, m_raw + f_raw))

    totals = {
        REASON_MAINTENANCE: total_minutes(m_eff),
        REASON_FAILURE: total_minutes(f_eff),
        REASON_PENDING: total_minutes(p_eff),
    }

    effective = {REASON_MAINTENANCE: m_eff, REASON_FAILURE: f_eff, REASON_PENDING: p_eff}
    used: list[dict] = []
    for reason in REASONS:
        for iv, ev in clipped[reason]:
            if any(s < iv[1] and iv[0] < e for s, e in effective[reason]):
                used.append(
                    {"source_id": ev.source_id, "version": ev.version, "reason": reason}
                )
    return totals, used


def attribute(
    shift_start: datetime,
    shift_end: datetime,
    events: Iterable[EventView],
) -> tuple[dict[str, int], list[dict]]:
    """计算班次窗口 [shift_start, shift_end) 内各原因的停机分钟数（纯事件归因）。

    返回 (totals, used_versions)：
      totals        : {"maintenance": m, "failure": f, "pending": p}
      used_versions : 实际贡献了分钟数的事件版本列表
                      [{"source_id": ..., "version": ..., "reason": ...}, ...]，
                      按优先级（保养→故障→待判）排列。
    """
    return _attribute_window(shift_start, shift_end, events)


def attribute_with_rulings(
    shift_start: datetime,
    shift_end: datetime,
    events: Iterable[EventView],
    rulings: Iterable[RulingView],
) -> tuple[dict[str, int], list[dict]]:
    """带现场人工裁定的归因。

    - 裁定覆盖自动事件归因：被任一未撤销裁定覆盖的分钟，完全以生效顺序最新
      （seq 最大）的裁定原因为准，事件原因（含保养）一律被覆盖；
    - 事件只在未被裁定覆盖的剩余窗口（可被裁定切成多段）内，按
      保养 > 故障 > 待判归因；
    - 裁定之间不做原因优先级覆盖：重叠分钟由生效顺序最新的裁定决定
      （撤销后自然回退到此前仍有效的裁定）。

    返回 (totals, used_event_versions)；used_event_versions 仅含在剩余窗口内
    仍实际贡献分钟的事件版本（语义与纯事件归因一致，历史修订字段保持兼容）。
    """
    events = list(events)

    # 收集裁剪到班次窗口内的裁定分钟：重叠分钟保留 seq 最大者
    winner: dict[datetime, tuple[int, str]] = {}
    clipped_rulings: list[tuple[Interval, RulingView]] = []
    for r in rulings:
        if r.reason not in PRIORITY:
            raise ValueError(f"unknown ruling reason: {r.reason!r}")
        iv = _clip(r.start, r.end, shift_start, shift_end)
        if iv is None:
            continue
        clipped_rulings.append((iv, r))
        cur = iv[0]
        while cur < iv[1]:
            prev = winner.get(cur)
            if prev is None or r.seq > prev[0]:
                winner[cur] = (r.seq, r.reason)
            cur += timedelta(minutes=1)

    # 裁定分钟按原因汇总（每原因对应分钟即该原因的有效区间）
    by_reason: dict[str, list[Interval]] = {r: [] for r in REASONS}
    for minute, (_seq, reason) in winner.items():
        by_reason[reason].append((minute, minute + timedelta(minutes=1)))
    r_intervals = {r: merge(ivs) for r, ivs in by_reason.items()}

    # 事件只在未被任何裁定覆盖的剩余窗口内归因
    ruled = [iv for iv, _ in clipped_rulings]
    free = subtract([(shift_start, shift_end)], ruled)

    totals = {
        reason: total_minutes(r_intervals[reason]) for reason in REASONS
    }
    used: list[dict] = []
    for lo, hi in free:
        sub_totals, sub_used = _attribute_window(lo, hi, events)
        for reason in REASONS:
            totals[reason] += sub_totals[reason]
        used.extend(sub_used)
    # 去重（同一事件版本可能在多个剩余窗口内贡献分钟）并按原因优先级排序
    def _key(item: dict) -> tuple[int, str, int]:
        return (PRIORITY[item["reason"]], item["source_id"], item["version"])

    return totals, sorted(
        {(item["source_id"], item["version"], item["reason"]): item for item in used}.values(),
        key=_key,
    )


# --------------------------------------------------------------- 归因区间明细


def _event_ref(ev: EventView) -> dict:
    return {"source_id": ev.source_id, "version": ev.version, "reason": ev.reason}


def _basis_key(basis: dict) -> tuple:
    """依据的可比较标识：依据不同即在此处拆段。"""
    if basis["type"] == "ruling":
        return ("ruling", basis["ruling_no"])
    return (
        "events",
        tuple(
            sorted(
                (e["source_id"], e["version"], e["reason"]) for e in basis["events"]
            )
        ),
    )


def _events_basis(events: list[EventView]) -> dict:
    return {"type": "events", "events": [_event_ref(ev) for ev in events]}


def attribute_segments(
    shift_start: datetime,
    shift_end: datetime,
    events: Iterable[EventView],
    rulings: Iterable[RulingView],
) -> list[AttributionSegment]:
    """计算班次内停机归因的**区间明细**（不可变快照的原始材料）。

    逐段（而非逐分钟）判定，产出按起点排序、互不重叠的半开区间：
      - 裁定沿用现有生效顺序覆盖事件：某分钟由覆盖它的、seq 最大的
        未撤销裁定决定（裁定之间不比原因优先级），依据为该裁定单号；
      - 未裁定分钟沿用 保养 > 故障 > 待判，依据为该段内胜出原因的
        全部覆盖事件版本（与 attribute_with_rulings 中 used_versions
        口径一致：只列对最终原因实际有贡献的事件）；
      - 在依据变化处拆段（胜出裁定不同、或胜出事件集合不同）；
      - 相邻且原因、依据均相同的段合并；
      - 不产出未停机分钟（无裁定且无任何事件覆盖的分钟）。

    各段分钟之和（按原因）与 attribute_with_rulings 的 totals 严格一致。
    """
    # 裁剪到班次窗口内的事件 / 裁定（与汇总归因同一裁剪口径）
    clipped_events: list[EventView] = []
    for ev in events:
        if ev.reason not in PRIORITY:
            raise ValueError(f"unknown reason: {ev.reason!r}")
        s, e = max(ev.start, shift_start), min(ev.end, shift_end)
        if s < e:
            clipped_events.append(
                EventView(
                    source_id=ev.source_id,
                    version=ev.version,
                    reason=ev.reason,
                    start=s,
                    end=e,
                )
            )

    clipped_rulings: list[RulingView] = []
    for r in rulings:
        if r.reason not in PRIORITY:
            raise ValueError(f"unknown ruling reason: {r.reason!r}")
        s, e = max(r.start, shift_start), min(r.end, shift_end)
        if s < e:
            clipped_rulings.append(
                RulingView(ruling_no=r.ruling_no, reason=r.reason, start=s, end=e, seq=r.seq)
            )

    # 所有原始切片：每个区间（事件或裁定）各自覆盖 [start, end)。
    # 每个切片携带其 kind 与“若此切片单独决定一分钟”所需信息。
    # 先收集全部断点，切出依据恒定的最小段，再为每段选胜出者。
    raw: list[dict] = []  # {"s","e","kind","reason", winner-data}
    for ev in clipped_events:
        raw.append({"s": ev.start, "e": ev.end, "kind": "event", "ev": ev})
    for r in clipped_rulings:
        raw.append({"s": r.start, "e": r.end, "kind": "ruling", "rv": r})

    if not raw:
        return []

    # 断点（全部原始区间端点），相邻断点之间无任何覆盖关系变化
    points = sorted({shift_start, shift_end, *(x["s"] for x in raw), *(x["e"] for x in raw)})

    # 预先按起点排序原始区间，便于逐段增量判定（班次/事件规模都很小，
    # 直接线性扫描即可，保持与纯标准库算法一致的可读性）
    raw_sorted = sorted(raw, key=lambda x: (x["s"], x["e"]))

    segments: list[AttributionSegment] = []
    for lo, hi in zip(points, points[1:]):
        if lo < shift_start or hi > shift_end or lo >= hi:
            continue
        # 覆盖该最小段 [lo, hi) 的所有原始区间（半开：s < hi and e > lo）
        covering = [x for x in raw_sorted if x["s"] < hi and x["e"] > lo]
        ruling_winners = [x for x in covering if x["kind"] == "ruling"]
        if ruling_winners:
            # 生效顺序最新（seq 最大）的未撤销裁定绝对覆盖事件
            win = max(ruling_winners, key=lambda x: x["rv"].seq)["rv"]
            segments.append(
                AttributionSegment(
                    start=lo,
                    end=hi,
                    reason=win.reason,
                    basis={"type": "ruling", "ruling_no": win.ruling_no},
                )
            )
            continue
        cov_events = [x["ev"] for x in covering]
        if not cov_events:
            continue  # 未停机分钟：不产出段
        # 保养 > 故障 > 待判：取覆盖该段的事件中优先级最高（数值最小）的原因
        winner_reason = min(
            (ev.reason for ev in cov_events), key=lambda reason: PRIORITY[reason]
        )
        # 依据 = 该段内所有以胜出原因覆盖的事件版本（含并列共同决定者）
        winners = [ev for ev in cov_events if ev.reason == winner_reason]
        segments.append(
            AttributionSegment(
                start=lo,
                end=hi,
                reason=winner_reason,
                basis=_events_basis(winners),
            )
        )

    # 合并相邻（端点相接）且原因、依据均相同的段
    merged: list[AttributionSegment] = []
    for seg in segments:
        if (
            merged
            and merged[-1].end == seg.start
            and merged[-1].reason == seg.reason
            and _basis_key(merged[-1].basis) == _basis_key(seg.basis)
        ):
            merged[-1] = AttributionSegment(
                start=merged[-1].start,
                end=seg.end,
                reason=seg.reason,
                basis=seg.basis,
            )
        else:
            merged.append(seg)
    return merged


def summarize_segments(
    segments: Iterable[AttributionSegment],
) -> tuple[dict[str, int], list[dict]]:
    """从区间明细汇总 (totals, used_event_versions)。

    与 attribute_with_rulings 的返回口径一致：used 仅含实际贡献了停机
    分钟的事件版本（裁定单号不计入事件版本列表），按原因优先级排序。
    """
    totals = {r: 0 for r in REASONS}
    used_map: dict[tuple[str, int, str], dict] = {}
    for seg in segments:
        totals[seg.reason] += int((seg.end - seg.start).total_seconds()) // 60
        if seg.basis["type"] == "events":
            for ref in seg.basis["events"]:
                used_map[(ref["source_id"], ref["version"], ref["reason"])] = ref
    used = sorted(
        used_map.values(),
        key=lambda item: (PRIORITY[item["reason"]], item["source_id"], item["version"]),
    )
    return totals, used
