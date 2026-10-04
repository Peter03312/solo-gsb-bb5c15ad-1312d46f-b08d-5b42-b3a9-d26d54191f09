"""归因区间明细端到端测试：需要可用的 PostgreSQL（在 Compose 环境内运行）。

运行方式：docker compose exec app pytest tests/test_attribution_detail.py
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from app.db import engine
from app.main import app

DEVICE = {"id": "CNC-01", "name": "加工中心1号"}
SHIFT = {
    "device_id": "CNC-01",
    "start": "2026-10-01T08:00:00+08:00",
    "end": "2026-10-01T16:00:00+08:00",
}
E_MAINT = {  # 保养 10:00-12:00
    "device_id": "CNC-01",
    "source_id": "MES-1",
    "version": 1,
    "start": "2026-10-01T10:00:00+08:00",
    "end": "2026-10-01T12:00:00+08:00",
    "reason": "maintenance",
}
E_FAIL = {  # 故障 11:00-13:00
    "device_id": "CNC-01",
    "source_id": "PLC-9",
    "version": 1,
    "start": "2026-10-01T11:00:00+08:00",
    "end": "2026-10-01T13:00:00+08:00",
    "reason": "failure",
}

TABLES = (
    "manual_adjudications",
    "shift_results",
    "event_versions",
    "event_sources",
    "shifts",
    "devices",
    "change_counter",
)


@pytest.fixture()
def client():
    with TestClient(app) as c:
        with engine.begin() as conn:
            for t in TABLES:
                conn.execute(text(f"TRUNCATE {t} CASCADE"))
        yield c


def _settled_shift(client) -> int:
    assert client.post("/devices", json=DEVICE).status_code == 201
    r = client.post("/shifts", json=SHIFT)
    assert r.status_code == 201
    sid = r.json()["id"]
    client.post("/events", json=E_MAINT)
    client.post("/events", json=E_FAIL)
    assert client.post(f"/shifts/{sid}/settle").status_code == 200
    return sid


def _ruling(no, reason, start, end, note="现场核对"):
    return {"ruling_no": no, "start": start, "end": end, "reason": reason, "note": note}


def _segments_minutes(body):
    sums = {"maintenance": 0, "failure": 0, "pending": 0}
    for s in body["segments"]:
        sums[s["reason"]] += s["minutes"]
    return sums


# ---------------------------------------------------------------- 基本结构


def test_settled_revision_detail_structure_and_sum(client):
    sid = _settled_shift(client)
    r = client.get(f"/shifts/{sid}/revisions/1/attribution")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True
    assert body["revision"] == 1
    # 保养 10-12(120)；故障 12-13(60)，11-12 被保养覆盖
    assert body["totals"] == {"maintenance": 120, "failure": 60, "pending": 0}

    segs = body["segments"]
    # 按起点排序、互不重叠、半开、无空档停机分钟
    starts = [s["start"] for s in segs]
    assert starts == sorted(starts)
    for s in segs:
        assert s["start"] < s["end"]
    # 区间首尾相接（无停机空档：本例全部停机分钟连续落在 10-13）
    for a, b in zip(segs, segs[1:]):
        assert a["end"] == b["start"]

    # 每段依据：事件类型，含来源/版本/原因
    by_range = {(s["start"], s["end"]): s for s in segs}
    mseg = by_range[
        ("2026-10-01T02:00:00Z", "2026-10-01T04:00:00Z")
    ]
    assert mseg["reason"] == "maintenance"
    assert mseg["basis"]["type"] == "events"
    assert mseg["basis"]["ruling_no"] is None
    assert mseg["basis"]["events"] == [
        {"source_id": "MES-1", "version": 1, "reason": "maintenance"}
    ]
    fseg = by_range[
        ("2026-10-01T04:00:00Z", "2026-10-01T05:00:00Z")
    ]
    assert fseg["reason"] == "failure"
    assert fseg["basis"]["events"] == [
        {"source_id": "PLC-9", "version": 1, "reason": "failure"}
    ]

    # 各原因区间分钟之和 == 该修订总数
    assert _segments_minutes(body) == body["totals"]
    assert sum(_segments_minutes(body).values()) == 180


def test_segments_are_half_open_and_minutes(client):
    sid = _settled_shift(client)
    body = client.get(f"/shifts/{sid}/revisions/1/attribution").json()
    from datetime import datetime

    for s in body["segments"]:
        a = datetime.fromisoformat(s["start"])
        b = datetime.fromisoformat(s["end"])
        assert s["minutes"] == int((b - a).total_seconds()) // 60


# ---------------------------------------------------------------- 裁定依据


def test_ruling_revision_segment_basis_is_ruling_no(client):
    sid = _settled_shift(client)
    # 裁定 11:00-12:00 为待判（原被保养覆盖）
    body = _ruling(
        "R-001", "pending",
        "2026-10-01T11:00:00+08:00", "2026-10-01T12:00:00+08:00",
    )
    r = client.post(f"/shifts/{sid}/adjudications", json=body)
    assert r.status_code == 201
    rev = r.json()["revision"]
    assert rev == 2

    detail = client.get(f"/shifts/{sid}/revisions/{rev}/attribution").json()
    assert detail["available"] is True
    pending = [s for s in detail["segments"] if s["reason"] == "pending"]
    assert len(pending) == 1
    seg = pending[0]
    assert seg["start"] == "2026-10-01T03:00:00Z"
    assert seg["end"] == "2026-10-01T04:00:00Z"
    assert seg["basis"] == {"type": "ruling", "ruling_no": "R-001", "events": None}
    assert _segments_minutes(detail) == detail["totals"]
    assert detail["totals"] == {"maintenance": 60, "failure": 60, "pending": 60}


def test_overlapping_ruling_then_revoke_detail_falls_back(client):
    sid = _settled_shift(client)
    client.post(
        f"/shifts/{sid}/adjudications",
        json=_ruling("R1", "pending",
                     "2026-10-01T11:00:00+08:00", "2026-10-01T12:00:00+08:00"),
    )  # rev2
    client.post(
        f"/shifts/{sid}/adjudications",
        json=_ruling("R2", "failure",
                     "2026-10-01T11:30:00+08:00", "2026-10-01T12:30:00+08:00"),
    )  # rev3：重叠 30 分钟以 R2 为准
    r = client.post(f"/shifts/{sid}/adjudications/R2/revoke")  # rev4
    assert r.status_code == 201
    rev4 = r.json()["revision"]

    detail = client.get(f"/shifts/{sid}/revisions/{rev4}/attribution").json()
    # 撤销后 11:30-12:00 回退到 R1(pending)，依据为 R1 而非事件
    bases = {(s["start"], s["end"]): s["basis"] for s in detail["segments"]}
    assert bases[("2026-10-01T03:30:00Z", "2026-10-01T04:00:00Z")] == {
        "type": "ruling", "ruling_no": "R1", "events": None
    }
    assert _segments_minutes(detail) == detail["totals"]


# ---------------------------------------------------------------- 不可变性


def test_old_revision_detail_immutable_after_correction(client):
    sid = _settled_shift(client)
    before = client.get(f"/shifts/{sid}/revisions/1/attribution").json()
    # 事件更正：保养缩短为 10:00-11:00（v2）
    r = client.post(
        "/events",
        json={**E_MAINT, "version": 2, "end": "2026-10-01T11:00:00+08:00"},
    )
    assert r.status_code == 201
    assert r.json()["recomputed"][0]["revision"] == 2

    # 修订 1 的明细保持原样（仍是保养 120）
    after = client.get(f"/shifts/{sid}/revisions/1/attribution").json()
    assert after == before
    assert after["totals"]["maintenance"] == 120
    # 修订 2 反映新状态
    rev2 = client.get(f"/shifts/{sid}/revisions/2/attribution").json()
    assert rev2["available"] is True
    assert rev2["totals"]["maintenance"] == 60
    assert _segments_minutes(rev2) == rev2["totals"]


def test_revision_detail_counters_consistent_with_result(client):
    sid = _settled_shift(client)
    client.post(
        f"/shifts/{sid}/adjudications",
        json=_ruling("R1", "pending",
                     "2026-10-01T11:00:00+08:00", "2026-10-01T12:00:00+08:00"),
    )
    for rev in (1, 2):
        result = client.get(f"/shifts/{sid}/result", params={"revision": rev}).json()
        detail = client.get(f"/shifts/{sid}/revisions/{rev}/attribution").json()
        assert detail["totals"] == result["totals"]
        assert _segments_minutes(detail) == result["totals"]


# ---------------------------------------------------------------- 历史不可用


def test_preexisting_revision_without_detail_reports_unavailable(client):
    sid = _settled_shift(client)
    # 手动把修订 1 的明细置为 NULL，模拟本轮变更前的历史修订
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE shift_results SET attribution_detail = NULL WHERE revision = 1")
        )
    r = client.get(f"/shifts/{sid}/revisions/1/attribution")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is False
    assert body["segments"] is None
    assert body["reason"]  # 明确给出不可用说明
    # totals 仍可从修订行读出，但不得据此/当前状态推算明细
    assert body["totals"]["maintenance"] == 120

    # 新生成的修订（事件更正）仍带明细
    r = client.post(
        "/events",
        json={**E_MAINT, "version": 2, "end": "2026-10-01T11:00:00+08:00"},
    )
    new_rev = r.json()["recomputed"][0]["revision"]
    detail = client.get(f"/shifts/{sid}/revisions/{new_rev}/attribution").json()
    assert detail["available"] is True
    assert detail["segments"]


# ---------------------------------------------------------------- 错误语义


def test_detail_error_semantics_match_result_query(client):
    # 未知班次
    r = client.get("/shifts/99999/revisions/1/attribution")
    assert r.status_code == 404
    assert r.json()["detail"]["code"] == "SHIFT_NOT_FOUND"

    sid = _settled_shift(client)
    # 未知修订
    r = client.get(f"/shifts/{sid}/revisions/99/attribution")
    assert r.status_code == 404
    assert r.json()["detail"]["code"] == "REVISION_NOT_FOUND"

    # 非正修订号 → 422（沿用 FastAPI 路径参数校验，不吞掉为 404）
    r = client.get(f"/shifts/{sid}/revisions/0/attribution")
    assert r.status_code == 422


def test_detail_unsettled_shift_returns_result_not_found(client):
    assert client.post("/devices", json=DEVICE).status_code == 201
    r = client.post("/shifts", json=SHIFT)
    sid = r.json()["id"]
    r = client.get(f"/shifts/{sid}/revisions/1/attribution")
    assert r.status_code == 404
    assert r.json()["detail"]["code"] == "RESULT_NOT_FOUND"
