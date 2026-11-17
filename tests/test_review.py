"""标签复核裁决流程：双 CSV 校验、分歧裁决、版本新鲜度、幂等提交、端到端划分。"""
import io

import pytest

from app.db import get_db
from app.parser import InputError, parse_review_csv
from app.service import Service
from tests.conftest import make_csv, make_dup, upload


def base_rows():
    return [
        ("I1", "P1", "A"), ("I2", "P1", "A"), ("I3", "P2", "A"), ("I4", "P3", "A"),
        ("I5", "P4", "B"), ("I6", "P5", "B"), ("I7", "P6", "B"),
        ("I8", "P7", "C"), ("I9", "P8", "C"), ("I10", "P9", "D"),
    ]


QUOTA = {"A": 2, "B": 1, "C": 1, "D": 1}


def review_csv(mapping, headers=("图像号", "标签")):
    lines = [",".join(headers)]
    for img, label in mapping.items():
        lines.append(f"{img},{label}")
    return "\n".join(lines) + "\n"


def labels(rows, overrides=None):
    m = {r[0]: r[2] for r in rows}
    if overrides:
        m.update(overrides)
    return m


@pytest.fixture()
def prepared(client):
    """上传 v1 并成功划分一次，使侧别登记有内容。"""
    c, svc = client
    upload(c, base_rows(), QUOTA)
    c.post("/partition", follow_redirects=True)
    return c, svc


def _upload_reviews(client, map_a, map_b, headers=("图像号", "标签")):
    c, _ = client
    data = {
        "review_a_csv": (io.BytesIO(review_csv(map_a, headers).encode()), "a.csv"),
        "review_b_csv": (io.BytesIO(review_csv(map_b, headers).encode()), "b.csv"),
    }
    return c.post(
        "/reviews/upload",
        data=data,
        content_type="multipart/form-data",
        follow_redirects=True,
    )


def _submit(client, draft_id, decisions):
    c, _ = client
    data = {f"decision::{img}": lbl for img, lbl in decisions.items()}
    return c.post(
        f"/reviews/{draft_id}/submit",
        data=data,
        follow_redirects=True,
    )


# ---------------------------------------------------------------- 解析器
def test_parse_review_csv_headers_and_no_header():
    rows = parse_review_csv("图像号,标签\nI1,A\nI2,B\n")
    assert [(r.image_no, r.label) for r in rows] == [("I1", "A"), ("I2", "B")]
    rows2 = parse_review_csv("image_no,label\nI1,A\n")
    assert rows2[0].image_no == "I1"
    # 无表头按列顺序
    rows3 = parse_review_csv("I1,A\nI2,B\n")
    assert len(rows3) == 2
    with pytest.raises(InputError):
        parse_review_csv("图像号,标签\nI1,A\nI1,B\n")  # 重复图像号
    with pytest.raises(InputError):
        parse_review_csv("图像号,标签\nI1,\n")         # 空标签


# ---------------------------------------------------------------- 校验
def test_review_must_cover_all_current_images(prepared):
    c, svc = prepared
    full = labels(base_rows())
    # A 缺一张
    missing = dict(full)
    del missing["I10"]
    resp = _upload_reviews((c, svc), missing, full)
    assert "未覆盖当前版本全部图像" in resp.data.decode()
    # B 多一张不存在的
    extra = dict(full)
    extra["NOPE"] = "A"
    resp = _upload_reviews((c, svc), full, extra)
    assert "不存在的图像号" in resp.data.decode()
    # 校验失败不落草稿
    assert svc.list_drafts() == []


def test_review_labels_limited_to_current_classes(prepared):
    c, svc = prepared
    full_a = labels(base_rows())
    full_b = labels(base_rows(), {"I1": "X"})  # X 不在当前类别
    resp = _upload_reviews((c, svc), full_a, full_b)
    assert "当前类别之外的标签" in resp.data.decode()
    assert svc.list_drafts() == []


def test_english_header_accepted(prepared):
    c, svc = prepared
    full = labels(base_rows())
    resp = _upload_reviews((c, svc), full, full, headers=("image_no", "label"))
    assert resp.status_code == 200
    drafts = svc.list_drafts()
    assert len(drafts) == 1 and drafts[0]["disagreed_count"] == 0


# ---------------------------------------------------------------- 草稿/裁决
def test_agreed_auto_confirmed_disagreements_need_adjudication(prepared):
    c, svc = prepared
    map_a = labels(base_rows())
    map_b = labels(base_rows(), {"I1": "B", "I5": "C"})
    _upload_reviews((c, svc), map_a, map_b)
    draft = svc.list_drafts()[0]
    assert draft["agreed_count"] == 8 and draft["disagreed_count"] == 2
    items = {i["image_no"]: i for i in svc.get_draft_items(draft["id"])}
    assert items["I3"]["is_agreed"] == 1 and items["I3"]["confirmed_label"] == "A"
    assert items["I1"]["is_agreed"] == 0 and items["I1"]["confirmed_label"] is None
    # 未裁决齐不能提交
    with pytest.raises(InputError) as ei:
        svc.submit_review_draft(draft["id"], {"I1": "B"})
    assert "分歧项未经质控裁决" in str(ei.value)
    # 原版本仍 active、无新版本
    assert svc.current_version()["id"] == 1


def test_adjudication_choices_limited_to_current_classes(prepared):
    c, svc = prepared
    map_a = labels(base_rows())
    map_b = labels(base_rows(), {"I1": "B"})
    _upload_reviews((c, svc), map_a, map_b)
    draft_id = svc.list_drafts()[0]["id"]
    with pytest.raises(InputError):
        svc.save_adjudication(draft_id, {"I1": "Z"})


def test_only_one_open_draft_per_version(prepared):
    c, svc = prepared
    full = labels(base_rows())
    _upload_reviews((c, svc), full, full)
    # 同版本第二份草稿被拒
    with pytest.raises(InputError) as ei:
        svc.create_review_draft(review_csv(full), review_csv(full))
    assert "进行中的复核草稿" in str(ei.value)
    # 作废后可重新上传
    svc.discard_draft(svc.list_drafts()[0]["id"])
    new_id = svc.create_review_draft(review_csv(full), review_csv(full))
    assert new_id >= 1


# ---------------------------------------------------------------- 提交
def test_submit_generates_new_version_preserving_patients_dups_quota(prepared):
    c, svc = prepared
    rows = base_rows()
    map_a = labels(rows)
    # I1: A 改 B；I5: B 改 A；I8: C 改 D（类别集合仍为 A/B/C/D）
    map_b = labels(rows, {"I1": "B", "I5": "A", "I8": "D"})
    _upload_reviews((c, svc), map_a, map_b)
    draft_id = svc.list_drafts()[0]["id"]
    # 裁决：I1 取 B 的意见，I5 取 A 的意见，I8 维持 C 的原始标签
    new_vid = svc.submit_review_draft(
        draft_id, {"I1": "B", "I5": "A", "I8": "C"}
    )
    assert new_vid == 2
    v1, v2 = svc.get_version(1), svc.get_version(2)
    assert v1["status"] == "superseded" and v2["status"] == "active"
    # 类别集合与配额不变
    assert v2["class_names"] == v1["class_names"]
    assert v2["quotas"] == v1["quotas"]
    assert v2["image_count"] == v1["image_count"]

    conn = get_db(svc.db_path)
    try:
        old = {r["image_no"]: r for r in conn.execute(
            "SELECT * FROM images WHERE version_id=1").fetchall()}
        new = {r["image_no"]: r for r in conn.execute(
            "SELECT * FROM images WHERE version_id=2").fetchall()}
        # 患者归属逐图像一致
        for img in old:
            assert new[img]["patient_no"] == old[img]["patient_no"]
        # 标签按裁决更新
        assert new["I1"]["class_name"] == "B"
        assert new["I5"]["class_name"] == "A"
        assert new["I8"]["class_name"] == "C"
        assert new["I3"]["class_name"] == "A"
        # 重复关系沿用
        e1 = conn.execute(
            "SELECT COUNT(*) c FROM dup_edges WHERE version_id=1").fetchone()["c"]
        e2 = conn.execute(
            "SELECT COUNT(*) c FROM dup_edges WHERE version_id=2").fetchone()["c"]
        assert e1 == e2
    finally:
        conn.close()

    draft = svc.get_draft(draft_id)
    assert draft["status"] == "submitted" and draft["new_version_id"] == 2


def test_class_set_must_remain_unchanged(prepared):
    c, svc = prepared
    rows = base_rows()
    map_a = labels(rows)
    # 两位标注员一致把唯一的 D 图像 I10 改成 A → D 类消失
    map_b = labels(rows, {"I10": "A"})
    map_a2 = labels(rows, {"I10": "A"})
    _upload_reviews((c, svc), map_a2, map_b)  # 该项一致，自动确认
    draft_id = svc.list_drafts()[0]["id"]
    with pytest.raises(InputError) as ei:
        svc.submit_review_draft(draft_id)
    assert "类别集合必须不变" in str(ei.value)
    # 提交失败：原版仍 active、无新版本
    assert svc.current_version()["id"] == 1
    conn = get_db(svc.db_path)
    try:
        assert conn.execute(
            "SELECT COUNT(*) c FROM input_versions").fetchone()["c"] == 1
    finally:
        conn.close()


def test_stale_draft_cannot_submit_after_correction(prepared):
    c, svc = prepared
    rows = base_rows()
    map_a = labels(rows)
    map_b = labels(rows, {"I1": "B"})
    _upload_reviews((c, svc), map_a, map_b)
    draft_id = svc.list_drafts()[0]["id"]
    # 草稿期间更正全量数据（新输入版本）→ 草稿失效
    upload(c, rows, {"A": 3, "B": 1, "C": 1, "D": 1})
    draft = svc.get_draft(draft_id)
    assert draft["status"] == "stale"
    with pytest.raises(InputError) as ei:
        svc.submit_review_draft(draft_id, {"I1": "B"})
    assert "已被更正" in str(ei.value) or "失效" in str(ei.value)
    # 原 v2（更正版）仍是当前版本，未产生第三个版本
    assert svc.current_version()["id"] == 2
    conn = get_db(svc.db_path)
    try:
        assert conn.execute(
            "SELECT COUNT(*) c FROM input_versions").fetchone()["c"] == 2
    finally:
        conn.close()


def test_quota_correction_also_invalidates_draft(prepared):
    c, svc = prepared
    full = labels(base_rows())
    map_b = labels(base_rows(), {"I2": "B"})
    _upload_reviews((c, svc), full, map_b)
    draft_id = svc.list_drafts()[0]["id"]
    # 仅配额更正也生成新版本 → 草稿同样失效
    upload(c, base_rows(), {"A": 3, "B": 1, "C": 1, "D": 1})
    assert svc.get_draft(draft_id)["status"] == "stale"


def test_repeat_submit_is_idempotent(prepared):
    c, svc = prepared
    rows = base_rows()
    map_a = labels(rows)
    map_b = labels(rows, {"I1": "B"})
    _upload_reviews((c, svc), map_a, map_b)
    draft_id = svc.list_drafts()[0]["id"]
    first = svc.submit_review_draft(draft_id, {"I1": "B"})
    # 重复提交：返回同一新版本，不产生第二版
    second = svc.submit_review_draft(draft_id, {"I1": "B"})
    assert first == second == 2
    conn = get_db(svc.db_path)
    try:
        assert conn.execute(
            "SELECT COUNT(*) c FROM input_versions").fetchone()["c"] == 2
        assert conn.execute(
            "SELECT COUNT(*) c FROM review_drafts WHERE id=? AND new_version_id=2",
            (draft_id,)).fetchone()["c"] == 1
    finally:
        conn.close()


def test_failed_submit_keeps_original_version_and_download(prepared):
    c, svc = prepared
    m1 = svc.current_manifest()
    rows = base_rows()
    map_a = labels(rows)
    map_b = labels(rows, {"I1": "B"})
    _upload_reviews((c, svc), map_a, map_b)
    draft_id = svc.list_drafts()[0]["id"]
    # 不裁决直接提交 → 失败
    with pytest.raises(InputError):
        svc.submit_review_draft(draft_id)
    # 原版本仍 active，旧清单下载仍可用，草稿仍 open 并记录失败原因
    assert svc.current_version()["id"] == 1
    assert c.get(f"/manifests/{m1['id']}/download/val").status_code == 200
    draft = svc.get_draft(draft_id)
    assert draft["status"] == "open" and draft["fail_reason"]


def test_old_manifest_download_stops_after_review_submission(prepared):
    c, svc = prepared
    m1 = svc.current_manifest()
    assert c.get(f"/manifests/{m1['id']}/download/val").status_code == 200
    map_a = labels(base_rows())
    map_b = labels(base_rows(), {"I1": "B"})
    _upload_reviews((c, svc), map_a, map_b)
    draft_id = svc.list_drafts()[0]["id"]
    svc.submit_review_draft(draft_id, {"I1": "B"})
    # 旧清单随旧版本停供
    resp = c.get(f"/manifests/{m1['id']}/download/val", follow_redirects=False)
    assert resp.status_code == 302
    # 当前版本尚无新清单 → 首页提示停供
    home = c.get("/").data.decode()
    assert "下载均停供" in home


def test_repartition_after_review_respects_registry_and_exact_quota(prepared):
    """复核后重新划分必须遵守永久侧别与精确配额：可行则精确发布，不可达则失败不留半成品。"""
    c, svc = prepared
    m1 = svc.current_manifest()
    before = {e["image_no"]: e["side"] for e in svc.get_manifest_entries(m1["id"])}

    # 场景 1：把锁定在验证侧的 I2(A) 改成 B，使锁定验证侧 B=2 > 配额 1 → 不可达
    map_a = labels(base_rows())
    map_b = labels(base_rows(), {"I2": "B", "I6": "A"})
    _upload_reviews((c, svc), map_a, map_b)
    draft_id = svc.list_drafts()[0]["id"]
    svc.submit_review_draft(draft_id, {"I2": "B", "I6": "A"})
    c.post("/partition", follow_redirects=True)
    job = svc.list_jobs()[0]
    assert job["status"] == "failed" and job["fail_stage"] == "quota"
    assert svc.current_manifest() is None  # 不留半成品，新版本下载仍停供
    conn = get_db(svc.db_path)
    try:
        assert conn.execute(
            "SELECT COUNT(*) c FROM manifests WHERE version_id=2").fetchone()["c"] == 0
        # 永久侧别登记未被改动
        reg = {r["image_no"]: r["side"] for r in conn.execute(
            "SELECT image_no, side FROM side_registry").fetchall()}
    finally:
        conn.close()
    assert reg == before

    # 场景 2：可行的标签复核（I3、I4 均为训练侧 A，互换它们与训练侧 B 图像 I7 的标签，
    # 不改变锁定在验证侧的各类计数），重新划分成功且精确满足配额
    map_b2 = labels(base_rows(), {"I3": "B", "I7": "A"})
    did2 = svc.create_review_draft(review_csv(map_a), review_csv(map_b2))
    new_vid = svc.submit_review_draft(did2, {"I3": "B", "I7": "A"})
    assert new_vid == 3
    c.post("/partition", follow_redirects=True)
    m2 = svc.current_manifest()
    assert m2 is not None
    entries = svc.get_manifest_entries(m2["id"])
    val_counts: dict[str, int] = {}
    for e in entries:
        if e["side"] == "val":
            val_counts[e["class_name"]] = val_counts.get(e["class_name"], 0) + 1
    assert val_counts == QUOTA
    # 永久侧别：所有已登记图像的侧别与第一版一致
    after = {e["image_no"]: e["side"] for e in entries}
    for img, side in before.items():
        assert after[img] == side
    # 新清单下载恢复
    assert c.get(f"/manifests/{m2['id']}/download/val").status_code == 200


def test_repartition_infeasible_when_registry_conflicts_with_new_labels(prepared):
    """复核生成新版本后重新划分：成功则精确配额，无解则失败且不留半成品（沿用原求解规则）。"""
    c, svc = prepared
    rows = base_rows()
    # 只改一张 B 图像的标签（I5 B→A），B 类仍有 I6/I7，类别集合不变
    map_a = labels(rows)
    map_b = labels(rows, {"I5": "A"})
    _upload_reviews((c, svc), map_a, map_b)
    draft_id = svc.list_drafts()[0]["id"]
    svc.submit_review_draft(draft_id, {"I5": "A"})
    c.post("/partition", follow_redirects=True)
    job = svc.list_jobs()[0]
    # 无论成功或无解，都不允许出现半次分配；成功则精确配额，失败则无清单
    if job["status"] == "failed":
        assert svc.current_manifest() is None
        conn = get_db(svc.db_path)
        try:
            assert conn.execute(
                "SELECT COUNT(*) c FROM manifests WHERE version_id=2").fetchone()["c"] == 0
        finally:
            conn.close()
    else:
        m = svc.current_manifest()
        counts: dict[str, int] = {}
        for e in svc.get_manifest_entries(m["id"]):
            if e["side"] == "val":
                counts[e["class_name"]] = counts.get(e["class_name"], 0) + 1
        assert counts == QUOTA


# ---------------------------------------------------------------- 网页流程
def test_web_full_adjudication_flow(prepared):
    c, svc = prepared
    # 复核列表页可访问
    page = c.get("/reviews").data.decode()
    assert "上传两位标注员的复核 CSV" in page
    map_a = labels(base_rows())
    map_b = labels(base_rows(), {"I1": "B", "I9": "D"})
    resp = _upload_reviews((c, svc), map_a, map_b)
    assert resp.status_code == 200
    draft_id = svc.list_drafts()[0]["id"]
    detail = c.get(f"/reviews/{draft_id}").data.decode()
    assert "分歧" in detail and "decision::I1" in detail and "待裁决" in detail
    # 暂存
    resp = c.post(f"/reviews/{draft_id}/save",
                  data={"decision::I1": "B"}, follow_redirects=True)
    assert "裁决已暂存" in resp.data.decode()
    items = {i["image_no"]: i for i in svc.get_draft_items(draft_id)}
    assert items["I1"]["confirmed_label"] == "B" and items["I9"]["confirmed_label"] is None
    # 提交（补齐 I9）
    resp = c.post(f"/reviews/{draft_id}/submit",
                  data={"decision::I1": "B", "decision::I9": "D"},
                  follow_redirects=True)
    assert "确认标签生成新版本 v2" in resp.data.decode()
    assert svc.current_version()["id"] == 2
    # 裁决页展示已提交状态
    detail = c.get(f"/reviews/{draft_id}").data.decode()
    assert "已提交" in detail and "v2" in detail


def test_web_submit_with_unresolved_redirects_with_error(prepared):
    c, svc = prepared
    map_a = labels(base_rows())
    map_b = labels(base_rows(), {"I1": "B"})
    _upload_reviews((c, svc), map_a, map_b)
    draft_id = svc.list_drafts()[0]["id"]
    resp = c.post(f"/reviews/{draft_id}/submit", data={}, follow_redirects=True)
    assert "提交失败" in resp.data.decode()
    assert svc.current_version()["id"] == 1


def test_discard_frees_slot_and_history_visible(prepared):
    c, svc = prepared
    full = labels(base_rows())
    _upload_reviews((c, svc), full, full)
    draft_id = svc.list_drafts()[0]["id"]
    resp = c.post(f"/reviews/{draft_id}/discard", follow_redirects=True)
    assert "已作废" in resp.data.decode()
    assert svc.list_drafts()[0]["status"] == "discarded"
    # 同版本可再建草稿
    new_id = svc.create_review_draft(review_csv(full), review_csv(full))
    assert new_id != draft_id
    page = c.get("/reviews").data.decode()
    assert f"#{draft_id}" in page
