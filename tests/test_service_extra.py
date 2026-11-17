"""防御路径：发布事务中的侧别冲突复查、失败可重试、历史版本隔离。"""
import time

from app.db import get_db
from app.service import Service
from tests.conftest import make_csv, make_dup


def rows():
    return [
        ("I1", "P1", "A"), ("I2", "P2", "A"),
        ("I3", "P3", "B"), ("I4", "P4", "B"),
    ]


def test_publish_conflict_check_blocks_half_allocation(tmp_path):
    """白盒：发布事务对永久登记做最后复查，冲突时整体不写任何分配。"""
    db = str(tmp_path / "p.db")
    svc = Service(db, synchronous=True)
    svc.create_version(make_csv(rows()), make_dup([]), {"A": 1, "B": 1})
    svc.start_partition()
    m1 = svc.current_manifest()
    i1_side = next(
        e["side"] for e in svc.get_manifest_entries(m1["id"]) if e["image_no"] == "I1"
    )
    flipped = "train" if i1_side == "val" else "val"
    # 新版本 + 一个作业行，然后伪造一份与登记冲突的发布数据直接走发布事务
    vid, _ = svc.create_version(make_csv(rows()), make_dup([]), {"A": 1, "B": 1})
    conn = get_db(db)
    job_id = conn.execute(
        "INSERT INTO partition_jobs(version_id,status) VALUES (?,'running')",
        (vid,),
    ).lastrowid
    conn.commit()
    conn.close()
    bad_entries = [
        {"image_no": "I1", "patient_no": "P1", "class_name": "A",
         "component_id": 0, "side": flipped},
        {"image_no": "I2", "patient_no": "P2", "class_name": "A",
         "component_id": 1, "side": "val" if flipped == "train" else "train"},
        {"image_no": "I3", "patient_no": "P3", "class_name": "B",
         "component_id": 2, "side": "val"},
        {"image_no": "I4", "patient_no": "P4", "class_name": "B",
         "component_id": 3, "side": "train"},
    ]
    svc._publish(job_id, vid, {"entries": bad_entries, "classes": ["A", "B"],
                               "quota": {"A": 1, "B": 1}, "locked": []})
    job = svc.get_job(job_id)
    assert job["status"] == "failed"
    assert job["fail_stage"] == "conflict"
    conn = get_db(db)
    try:
        # 仍只有第一份清单，新版本无任何明细，登记未被改动
        assert conn.execute("SELECT COUNT(*) c FROM manifests").fetchone()["c"] == 1
        assert conn.execute(
            "SELECT side FROM side_registry WHERE image_no='I1'"
        ).fetchone()["side"] == i1_side
    finally:
        conn.close()
    svc.close()


def test_failed_job_allows_retry_then_success(tmp_path):
    db = str(tmp_path / "r.db")
    svc = Service(db, synchronous=True)
    bad_rows = [("I1", "P1", "A"), ("I2", "P1", "A")]  # 分量 2 张
    svc.create_version(make_csv(bad_rows), make_dup([]), {"A": 1})  # 不可达
    j1 = svc.start_partition()
    assert svc.get_job(j1)["status"] == "failed"
    # 同一输入更正配额后重试：失败不占唯一索引位，允许新作业
    svc.create_version(make_csv(bad_rows), make_dup([]), {"A": 2})
    j2 = svc.start_partition()
    assert svc.get_job(j2)["status"] == "succeeded"
    assert svc.current_manifest() is not None
    svc.close()


def test_old_version_manifest_never_downloadable(tmp_path):
    db = str(tmp_path / "h.db")
    svc = Service(db, synchronous=True)
    svc.create_version(make_csv(rows()), make_dup([]), {"A": 1, "B": 1})
    svc.start_partition()
    m1 = svc.current_manifest()
    svc.create_version(make_csv(rows()), make_dup([]), {"A": 2, "B": 0})
    # 旧版本清单在数据层仍可查（历史），但状态所属版本已 superseded
    conn = get_db(db)
    try:
        row = conn.execute(
            "SELECT m.id, v.status FROM manifests m JOIN input_versions v "
            "ON v.id=m.version_id WHERE m.id=?",
            (m1["id"],),
        ).fetchone()
        assert row["status"] == "superseded"
    finally:
        conn.close()
    assert svc.current_manifest() is None  # 当前版本无清单 → 下载停供
    svc.close()


def test_async_successful_job_writes_all_atomically(tmp_path):
    db = str(tmp_path / "a.db")
    svc = Service(db, synchronous=False)
    svc.create_version(make_csv(rows()), make_dup([]), {"A": 1, "B": 1})
    jid = svc.start_partition()
    deadline = time.time() + 10
    while time.time() < deadline:
        if svc.get_job(jid)["status"] in ("succeeded", "failed"):
            break
        time.sleep(0.1)
    job = svc.get_job(jid)
    assert job["status"] == "succeeded"
    conn = get_db(db)
    try:
        n_entries = conn.execute(
            "SELECT COUNT(*) c FROM manifest_entries WHERE manifest_id=?",
            (job["manifest_id"],),
        ).fetchone()["c"]
        n_reg = conn.execute("SELECT COUNT(*) c FROM side_registry").fetchone()["c"]
    finally:
        conn.close()
    assert n_entries == 4 and n_reg == 4
    svc.close()
