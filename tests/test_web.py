"""端到端关键流程：上传、划分、锁定、更正、停供、并发、重启、无解。"""
import time

from app.db import get_db
from app.service import Service
from tests.conftest import make_csv, make_dup, upload


def rows_basic():
    return [
        ("I1", "P1", "A"), ("I2", "P2", "A"), ("I3", "P3", "A"), ("I4", "P4", "A"),
        ("I5", "P5", "B"), ("I6", "P6", "B"), ("I7", "P7", "B"),
        ("I8", "P8", "C"), ("I9", "P9", "C"), ("I10", "P10", "D"),
    ]


def test_full_happy_path_and_csv_download(client):
    c, svc = client
    resp = upload(c, rows_basic(), {"A": 2, "B": 1, "C": 1, "D": 1})
    assert resp.status_code == 200
    resp = c.post("/partition", follow_redirects=True)
    assert resp.status_code == 200
    manifest = svc.current_manifest()
    assert manifest is not None
    entries = svc.get_manifest_entries(manifest["id"])
    val = [e for e in entries if e["side"] == "val"]
    counts = {}
    for e in val:
        counts[e["class_name"]] = counts.get(e["class_name"], 0) + 1
    assert counts == {"A": 2, "B": 1, "C": 1, "D": 1}
    # 全部图像都在清单里，且只有两侧
    assert len(entries) == 10
    assert {e["side"] for e in entries} <= {"train", "val"}

    val_csv = c.get(f"/manifests/{manifest['id']}/download/val")
    train_csv = c.get(f"/manifests/{manifest['id']}/download/train")
    assert val_csv.status_code == 200
    assert train_csv.status_code == 200
    val_lines = [l for l in val_csv.data.decode("utf-8").strip().split("\n") if l]
    train_lines = [l for l in train_csv.data.decode("utf-8").strip().split("\n") if l]
    assert len(val_lines) == 1 + 5   # 表头 + 5 张验证
    assert len(train_lines) == 1 + 5


def test_validation_rejects_bad_inputs(client):
    c, _ = client
    # 超过 40 张
    big = [(f"I{i}", f"P{i}", "A") for i in range(41)]
    resp = upload(c, big, {"A": 1})
    assert "超过上限 40" in resp.data.decode("utf-8")
    # 重复图像号
    dup_id = [("I1", "P1", "A"), ("I1", "P2", "B")]
    resp = upload(c, dup_id, {"A": 1, "B": 1})
    assert "图像号重复" in resp.data.decode("utf-8")
    # 重复关系引用不存在图像
    resp = upload(c, rows_basic(), {"A": 1, "B": 1, "C": 1, "D": 1},
                  dup_text=make_dup([("I1", "NOPE")]))
    assert "不存在的图像号" in resp.data.decode("utf-8")
    # 配额超过总数
    resp = upload(c, rows_basic(), {"A": 99, "B": 0, "C": 0, "D": 0})
    assert "超过全量图像" in resp.data.decode("utf-8")
    # 校验失败不应留下任何版本
    svc = c.application.extensions["service"]
    assert svc.current_version() is None


def test_infeasible_no_manifest_and_reason_shown(client):
    c, svc = client
    # P1 两张 A 绑定，A 配额 1 不可达
    rows = [("I1", "P1", "A"), ("I2", "P1", "A"), ("I3", "P2", "B")]
    upload(c, rows, {"A": 1, "B": 0})
    c.post("/partition", follow_redirects=True)
    assert svc.current_manifest() is None
    job = svc.list_jobs()[0]
    assert job["status"] == "failed"
    assert job["fail_stage"] == "quota"
    assert "配额不可达" in job["reason"]
    # 页面能看到原因
    page = c.get(f"/jobs/{job['id']}").data.decode("utf-8")
    assert "配额不可达" in page
    # 没有任何侧别登记、没有半成品
    assert svc.registry_snapshot() == []


def test_correction_supersedes_and_stops_download_until_repartition(client):
    c, svc = client
    upload(c, rows_basic(), {"A": 2, "B": 1, "C": 1, "D": 1})
    c.post("/partition", follow_redirects=True)
    m1 = svc.current_manifest()
    assert c.get(f"/manifests/{m1['id']}/download/val").status_code == 200

    # 更正：换成一批全新图像（未被侧别登记）并改配额 → 新版本
    new_rows = [
        ("J1", "Q1", "A"), ("J2", "Q2", "A"), ("J3", "Q3", "B"),
        ("J4", "Q4", "B"), ("J5", "Q5", "C"), ("J6", "Q6", "D"),
    ]
    upload(c, new_rows, {"A": 1, "B": 1, "C": 1, "D": 0})
    v2 = svc.current_version()
    assert v2["id"] == 2
    # 旧清单下载停供
    resp = c.get(f"/manifests/{m1['id']}/download/val", follow_redirects=False)
    assert resp.status_code == 302
    page = c.get(f"/manifests/{m1['id']}/download/val", follow_redirects=True)
    assert "停供" in page.data.decode("utf-8")
    # 当前无清单，首页提示停供
    home = c.get("/").data.decode("utf-8")
    assert "下载均停供" in home
    # 重新划分成功 → 恢复
    c.post("/partition", follow_redirects=True)
    m2 = svc.current_manifest()
    assert m2["id"] != m1["id"]
    assert c.get(f"/manifests/{m2['id']}/download/val").status_code == 200


def test_registry_persists_across_delete_and_reupload(client):
    c, svc = client
    upload(c, rows_basic(), {"A": 2, "B": 1, "C": 1, "D": 1})
    c.post("/partition", follow_redirects=True)
    m1 = svc.current_manifest()
    first_side = {e["image_no"]: e["side"] for e in svc.get_manifest_entries(m1["id"])}
    locked_before = {r["image_no"]: r["side"] for r in svc.registry_snapshot()}
    assert locked_before == first_side

    # “删后重传”：同一批图像重新上传（新版本），配额与首次完全一致。
    # 所有图像都已永久登记，重新划分必须逐图像服从登记，结果与第一次完全一致。
    upload(c, rows_basic(), {"A": 2, "B": 1, "C": 1, "D": 1})
    c.post("/partition", follow_redirects=True)
    m2 = svc.current_manifest()
    second_side = {e["image_no"]: e["side"] for e in svc.get_manifest_entries(m2["id"])}
    # 所有曾发布图像侧别永久固定，与第一次完全一致
    assert first_side == second_side
    registry = {r["image_no"]: r["side"] for r in svc.registry_snapshot()}
    assert registry == first_side


def test_lock_makes_new_quota_infeasible(client):
    c, svc = client
    # 第一次：A 配额 3，I1..I3 进验证并永久锁定
    upload(c, rows_basic(), {"A": 3, "B": 0, "C": 0, "D": 0})
    c.post("/partition", follow_redirects=True)
    locked_val = [
        r["image_no"] for r in svc.registry_snapshot() if r["side"] == "val"
    ]
    assert len(locked_val) == 3
    # 更正：A 配额改为 2。锁定的 3 张 A 无法撤回 → 不可达，不发布
    upload(c, rows_basic(), {"A": 2, "B": 0, "C": 0, "D": 0})
    c.post("/partition", follow_redirects=True)
    job = svc.list_jobs()[0]
    assert job["status"] == "failed"
    assert job["fail_stage"] == "quota"
    assert svc.current_manifest() is None


def test_conflict_via_dup_edge_with_divergent_history(client):
    c, svc = client
    # 第一次：I1 锁 val、I5 锁 train（不同患者，互不连通）
    upload(c, rows_basic(), {"A": 1, "B": 0, "C": 0, "D": 0})
    c.post("/partition", follow_redirects=True)
    side_i1 = next(r["side"] for r in svc.registry_snapshot() if r["image_no"] == "I1")
    side_i5 = next(r["side"] for r in svc.registry_snapshot() if r["image_no"] == "I5")
    assert (side_i1, side_i5) == ("val", "train")
    # 更正：增加重复关系 I1-I5，把两个历史侧别不同的图像连通 → 关联冲突
    upload(c, rows_basic(), {"A": 1, "B": 0, "C": 0, "D": 0},
           dup_text=make_dup([("I1", "I5")]))
    c.post("/partition", follow_redirects=True)
    job = svc.list_jobs()[0]
    assert job["status"] == "failed"
    assert job["fail_stage"] == "conflict"
    assert "关联冲突" in job["reason"]
    assert svc.current_manifest() is None


def test_concurrent_partition_single_manifest(tmp_path):
    db = str(tmp_path / "c.db")
    svc = Service(db, synchronous=False)
    rows = rows_basic()
    svc.create_version(make_csv(rows), make_dup([]),
                       {"A": 2, "B": 1, "C": 1, "D": 1})
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=5) as pool:
        ids = list(pool.map(lambda _: svc.start_partition(), range(5)))
    # 重复发起只返回同一个作业 id
    assert len(set(ids)) == 1
    _wait_jobs(svc)
    conn = get_db(db)
    try:
        n = conn.execute(
            "SELECT COUNT(*) c FROM manifests WHERE status='published'"
        ).fetchone()["c"]
        n_reg = conn.execute("SELECT COUNT(*) c FROM side_registry").fetchone()["c"]
    finally:
        conn.close()
    assert n == 1
    assert n_reg == len(rows)
    svc.close()


def test_stale_basis_never_published(tmp_path):
    db = str(tmp_path / "stale.db")
    # 求解延迟 0.6s，期间更正输入，作业完成时必须判 stale
    svc = Service(db, synchronous=False, solve_delay=0.6)
    svc.create_version(make_csv(rows_basic()), make_dup([]),
                       {"A": 2, "B": 1, "C": 1, "D": 1})
    job_id = svc.start_partition()
    time.sleep(0.2)
    # 求解中更正
    svc.create_version(make_csv(rows_basic()), make_dup([]),
                       {"A": 1, "B": 1, "C": 1, "D": 1})
    _wait_jobs(svc)
    job = svc.get_job(job_id)
    assert job["status"] == "failed"
    assert job["fail_stage"] == "stale"
    # 旧版本没有清单，新版本也没有（更正后需重新划分）
    assert svc.current_manifest() is None
    conn = get_db(db)
    try:
        assert conn.execute("SELECT COUNT(*) c FROM manifests").fetchone()["c"] == 0
        assert conn.execute("SELECT COUNT(*) c FROM side_registry").fetchone()["c"] == 0
    finally:
        conn.close()
    svc.close()


def test_restart_recovers_interrupted_job(tmp_path):
    db = str(tmp_path / "r.db")
    svc = Service(db, synchronous=False, solve_delay=1.0)
    svc.create_version(make_csv(rows_basic()), make_dup([]),
                       {"A": 2, "B": 1, "C": 1, "D": 1})
    job_id = svc.start_partition()
    time.sleep(0.25)
    # 模拟进程重启：新 Service 初始化时恢复残留作业
    svc2 = Service(db, synchronous=True)
    job = svc2.get_job(job_id)
    assert job["status"] == "failed"
    assert job["fail_stage"] == "restart"
    conn = get_db(db)
    try:
        assert conn.execute("SELECT COUNT(*) c FROM manifests").fetchone()["c"] == 0
        assert conn.execute("SELECT COUNT(*) c FROM side_registry").fetchone()["c"] == 0
    finally:
        conn.close()
    # 重启后可正常重新划分
    j2 = svc2.start_partition()
    assert svc2.get_job(j2)["status"] == "succeeded"
    assert svc2.current_manifest() is not None
    svc.close()
    svc2.close()


def test_dup_chain_and_patient_chain_same_side(client):
    c, svc = client
    rows = [
        ("I1", "P1", "A"), ("I2", "P2", "A"), ("I3", "P3", "A"),
        ("I4", "P4", "B"), ("I5", "P5", "B"),
    ]
    # I1-I2 重复，I2-I3 重复 → 三图连通；P4、P5 独立
    upload(c, rows, {"A": 3, "B": 1}, dup_text=make_dup([("I1", "I2"), ("I2", "I3")]))
    c.post("/partition", follow_redirects=True)
    m = svc.current_manifest()
    sides = {e["image_no"]: e["side"] for e in svc.get_manifest_entries(m["id"])}
    assert sides["I1"] == sides["I2"] == sides["I3"] == "val"
    # 同患者链：新增同患者两张，必须同侧
    rows2 = rows + [("I6", "P1", "B"), ("I7", "P1", "B")]
    upload(c, rows2, {"A": 3, "B": 3})
    c.post("/partition", follow_redirects=True)
    m2 = svc.current_manifest()
    sides2 = {e["image_no"]: e["side"] for e in svc.get_manifest_entries(m2["id"])}
    assert sides2["I1"] == sides2["I6"] == sides2["I7"]


def test_history_and_registry_pages(client):
    c, _ = client
    upload(c, rows_basic(), {"A": 2, "B": 1, "C": 1, "D": 1})
    c.post("/partition", follow_redirects=True)
    upload(c, rows_basic(), {"A": 1, "B": 1, "C": 1, "D": 1})  # 更正产生历史
    hist = c.get("/history").data.decode("utf-8")
    assert "v1" in hist and "v2" in hist and "历史" in hist
    reg = c.get("/registry").data.decode("utf-8")
    assert "I1" in reg and "永久" in reg
    v1 = c.get("/versions/1").data.decode("utf-8")
    assert "原始 CSV 留档" in v1


def _wait_jobs(svc, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        jobs = svc.list_jobs()
        if all(j["status"] in ("succeeded", "failed") for j in jobs):
            return
        time.sleep(0.1)
    raise AssertionError("jobs did not finish in time")
