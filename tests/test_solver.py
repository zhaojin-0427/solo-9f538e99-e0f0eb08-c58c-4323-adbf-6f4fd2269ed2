"""求解器单元测试：连通分量、锁定、精确配额、无解原因。"""
import pytest

from app.parser import ImageRow, validate_and_build
from app.solver import SolveError, solve
from app.db import TRAIN, VALID


def build(rows, edges=None):
    images = [ImageRow(*r) for r in rows]
    comps, classes = validate_and_build(images, edges or [])
    return comps, classes


def test_same_patient_must_stay_same_side():
    # 全部 A 类图像都在 P1 这一个分量（2 张），配额 1 不可达，配额 2 整体进验证
    rows = [
        ("I1", "P1", "A"), ("I2", "P1", "A"),
        ("I3", "P2", "B"),
    ]
    comps, classes = build(rows)
    with pytest.raises(SolveError) as ei:
        solve(comps, classes, {"A": 1, "B": 0}, {})
    assert ei.value.stage == "quota"
    res = solve(comps, classes, {"A": 2, "B": 0}, {})
    cid_p1 = next(c.cid for c in comps if "I1" in c.image_nos)
    assert res.assignment[cid_p1] == VALID
    # 分量内两张图像严格同侧
    assert res.assignment[cid_p1] == res.assignment[next(c.cid for c in comps if "I2" in c.image_nos)]


def test_dup_edges_connect_components():
    # I1(A) 与 I3(B) 经重复关系连通，且它们是各自类别唯一的图像
    rows = [
        ("I1", "P1", "A"), ("I3", "P3", "B"),
    ]
    comps, classes = build(rows, edges=[("I1", "I3")])
    big = next(c for c in comps if len(c.image_nos) == 2)
    assert big.class_counts == {"A": 1, "B": 1}
    res = solve(comps, classes, {"A": 1, "B": 1}, {})
    assert res.assignment[big.cid] == VALID
    # 只要 A 不要 B 不可能（分量不可分割）
    with pytest.raises(SolveError) as ei:
        solve(comps, classes, {"A": 1, "B": 0}, {})
    assert ei.value.stage == "quota"


def test_exact_quota_dp_subset_sum():
    # 5 个独立 A 类分量（每图 1 张），精确选 3
    rows = [(f"I{i}", f"P{i}", "A") for i in range(1, 6)]
    comps, classes = build(rows)
    res = solve(comps, classes, {"A": 3}, {})
    chosen = [cid for cid, side in res.assignment.items() if side == VALID]
    assert len(chosen) == 3
    assert sum(c.class_counts.get("A", 0) for c in comps if res.assignment[c.cid] == VALID) == 3


def test_registry_lock_forces_side_and_is_respected():
    rows = [("I1", "P1", "A"), ("I2", "P2", "A"), ("I3", "P3", "A")]
    comps, classes = build(rows)
    # I2 曾发布在验证侧 → 它所在分量强制 val，只需再凑 1
    res = solve(comps, classes, {"A": 2}, {"I2": VALID})
    cid_i2 = next(c.cid for c in comps if "I2" in c.image_nos)
    assert res.assignment[cid_i2] == VALID
    assert sum(1 for s in res.assignment.values() if s == VALID) == 2
    # I2 锁在训练侧：配额 2 仍可达（其余两张进 val），配额 3 不可达
    res2 = solve(comps, classes, {"A": 2}, {"I2": TRAIN})
    assert res2.assignment[cid_i2] == TRAIN
    with pytest.raises(SolveError) as ei:
        solve(comps, classes, {"A": 3}, {"I2": TRAIN})
    assert ei.value.stage == "quota"


def test_locked_val_exceeds_quota_rejected():
    rows = [("I1", "P1", "A"), ("I2", "P2", "A")]
    comps, classes = build(rows)
    with pytest.raises(SolveError) as ei:
        solve(comps, classes, {"A": 0}, {"I1": VALID})
    assert ei.value.stage == "quota"
    assert "永久固定" in ei.value.reason


def test_conflict_locked_on_both_sides_in_one_component():
    # I1、I2 同患者（同分量），但历史登记侧别不同 → 关联冲突
    rows = [("I1", "P1", "A"), ("I2", "P1", "B")]
    comps, classes = build(rows)
    with pytest.raises(SolveError) as ei:
        solve(comps, classes, {"A": 1, "B": 0}, {"I1": VALID, "I2": TRAIN})
    assert ei.value.stage == "conflict"
    assert "关联冲突" in ei.value.reason
    assert ei.value.detail["component_id"] == 0


def test_multi_class_quota_exact_vectors():
    rows = [
        ("I1", "P1", "A"), ("I2", "P1", "B"),   # 分量0: (1A,1B)
        ("I3", "P2", "A"), ("I4", "P3", "B"),   # 分量1: (1A) 分量2: (1B)
    ]
    comps, classes = build(rows)
    # 目标 (2A,1B)：选分量0+分量1
    res = solve(comps, classes, {"A": 2, "B": 1}, {})
    val_a = sum(c.class_counts.get("A", 0) for c in comps if res.assignment[c.cid] == VALID)
    val_b = sum(c.class_counts.get("B", 0) for c in comps if res.assignment[c.cid] == VALID)
    assert (val_a, val_b) == (2, 1)
    # (0A,1B) 可达：只选分量2
    res2 = solve(comps, classes, {"A": 0, "B": 1}, {})
    cid_b_only = next(c.cid for c in comps if c.image_nos == ["I4"])
    assert res2.assignment[cid_b_only] == VALID


def test_infeasible_reason_mentions_shortfall():
    # 只有 2 张 A，却要 5 张
    rows = [("I1", "P1", "A"), ("I2", "P2", "A")]
    comps, classes = build(rows)
    with pytest.raises(SolveError) as ei:
        solve(comps, classes, {"A": 5}, {})
    assert ei.value.stage == "quota"
    assert "配额不可达" in ei.value.reason
    assert "尚缺" in ei.value.reason


def test_all_components_locked_respects_registry_fully():
    # 两张图像都已登记：I1=val, I2=train；没有任何自由分量
    rows = [("I1", "P1", "A"), ("I2", "P2", "A")]
    comps, classes = build(rows)
    res = solve(comps, classes, {"A": 1}, {"I1": VALID, "I2": TRAIN})
    assert res.assignment == {0: VALID, 1: TRAIN}
    # 与登记冲突的新配额直接拒绝
    with pytest.raises(SolveError) as ei:
        solve(comps, classes, {"A": 0}, {"I1": VALID, "I2": TRAIN})
    assert ei.value.stage == "quota"
    with pytest.raises(SolveError) as ei:
        solve(comps, classes, {"A": 2}, {"I1": VALID, "I2": TRAIN})
    assert ei.value.stage == "quota"


def test_forty_images_four_classes_limit():
    from app.parser import InputError, parse_images_csv, parse_dups_csv

    rows = [(f"I{i:02d}", f"P{i:02d}", f"C{i % 4}") for i in range(40)]
    lines = ["图像号,患者号,类别"] + [",".join(r) for r in rows]
    comps, classes = build(rows)
    assert len(classes) == 4
    assert len(comps) == 40
    # 41 张触发上限
    rows41 = rows + [("I41", "P41", "C0")]
    images = [ImageRow(*r) for r in rows41]
    with pytest.raises(InputError):
        validate_and_build(images, [])
    # 第 5 类触发类别上限
    rows5 = rows + [("I41", "P41", "C4")]
    with pytest.raises(InputError):
        validate_and_build([ImageRow(*r) for r in rows5], [])
    # CSV 往返解析
    parsed = parse_images_csv("\n".join(lines))
    assert len(parsed) == 40
    assert parse_dups_csv("") == []
