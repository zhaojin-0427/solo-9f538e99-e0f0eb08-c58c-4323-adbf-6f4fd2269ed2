"""划分求解器。

分配单元是连通分量（同患者 + 重复关系），整个分量只能同去训练侧或验证侧。
- 侧别登记表 (registry) 中已有记录的图像，其侧别永久固定，分量必须服从；
- 未锁定的自由分量用多维 0/1 背包 DP 选择一个子集，使验证侧各类张数精确等于配额；
- 无解时给出可解释的原因：关联冲突 / 配额不可达（含最接近状态与剩余分量）。
"""
from __future__ import annotations

from dataclasses import dataclass

from .db import TRAIN, VALID
from .parser import Component


class SolveError(Exception):
    def __init__(self, stage: str, reason: str, detail: dict | None = None):
        super().__init__(reason)
        self.stage = stage          # conflict | quota
        self.reason = reason
        self.detail = detail or {}


@dataclass
class LockedComponent:
    component: Component
    side: str
    locked_images: list[str]


@dataclass
class SolveResult:
    assignment: dict[int, str]              # cid -> side
    locked: list[LockedComponent]
    free_chosen: list[int]                  # 进入验证侧的自由分量 cid
    target: dict[str, int]
    classes: list[str]


def _vec_add(base: tuple[int, ...], vec: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(a + b for a, b in zip(base, vec))


def _vec_le(a: tuple[int, ...], b: tuple[int, ...]) -> bool:
    return all(x <= y for x, y in zip(a, b))


def solve(
    components: list[Component],
    classes: list[str],
    quota: dict[str, int],
    registry: dict[str, str],
) -> SolveResult:
    target = tuple(quota[c] for c in classes)
    ncls = len(classes)
    zero = tuple(0 for _ in classes)

    # ---- 1. 处理锁定分量，检测关联冲突 -------------------------------------
    locked_sum = [0] * ncls
    locked_map: dict[int, LockedComponent] = {}
    for comp in components:
        sides = {registry[img] for img in comp.image_nos if img in registry}
        if len(sides) > 1:
            locked_here = [
                (img, registry[img]) for img in comp.image_nos if img in registry
            ]
            detail = {
                "component_id": comp.cid,
                "images": comp.image_nos,
                "patients": sorted(comp.patients),
                "locked": locked_here,
            }
            raise SolveError(
                "conflict",
                f"关联冲突：连通分量 {comp.cid}（含图像 {', '.join(comp.image_nos)}，"
                f"患者 {', '.join(sorted(comp.patients))}）中的已发布图像被永久固定在"
                f"不同侧（{ '、'.join(sorted(sides)) }），而同患者/重复关系连通的图像必须同侧，无法划分。",
                detail,
            )
        if sides:
            side = sides.pop()
            locked_map[comp.cid] = LockedComponent(
                comp, side, [img for img in comp.image_nos if img in registry]
            )
            if side == VALID:
                for k, cls in enumerate(classes):
                    locked_sum[k] += comp.class_counts.get(cls, 0)

    locked_vec = tuple(locked_sum)
    if not _vec_le(locked_vec, target):
        over = [
            f"{classes[k]} 已固定验证 {locked_vec[k]} 张 > 配额 {target[k]} 张"
            for k in range(ncls)
            if locked_vec[k] > target[k]
        ]
        raise SolveError(
            "quota",
            "配额不可达：曾发布图像的侧别永久固定，已固定在验证侧的图像超出配额——"
            + "；".join(over)
            + "。这些侧别不能更改，请上调对应配额或更正输入。",
            {"locked_vec": locked_vec, "target": target},
        )

    need = tuple(t - l for t, l in zip(target, locked_vec))

    # ---- 2. 自由分量 DP ----------------------------------------------------
    free_comps = [c for c in components if c.cid not in locked_map]
    # 大分量优先，剪枝效果更好
    free_comps.sort(key=lambda c: (-sum(c.class_counts.values()), c.cid))
    vectors: dict[int, tuple[int, ...]] = {
        c.cid: tuple(c.class_counts.get(cls, 0) for cls in classes) for c in free_comps
    }

    chosen: list[int] = []
    if free_comps:
        # 可达状态：当前验证侧增量向量 -> (上一状态, 选入的分量 cid)
        reachable: dict[tuple[int, ...], tuple[tuple[int, ...], int | None]] = {zero: (zero, None)}
        for comp in free_comps:
            v = vectors[comp.cid]
            additions: dict[tuple[int, ...], tuple[tuple[int, ...], int | None]] = {}
            for state in reachable:
                nxt = _vec_add(state, v)
                if _vec_le(nxt, need) and nxt not in reachable and nxt not in additions:
                    additions[nxt] = (state, comp.cid)
            reachable.update(additions)

        if need not in reachable:
            reason, detail = _explain_infeasible(need, reachable, free_comps, vectors, classes, locked_vec)
            raise SolveError("quota", reason, detail)

        # ---- 3. 回溯 -------------------------------------------------------
        state = need
        while state != zero:
            prev, cid = reachable[state]
            assert cid is not None
            chosen.append(cid)
            state = prev
    elif need != zero:
        # 所有分量均已被历史登记锁定，没有可调整的自由分量
        raise SolveError(
            "quota",
            "配额不可达：本版本的全部图像都曾发布且侧别永久固定，没有可重新划分的图像，"
            "固定在验证侧的计数与新配额不一致。请恢复原配额，或更正输入引入新图像。",
            {"locked_vec": locked_vec, "target": target},
        )

    assignment: dict[int, str] = {}
    for cid, lc in locked_map.items():
        assignment[cid] = lc.side
    chosen_set = set(chosen)
    for comp in free_comps:
        assignment[comp.cid] = VALID if comp.cid in chosen_set else TRAIN

    # 最终自检：精确满足配额
    check = [0] * ncls
    for comp in components:
        if assignment[comp.cid] == VALID:
            for k, cls in enumerate(classes):
                check[k] += comp.class_counts.get(cls, 0)
    assert tuple(check) == target, "求解器内部错误：验证侧计数与配额不一致"

    return SolveResult(
        assignment=assignment,
        locked=list(locked_map.values()),
        free_chosen=sorted(chosen),
        target=dict(zip(classes, target)),
        classes=classes,
    )


def _explain_infeasible(
    need: tuple[int, ...],
    reachable: dict[tuple[int, ...], tuple[tuple[int, ...], int | None]],
    free_comps: list[Component],
    vectors: dict[int, tuple[int, ...]],
    classes: list[str],
    locked_vec: tuple[int, ...],
) -> tuple[str, dict]:
    # 最接近目标的可达状态（L1 距离最小，并列取计数较大者）
    def dist(s: tuple[int, ...]) -> tuple[int, int]:
        return (sum(abs(a - b) for a, b in zip(s, need)), -sum(s))

    best = min(reachable.keys(), key=dist)
    best_total = tuple(b + l for b, l in zip(best, locked_vec))
    target_total = tuple(n + l for n, l in zip(need, locked_vec))
    gaps = [
        f"{classes[k]} 需 {target_total[k]} 张，自由分量最多只能凑到 {best_total[k]} 张"
        for k in range(len(classes))
        if best[k] != need[k]
    ]
    # 逐类短缺诊断：某类所有自由分量之和仍不够
    class_totals = [0] * len(classes)
    for comp in free_comps:
        for k, cls in enumerate(classes):
            class_totals[k] += comp.class_counts.get(cls, 0)
    short_classes = [
        f"{classes[k]}：自由分量共含 {class_totals[k]} 张，尚缺 {need[k] - class_totals[k]} 张"
        for k in range(len(classes))
        if class_totals[k] < need[k]
    ]
    detail = {
        "need_from_free": dict(zip(classes, need)),
        "closest_reachable": dict(zip(classes, best_total)),
        "free_components": [
            {
                "component_id": c.cid,
                "images": c.image_nos,
                "patients": sorted(c.patients),
                "vector": {classes[k]: v for k, v in enumerate(vectors[c.cid]) if v},
            }
            for c in free_comps
        ],
    }
    reason = "配额不可达：不存在让验证侧各类张数精确等于配额的分量组合。"
    if short_classes:
        reason += "根本性短缺——" + "；".join(short_classes) + "。"
    if gaps:
        reason += "最接近的可行组合为（" + "；".join(gaps) + "）。"
    reason += (
        "连通分量必须整体同侧，请调整配额，或更正重复关系/患者归属后重试。"
        "注意：已发布图像的侧别永久固定，不参与重新选择。"
    )
    return reason, detail
