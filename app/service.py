"""业务服务层：输入版本、作业调度、原子发布、历史与下载、标签复核裁决。"""
from __future__ import annotations

import csv
import io
import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from . import db as dbmod
from .db import get_db
from .parser import (
    InputError,
    parse_dups_csv,
    parse_images_csv,
    parse_quota,
    parse_review_csv,
    validate_and_build,
)
from .solver import SolveError, solve


class Service:
    def __init__(self, db_path: str, synchronous: bool = False, solve_delay: float = 0.0):
        self.db_path = db_path
        # synchronous=True：作业在请求线程内立刻执行，便于自测确定性
        self.synchronous = synchronous
        self.solve_delay = solve_delay
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="partition")
        self._lock = threading.Lock()
        dbmod.init_db(db_path)

    def close(self) -> None:
        """关闭后台线程池，避免多实例下线程跨库争用作业。"""
        self._pool.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------------ 版本
    def create_version(
        self, image_csv: str, dup_csv: str, quota_raw: dict
    ) -> tuple[int, list[str]]:
        """解析+校验全部输入，通过后整体写入新版本；旧版本立即失效。

        返回 (version_id, 类别列表)。任何一步失败都不写库。
        """
        images = parse_images_csv(image_csv)
        dup_edges = parse_dups_csv(dup_csv or "")
        components, classes = validate_and_build(images, dup_edges)
        quota = parse_quota(quota_raw, classes)
        total_imgs = len(images)
        total_quota = sum(quota.values())
        if total_quota > total_imgs:
            raise InputError(
                f"验证配额合计 {total_quota} 张，超过全量图像 {total_imgs} 张"
            )

        conn = get_db(self.db_path)
        try:
            conn.execute("BEGIN IMMEDIATE")
            vid = self._insert_version(conn, image_csv, dup_csv or "", images, dup_edges,
                                       classes, quota,
                                       note=f"用户上传于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            # 新版本生效的同一事务里，旧版本全部转为历史
            conn.execute(
                "UPDATE input_versions SET status='superseded' WHERE id<>?", (vid,)
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()
        return vid, classes

    def _insert_version(self, conn: sqlite3.Connection, image_csv: str, dup_csv: str,
                        images, dup_edges, classes: list[str], quota: dict,
                        note: str) -> int:
        """在已开启的写事务内落盘一个输入版本及其图像/重复关系，返回新版本 id。"""
        cur = conn.execute(
            "INSERT INTO input_versions(image_csv, dup_csv, image_count, "
            "class_names, quotas, note) VALUES (?,?,?,?,?,?)",
            (
                image_csv,
                dup_csv or "",
                len(images),
                ",".join(classes),
                json.dumps(quota, ensure_ascii=False),
                note,
            ),
        )
        vid = cur.lastrowid
        conn.executemany(
            "INSERT INTO images(version_id, image_no, patient_no, class_name) "
            "VALUES (?,?,?,?)",
            [(vid, r.image_no, r.patient_no, r.class_name) for r in images],
        )
        conn.executemany(
            "INSERT INTO dup_edges(version_id, image_a, image_b) VALUES (?,?,?)",
            [(vid, a, b) for a, b in dup_edges],
        )
        return vid

    def current_version(self) -> sqlite3.Row | None:
        conn = get_db(self.db_path)
        try:
            return conn.execute(
                "SELECT * FROM input_versions WHERE status='active' "
                "ORDER BY id DESC LIMIT 1"
            ).fetchone()
        finally:
            conn.close()

    def get_version(self, vid: int) -> sqlite3.Row | None:
        conn = get_db(self.db_path)
        try:
            return conn.execute(
                "SELECT * FROM input_versions WHERE id=?", (vid,)
            ).fetchone()
        finally:
            conn.close()

    def list_versions(self) -> list[sqlite3.Row]:
        conn = get_db(self.db_path)
        try:
            return list(
                conn.execute("SELECT * FROM input_versions ORDER BY id DESC").fetchall()
            )
        finally:
            conn.close()

    # ------------------------------------------------------------------ 作业
    def start_partition(self) -> int:
        version = self.current_version()
        if version is None:
            raise InputError("尚未上传任何输入数据，请先上传 CSV 并填写配额")
        # 部分唯一索引 uq_job_active 保证同版本至多一个进行中/成功的作业。
        # - 并发发起：INSERT OR IGNORE 撞约束，复用同一作业；
        # - 失败作业不占约束位，允许重新发起（新作业行，历史失败仍保留）。
        with self._lock:
            conn = get_db(self.db_path)
            try:
                cur = conn.execute(
                    "INSERT OR IGNORE INTO partition_jobs(version_id, status) "
                    "VALUES (?,'queued')",
                    (version["id"],),
                )
                if cur.rowcount == 1:
                    job_id = cur.lastrowid
                    is_new = True
                else:
                    row = conn.execute(
                        "SELECT id FROM partition_jobs WHERE version_id=? "
                        "AND status IN ('queued','running','succeeded') ORDER BY id LIMIT 1",
                        (version["id"],),
                    ).fetchone()
                    job_id = row["id"]
                    is_new = False
            finally:
                conn.close()

        # 只有新插入的作业需要投递；已有作业（含运行中/已成功）直接返回其 id
        if is_new:
            if self.synchronous:
                self.run_job(job_id)
            else:
                self._pool.submit(self.run_job, job_id)
        return job_id

    def get_job(self, job_id: int) -> sqlite3.Row | None:
        conn = get_db(self.db_path)
        try:
            return conn.execute(
                "SELECT * FROM partition_jobs WHERE id=?", (job_id,)
            ).fetchone()
        finally:
            conn.close()

    def list_jobs(self, limit: int = 20) -> list[sqlite3.Row]:
        conn = get_db(self.db_path)
        try:
            return list(
                conn.execute(
                    "SELECT j.*, v.status AS version_status FROM partition_jobs j "
                    "JOIN input_versions v ON v.id=j.version_id "
                    "ORDER BY j.id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            )
        finally:
            conn.close()

    def run_job(self, job_id: int) -> None:
        """作业主体：求解 + 原子发布。整个发布在单个写事务内，失败不留半成品。"""
        conn = get_db(self.db_path)
        try:
            conn.execute("BEGIN IMMEDIATE")
            job = conn.execute(
                "SELECT * FROM partition_jobs WHERE id=?", (job_id,)
            ).fetchone()
            if job is None or job["status"] not in ("queued", "running"):
                conn.execute("ROLLBACK")
                return
            version = conn.execute(
                "SELECT * FROM input_versions WHERE id=?", (job["version_id"],)
            ).fetchone()
            # 过期依据不得发布：求解期间输入被更正则版本已 superseded
            if version["status"] != "active":
                conn.execute(
                    "UPDATE partition_jobs SET status='failed', fail_stage='stale', "
                    "finished_at=datetime('now'), reason=? WHERE id=?",
                    (
                        "求解期间输入或配额已被更正，该版本已成为历史，过期依据不得发布。请基于当前版本重新划分。",
                        job_id,
                    ),
                )
                conn.execute("COMMIT")
                return
            # 并发保护：同版本已有成功清单则直接复用，不产生第二份
            existing = conn.execute(
                "SELECT id FROM manifests WHERE version_id=? AND status='published'",
                (version["id"],),
            ).fetchone()
            if existing:
                conn.execute(
                    "UPDATE partition_jobs SET status='succeeded', "
                    "finished_at=datetime('now'), manifest_id=? WHERE id=?",
                    (existing["id"], job_id),
                )
                conn.execute("COMMIT")
                return

            conn.execute(
                "UPDATE partition_jobs SET status='running', "
                "started_at=datetime('now') WHERE id=?",
                (job_id,),
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

        # ---- 事务外求解（CPU 工作，不持有写锁；发布前再次校验版本新鲜度）----
        try:
            result_data = self._compute(version)
        except SolveError as exc:
            self._fail_job(job_id, exc.stage, exc.reason, exc.detail)
            return
        except InputError as exc:
            self._fail_job(job_id, "validation", str(exc), {})
            return
        except Exception as exc:  # 兜底：任何意外都收敛为失败作业，绝不悬挂
            self._fail_job(job_id, "error", f"求解发生未预期错误：{exc}", {})
            return

        self._publish(job_id, version["id"], result_data)

    def _compute(self, version: sqlite3.Row) -> dict:
        images = parse_images_csv(version["image_csv"])
        dup_edges = parse_dups_csv(version["dup_csv"])
        components, classes = validate_and_build(images, dup_edges)
        quota = json.loads(version["quotas"])
        conn = get_db(self.db_path)
        try:
            registry = {
                r["image_no"]: r["side"]
                for r in conn.execute("SELECT image_no, side FROM side_registry").fetchall()
            }
        finally:
            conn.close()
        if self.solve_delay:
            import time

            time.sleep(self.solve_delay)
        result = solve(components, classes, quota, registry)
        img_index = {r.image_no: r for r in images}
        entries = []
        for comp in components:
            side = result.assignment[comp.cid]
            for img in comp.image_nos:
                row = img_index[img]
                entries.append(
                    {
                        "image_no": img,
                        "patient_no": row.patient_no,
                        "class_name": row.class_name,
                        "component_id": comp.cid,
                        "side": side,
                    }
                )
        return {
            "entries": entries,
            "classes": classes,
            "quota": quota,
            "locked": [
                {
                    "component_id": lc.component.cid,
                    "side": lc.side,
                    "locked_images": lc.locked_images,
                }
                for lc in result.locked
            ],
        }

    def _publish(self, job_id: int, version_id: int, data: dict) -> None:
        """原子发布：版本新鲜度复查 → 清单 → 侧别登记 → 作业成功，同一事务。"""
        conn = get_db(self.db_path)
        try:
            conn.execute("BEGIN IMMEDIATE")
            version = conn.execute(
                "SELECT status FROM input_versions WHERE id=?", (version_id,)
            ).fetchone()
            if version["status"] != "active":
                conn.execute(
                    "UPDATE partition_jobs SET status='failed', fail_stage='stale', "
                    "finished_at=datetime('now'), reason=? WHERE id=?",
                    ("求解期间输入或配额已被更正，过期依据不得发布。", job_id),
                )
                conn.execute("COMMIT")
                return
            dup = conn.execute(
                "SELECT id FROM manifests WHERE version_id=? AND status='published'",
                (version_id,),
            ).fetchone()
            if dup:
                conn.execute(
                    "UPDATE partition_jobs SET status='succeeded', "
                    "finished_at=datetime('now'), manifest_id=? WHERE id=?",
                    (dup["id"], job_id),
                )
                conn.execute("COMMIT")
                return

            # 登记侧别前再次核对：永久登记与本次分配必须一致，冲突则整体失败
            for e in data["entries"]:
                row = conn.execute(
                    "SELECT side FROM side_registry WHERE image_no=?",
                    (e["image_no"],),
                ).fetchone()
                if row is not None and row["side"] != e["side"]:
                    conn.execute(
                        "UPDATE partition_jobs SET status='failed', fail_stage='conflict', "
                        "finished_at=datetime('now'), reason=? WHERE id=?",
                        (
                            f"关联冲突：图像 {e['image_no']} 曾发布且侧别永久固定为 "
                            f"{row['side']}，本次求解结果为 {e['side']}，已中止发布，未写入任何分配。",
                            job_id,
                        ),
                    )
                    conn.execute("COMMIT")
                    return

            cur = conn.execute(
                "INSERT INTO manifests(version_id, status, reason) "
                "VALUES (?,'published', NULL)",
                (version_id,),
            )
            manifest_id = cur.lastrowid
            conn.executemany(
                "INSERT INTO manifest_entries(manifest_id, image_no, patient_no, "
                "class_name, component_id, side) VALUES (?,?,?,?,?,?)",
                [
                    (
                        manifest_id,
                        e["image_no"],
                        e["patient_no"],
                        e["class_name"],
                        e["component_id"],
                        e["side"],
                    )
                    for e in data["entries"]
                ],
            )
            # 永久侧别登记：已存在的不动（INSERT OR IGNORE 配合上面的冲突复查）
            conn.executemany(
                "INSERT OR IGNORE INTO side_registry(image_no, side, first_manifest_id) "
                "VALUES (?,?,?)",
                [
                    (e["image_no"], e["side"], manifest_id) for e in data["entries"]
                ],
            )
            conn.execute(
                "UPDATE partition_jobs SET status='succeeded', finished_at=datetime('now'), "
                "manifest_id=? WHERE id=?",
                (manifest_id, job_id),
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            # 事务回滚后把作业标记失败（独立事务），避免悬挂
            try:
                self._fail_job(job_id, "error", "发布时发生数据库错误，事务已整体回滚，未留下半次分配。", {})
            except Exception:
                pass
            raise
        finally:
            conn.close()

    def _fail_job(self, job_id: int, stage: str, reason: str, detail: dict) -> None:
        # 失败只登记在 partition_jobs（历史/原因页展示），绝不产生半成品 manifest。
        # reason 存人类可读说明，detail 单独成列存结构化诊断。
        conn = get_db(self.db_path)
        try:
            conn.execute(
                "UPDATE partition_jobs SET status='failed', fail_stage=?, reason=?, "
                "detail=?, finished_at=datetime('now') WHERE id=?",
                (stage, reason, json.dumps(detail, ensure_ascii=False) if detail else None, job_id),
            )
        finally:
            conn.close()

    # ------------------------------------------------------------------ 清单
    def current_manifest(self) -> sqlite3.Row | None:
        version = self.current_version()
        if version is None:
            return None
        conn = get_db(self.db_path)
        try:
            return conn.execute(
                "SELECT * FROM manifests WHERE version_id=? AND status='published'",
                (version["id"],),
            ).fetchone()
        finally:
            conn.close()

    def get_manifest_entries(self, manifest_id: int) -> list[sqlite3.Row]:
        conn = get_db(self.db_path)
        try:
            return list(
                conn.execute(
                    "SELECT * FROM manifest_entries WHERE manifest_id=? "
                    "ORDER BY side DESC, class_name, image_no",
                    (manifest_id,),
                ).fetchall()
            )
        finally:
            conn.close()

    def list_manifests(self) -> list[sqlite3.Row]:
        conn = get_db(self.db_path)
        try:
            return list(
                conn.execute(
                    "SELECT m.*, v.status AS version_status, v.image_count "
                    "FROM manifests m JOIN input_versions v ON v.id=m.version_id "
                    "ORDER BY m.id DESC"
                ).fetchall()
            )
        finally:
            conn.close()

    def registry_snapshot(self) -> list[sqlite3.Row]:
        conn = get_db(self.db_path)
        try:
            return list(
                conn.execute(
                    "SELECT * FROM side_registry ORDER BY image_no"
                ).fetchall()
            )
        finally:
            conn.close()

    # -------------------------------------------------------------- 标签复核
    def create_review_draft(self, review_a_csv: str, review_b_csv: str) -> int:
        """上传两份标注员复核 CSV，校验后生成绑定当前版本的复核草稿。

        规则：
        - 两份 CSV 均须覆盖当前版本的全部图像（不缺、不多、不重复）；
        - 标签必须属于当前版本的类别集合；
        - 一致标签直接进入草稿确认标签，分歧项留待质控员逐项裁决；
        - 同一输入版本至多一个进行中草稿（部分唯一索引硬约束）。
        """
        version = self.current_version()
        if version is None:
            raise InputError("尚未上传任何输入数据，请先上传全量图像 CSV")
        classes = version["class_names"].split(",")
        class_set = set(classes)

        rows_a = parse_review_csv(review_a_csv)
        rows_b = parse_review_csv(review_b_csv)
        map_a = {r.image_no: r.label for r in rows_a}
        map_b = {r.image_no: r.label for r in rows_b}

        conn = get_db(self.db_path)
        try:
            current_images = [
                r["image_no"]
                for r in conn.execute(
                    "SELECT image_no FROM images WHERE version_id=? ORDER BY image_no",
                    (version["id"],),
                ).fetchall()
            ]
        finally:
            conn.close()
        image_set = set(current_images)

        # 覆盖性：两份都必须恰好覆盖当前版本全部图像
        for name, mapping in (("标注员 A", map_a), ("标注员 B", map_b)):
            missing = sorted(image_set - set(mapping))
            extra = sorted(set(mapping) - image_set)
            if missing:
                raise InputError(
                    f"{name}复核 CSV 未覆盖当前版本全部图像，缺少 {len(missing)} 张："
                    + "、".join(missing[:8]) + ("…" if len(missing) > 8 else "")
                )
            if extra:
                raise InputError(
                    f"{name}复核 CSV 含当前版本不存在的图像号："
                    + "、".join(extra[:8]) + ("…" if len(extra) > 8 else "")
                )
            # 标签域：限当前类别
            bad = sorted({lbl for lbl in mapping.values() if lbl not in class_set})
            if bad:
                raise InputError(
                    f"{name}复核 CSV 含当前类别之外的标签："
                    + "、".join(bad)
                    + f"（当前版本类别：{'、'.join(classes)}）"
                )

        agreed = sum(1 for img in current_images if map_a[img] == map_b[img])
        disagreed = len(current_images) - agreed

        conn = get_db(self.db_path)
        try:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT id FROM review_drafts WHERE source_version_id=? AND status='open'",
                (version["id"],),
            ).fetchone()
            if existing:
                raise InputError(
                    f"当前版本已有进行中的复核草稿（#{existing['id']}），请先完成裁决并提交，"
                    "或在复核页作废后再上传新的复核 CSV。"
                )
            cur = conn.execute(
                "INSERT INTO review_drafts(source_version_id, status, review_a_csv, "
                "review_b_csv, agreed_count, disagreed_count) VALUES (?,'open',?,?,?,?)",
                (version["id"], review_a_csv, review_b_csv, agreed, disagreed),
            )
            draft_id = cur.lastrowid
            conn.executemany(
                "INSERT INTO review_items(draft_id, image_no, label_a, label_b, "
                "is_agreed, confirmed_label) VALUES (?,?,?,?,?,?)",
                [
                    (
                        draft_id,
                        img,
                        map_a[img],
                        map_b[img],
                        1 if map_a[img] == map_b[img] else 0,
                        map_a[img] if map_a[img] == map_b[img] else None,
                    )
                    for img in current_images
                ],
            )
            conn.execute("COMMIT")
        except sqlite3.IntegrityError as exc:
            conn.execute("ROLLBACK")
            raise InputError(f"复核草稿保存失败（可能已有进行中草稿）：{exc}")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()
        return draft_id

    def _mark_stale_if_needed(self, conn: sqlite3.Connection, draft: sqlite3.Row) -> sqlite3.Row:
        """草稿绑定的版本已被更正时，惰性把开放草稿标记为 stale（不可再提交）。"""
        if draft["status"] == "open":
            v = conn.execute(
                "SELECT status FROM input_versions WHERE id=?",
                (draft["source_version_id"],),
            ).fetchone()
            if v is None or v["status"] != "active":
                conn.execute(
                    "UPDATE review_drafts SET status='stale' WHERE id=? AND status='open'",
                    (draft["id"],),
                )
                draft = conn.execute(
                    "SELECT * FROM review_drafts WHERE id=?", (draft["id"],)
                ).fetchone()
        return draft

    def get_draft(self, draft_id: int) -> sqlite3.Row | None:
        conn = get_db(self.db_path)
        try:
            draft = conn.execute(
                "SELECT * FROM review_drafts WHERE id=?", (draft_id,)
            ).fetchone()
            if draft is None:
                return None
            draft = self._mark_stale_if_needed(conn, draft)
            return draft
        finally:
            conn.close()

    def get_draft_items(self, draft_id: int) -> list[sqlite3.Row]:
        conn = get_db(self.db_path)
        try:
            return list(
                conn.execute(
                    "SELECT * FROM review_items WHERE draft_id=? ORDER BY image_no",
                    (draft_id,),
                ).fetchall()
            )
        finally:
            conn.close()

    def list_drafts(self) -> list[sqlite3.Row]:
        conn = get_db(self.db_path)
        try:
            # 统一做一次惰性失效处理
            for d in conn.execute(
                "SELECT * FROM review_drafts WHERE status='open'"
            ).fetchall():
                self._mark_stale_if_needed(conn, d)
            return list(
                conn.execute(
                    "SELECT d.*, v.status AS version_status FROM review_drafts d "
                    "JOIN input_versions v ON v.id=d.source_version_id "
                    "ORDER BY d.id DESC"
                ).fetchall()
            )
        finally:
            conn.close()

    def save_adjudication(self, draft_id: int, decisions: dict[str, str]) -> None:
        """暂存质控员对分歧项的裁决（不生成新版本）。"""
        conn = get_db(self.db_path)
        try:
            conn.execute("BEGIN IMMEDIATE")
            draft = conn.execute(
                "SELECT * FROM review_drafts WHERE id=?", (draft_id,)
            ).fetchone()
            if draft is None:
                raise InputError("复核草稿不存在")
            draft = self._mark_stale_if_needed(conn, draft)
            if draft["status"] != "open":
                raise InputError(
                    f"草稿 #{draft_id} 状态为 {draft['status']}，不能再裁决；"
                    "绑定版本已被更正或草稿已提交/作废。"
                )
            version = conn.execute(
                "SELECT class_names FROM input_versions WHERE id=?",
                (draft["source_version_id"],),
            ).fetchone()
            class_set = set(version["class_names"].split(","))
            for item in conn.execute(
                "SELECT * FROM review_items WHERE draft_id=?", (draft_id,)
            ).fetchall():
                if item["is_agreed"]:
                    continue  # 一致项确认标签已固定，不接受修改
                label = decisions.get(item["image_no"], "")
                if label == "":
                    continue  # 暂存：未裁决的分歧项允许留空
                if label not in class_set:
                    raise InputError(
                        f"图像 {item['image_no']} 的裁决标签「{label}」不在当前类别集合内"
                    )
                conn.execute(
                    "UPDATE review_items SET confirmed_label=? WHERE id=?",
                    (label, item["id"]),
                )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    def discard_draft(self, draft_id: int) -> None:
        """作废进行中的草稿，释放该版本的复核名额。"""
        conn = get_db(self.db_path)
        try:
            conn.execute("BEGIN IMMEDIATE")
            cur = conn.execute(
                "UPDATE review_drafts SET status='discarded' WHERE id=? AND status='open'",
                (draft_id,),
            )
            if cur.rowcount == 0:
                raise InputError("草稿不存在或已不可作废（已提交/已作废/已失效）")
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    def submit_review_draft(self, draft_id: int, decisions: dict[str, str] | None = None) -> int:
        """提交复核草稿：裁决齐全且依据版本仍有效时，一次生成确认标签的新版本。

        - 提交失败（版本过期/裁决缺失/标签越界/任何数据库错误）整体回滚，
          原版本、原清单及其下载状态保持不变；
        - 同一草稿重复提交不产生第二版，直接返回已生成的新版本 id；
        - 新版本沿用原版本的患者归属、重复关系与配额，仅替换类别标签，
          类别集合保持不变；旧版本同事务转 superseded，旧清单随之停供。
        返回新版本 id。
        """
        conn = get_db(self.db_path)
        try:
            conn.execute("BEGIN IMMEDIATE")
            draft = conn.execute(
                "SELECT * FROM review_drafts WHERE id=?", (draft_id,)
            ).fetchone()
            if draft is None:
                raise InputError("复核草稿不存在")
            # 幂等：已成功提交过的草稿直接返回既有新版本，绝不产生第二版
            if draft["status"] == "submitted" and draft["new_version_id"] is not None:
                conn.execute("COMMIT")
                return draft["new_version_id"]
            draft = self._mark_stale_if_needed(conn, draft)
            if draft["status"] != "open":
                raise InputError(
                    f"草稿 #{draft_id} 已{self._draft_status_text(draft['status'])}，不能提交。"
                    "草稿绑定的输入版本已被更正（全量数据或配额变化），请基于当前版本重新上传复核 CSV。"
                )
            version = conn.execute(
                "SELECT * FROM input_versions WHERE id=?",
                (draft["source_version_id"],),
            ).fetchone()
            if version["status"] != "active":
                # 双保险：正常情况下上面的惰性标记已把草稿置 stale
                conn.execute(
                    "UPDATE review_drafts SET status='stale' WHERE id=?", (draft_id,)
                )
                conn.execute("COMMIT")
                raise InputError(
                    "草稿绑定的输入版本已被更正，提交已拒绝；原版本与下载状态保持不变，"
                    "请基于当前版本重新上传复核 CSV。"
                )
            classes = version["class_names"].split(",")
            class_set = set(classes)

            # 先把本次提交携带的裁决写入（仅分歧项、仅当前类别）
            items = list(
                conn.execute(
                    "SELECT * FROM review_items WHERE draft_id=? ORDER BY image_no",
                    (draft_id,),
                ).fetchall()
            )
            confirmed: dict[str, str] = {}
            unresolved: list[str] = []
            for item in items:
                label = item["confirmed_label"]
                if not item["is_agreed"] and decisions and item["image_no"] in decisions:
                    picked = decisions[item["image_no"]]
                    if picked != "":
                        if picked not in class_set:
                            raise InputError(
                                f"图像 {item['image_no']} 的裁决标签「{picked}」不在当前类别集合内"
                            )
                        label = picked
                        conn.execute(
                            "UPDATE review_items SET confirmed_label=? WHERE id=?",
                            (picked, item["id"]),
                        )
                if not label:
                    unresolved.append(item["image_no"])
                else:
                    if label not in class_set:
                        raise InputError(
                            f"图像 {item['image_no']} 的确认标签「{label}」不在当前类别集合内"
                        )
                    confirmed[item["image_no"]] = label
            if unresolved:
                raise InputError(
                    f"尚有 {len(unresolved)} 个分歧项未经质控裁决，不能提交："
                    + "、".join(unresolved[:8]) + ("…" if len(unresolved) > 8 else "")
                )

            # 类别集合必须不变：确认标签用到的类别集合与原版本完全一致
            if set(confirmed.values()) != class_set:
                used = sorted(set(confirmed.values()))
                raise InputError(
                    "提交后类别集合必须不变。确认标签实际类别集合为"
                    f"（{'、'.join(used)}），与当前版本类别集合（{'、'.join(classes)}）不一致。"
                )

            # 沿用患者、重复关系与配额，仅替换类别标签，重建全量图像 CSV
            old_images = list(
                conn.execute(
                    "SELECT image_no, patient_no FROM images WHERE version_id=? ORDER BY id",
                    (version["id"],),
                ).fetchall()
            )
            dup_edges = [
                (r["image_a"], r["image_b"])
                for r in conn.execute(
                    "SELECT image_a, image_b FROM dup_edges WHERE version_id=? ORDER BY id",
                    (version["id"],),
                ).fetchall()
            ]
            quota = json.loads(version["quotas"])
            new_image_csv = self._build_image_csv(old_images, confirmed)
            new_dup_csv = version["dup_csv"] or ""

            # 复用解析/校验链路，保证新版本结构合法（分量、类别上限等）
            parsed_images = parse_images_csv(new_image_csv)
            parsed_dups = parse_dups_csv(new_dup_csv)
            _, new_classes = validate_and_build(parsed_images, parsed_dups)
            if set(new_classes) != class_set:
                raise InputError(
                    "提交后类别集合必须不变；新标签解析出的类别集合与原版本不一致，已中止提交。"
                )
            new_note = (
                f"标签复核裁决生成（草稿 #{draft_id}，基于 v{version['id']}）于 "
                + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            new_vid = self._insert_version(
                conn, new_image_csv, new_dup_csv, parsed_images, parsed_dups,
                new_classes, quota, note=new_note,
            )
            # 旧版本同事务转历史 → 旧清单下载立即停供
            conn.execute(
                "UPDATE input_versions SET status='superseded' WHERE id<>?", (new_vid,)
            )
            conn.execute(
                "UPDATE review_drafts SET status='submitted', new_version_id=?, "
                "submitted_at=datetime('now'), fail_reason=NULL WHERE id=?",
                (new_vid, draft_id),
            )
            conn.execute("COMMIT")
            return new_vid
        except Exception as exc:
            conn.execute("ROLLBACK")
            # 回滚后把失败原因记到草稿（独立事务），不改变原版本/下载状态
            try:
                conn2 = get_db(self.db_path)
                try:
                    conn2.execute(
                        "UPDATE review_drafts SET fail_reason=? WHERE id=?",
                        (str(exc), draft_id),
                    )
                finally:
                    conn2.close()
            except Exception:
                pass
            raise
        finally:
            conn.close()

    @staticmethod
    def _draft_status_text(status: str) -> str:
        return {
            "open": "进行中",
            "submitted": "提交",
            "discarded": "作废",
            "stale": "失效",
        }.get(status, status)

    @staticmethod
    def _build_image_csv(old_images, confirmed: dict[str, str]) -> str:
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["图像号", "患者号", "类别"])
        for r in old_images:
            writer.writerow([r["image_no"], r["patient_no"], confirmed[r["image_no"]]])
        return buf.getvalue()
