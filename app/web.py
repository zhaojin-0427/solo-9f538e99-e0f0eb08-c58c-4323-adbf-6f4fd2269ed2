"""Flask 路由：浏览器更正、划分、查看原因、下载两侧清单。"""
from __future__ import annotations

import csv
import io
import json
import os

from flask import (
    Flask,
    Response,
    abort,
    flash,
    redirect,
    render_template,
    request,
    url_for,
)

from .db import VALID, get_db
from .parser import InputError
from .service import Service


def create_app(db_path: str | None = None) -> Flask:
    app = Flask(__name__)
    app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-me")
    db_path = db_path or os.environ.get("DB_PATH", "/data/pathology.db")
    synchronous = os.environ.get("PYX_SYNCHRONOUS", "0") == "1"
    solve_delay = float(os.environ.get("PYX_SOLVE_DELAY", "0"))
    svc = Service(db_path, synchronous=synchronous, solve_delay=solve_delay)
    app.extensions["service"] = svc

    # ------------------------------------------------------------ 首页/仪表盘
    @app.route("/")
    def index():
        version = svc.current_version()
        manifest = svc.current_manifest()
        entries = svc.get_manifest_entries(manifest["id"]) if manifest else []
        jobs = svc.list_jobs(limit=8)
        classes = version["class_names"].split(",") if version else []
        quotas = json.loads(version["quotas"]) if version else {}
        val_counts: dict[str, int] = {}
        train_counts: dict[str, int] = {}
        for e in entries:
            bucket = val_counts if e["side"] == VALID else train_counts
            bucket[e["class_name"]] = bucket.get(e["class_name"], 0) + 1
        return render_template(
            "index.html",
            version=version,
            manifest=manifest,
            entries=entries,
            jobs=jobs,
            classes=classes,
            quotas=quotas,
            val_counts=val_counts,
            train_counts=train_counts,
        )

    # ------------------------------------------------------------ 上传/更正
    @app.route("/upload", methods=["POST"])
    def upload():
        image_file = request.files.get("image_csv")
        dup_file = request.files.get("dup_csv")
        if not image_file or not image_file.filename:
            flash("请选择全量图像 CSV 文件", "error")
            return redirect(url_for("index"))
        # 编码兜底：utf-8-sig → gbk
        image_csv = _decode(image_file.read())
        dup_csv = ""
        if dup_file and dup_file.filename:
            dup_csv = _decode(dup_file.read())
        # 配额字段名由前端按类别动态生成，统一使用 quota::<类别> 前缀
        quota_raw = {
            key[len("quota::"):]: val
            for key, val in request.form.items()
            if key.startswith("quota::")
        }
        try:
            vid, classes = svc.create_version(image_csv, dup_csv, quota_raw)
        except InputError as exc:
            flash(f"输入校验失败，未保存任何内容：{exc}", "error")
            return redirect(url_for("index"))
        flash(
            f"新版本 v{vid} 已生效（类别：{'、'.join(classes)}）。"
            "旧版本及其清单已转为历史，当前下载已停供，重新划分成功后恢复。",
            "success",
        )
        return redirect(url_for("index"))

    # ------------------------------------------------------------ 发起划分
    @app.route("/partition", methods=["POST"])
    def partition():
        try:
            job_id = svc.start_partition()
        except InputError as exc:
            flash(str(exc), "error")
            return redirect(url_for("index"))
        return redirect(url_for("job_detail", job_id=job_id))

    @app.route("/jobs/<int:job_id>")
    def job_detail(job_id: int):
        job = svc.get_job(job_id)
        if job is None:
            abort(404)
        version = svc.get_version(job["version_id"])
        manifest = None
        entries = []
        if job["manifest_id"]:
            conn = get_db(svc.db_path)
            try:
                manifest = conn.execute(
                    "SELECT * FROM manifests WHERE id=?", (job["manifest_id"],)
                ).fetchone()
            finally:
                conn.close()
            entries = svc.get_manifest_entries(job["manifest_id"])
        return render_template(
            "job.html", job=job, version=version, manifest=manifest, entries=entries
        )

    @app.route("/jobs/<int:job_id>/status")
    def job_status(job_id: int):
        job = svc.get_job(job_id)
        if job is None:
            abort(404)
        return {
            "id": job["id"],
            "status": job["status"],
            "fail_stage": job["fail_stage"],
            "reason": job["reason"],
            "manifest_id": job["manifest_id"],
            "done": job["status"] in ("succeeded", "failed"),
        }

    # ------------------------------------------------------------ 历史
    @app.route("/history")
    def history():
        versions = svc.list_versions()
        jobs = svc.list_jobs(limit=100)
        manifests = svc.list_manifests()
        return render_template(
            "history.html", versions=versions, jobs=jobs, manifests=manifests
        )

    @app.route("/versions/<int:version_id>")
    def version_detail(version_id: int):
        version = svc.get_version(version_id)
        if version is None:
            abort(404)
        conn = get_db(svc.db_path)
        try:
            images = conn.execute(
                "SELECT * FROM images WHERE version_id=? ORDER BY image_no",
                (version_id,),
            ).fetchall()
            edges = conn.execute(
                "SELECT * FROM dup_edges WHERE version_id=? ORDER BY id", (version_id,)
            ).fetchall()
            manifest = conn.execute(
                "SELECT * FROM manifests WHERE version_id=? AND status='published'",
                (version_id,),
            ).fetchone()
        finally:
            conn.close()
        return render_template(
            "version.html",
            version=version,
            images=images,
            edges=edges,
            manifest=manifest,
            quotas=json.loads(version["quotas"]),
        )

    # ------------------------------------------------------------ 侧别登记
    @app.route("/registry")
    def registry():
        rows = svc.registry_snapshot()
        return render_template("registry.html", rows=rows)

    # ------------------------------------------------------------ 标签复核
    @app.route("/reviews")
    def reviews():
        version = svc.current_version()
        drafts = svc.list_drafts()
        classes = version["class_names"].split(",") if version else []
        return render_template(
            "reviews.html", version=version, drafts=drafts, classes=classes
        )

    @app.route("/reviews/upload", methods=["POST"])
    def review_upload():
        file_a = request.files.get("review_a_csv")
        file_b = request.files.get("review_b_csv")
        if not file_a or not file_a.filename or not file_b or not file_b.filename:
            flash("请同时选择两位标注员的复核 CSV 文件", "error")
            return redirect(url_for("reviews"))
        csv_a = _decode(file_a.read())
        csv_b = _decode(file_b.read())
        try:
            draft_id = svc.create_review_draft(csv_a, csv_b)
        except InputError as exc:
            flash(f"复核 CSV 校验失败，未保存草稿：{exc}", "error")
            return redirect(url_for("reviews"))
        draft = svc.get_draft(draft_id)
        flash(
            f"复核草稿 #{draft_id} 已生成：一致 {draft['agreed_count']} 张、"
            f"分歧 {draft['disagreed_count']} 张。一致标签已直接进入草稿，"
            "请逐项裁决分歧后提交。",
            "success",
        )
        return redirect(url_for("review_detail", draft_id=draft_id))

    @app.route("/reviews/<int:draft_id>")
    def review_detail(draft_id: int):
        draft = svc.get_draft(draft_id)
        if draft is None:
            abort(404)
        version = svc.get_version(draft["source_version_id"])
        items = svc.get_draft_items(draft_id)
        classes = version["class_names"].split(",")
        # 当前版本（可能已不是草稿绑定版本）
        current = svc.current_version()
        unresolved = [
            i for i in items if not i["is_agreed"] and not i["confirmed_label"]
        ]
        return render_template(
            "review_detail.html",
            draft=draft,
            version=version,
            current=current,
            items=items,
            classes=classes,
            unresolved=unresolved,
        )

    @app.route("/reviews/<int:draft_id>/save", methods=["POST"])
    def review_save(draft_id: int):
        decisions = _extract_decisions(request.form)
        try:
            svc.save_adjudication(draft_id, decisions)
        except InputError as exc:
            flash(f"裁决暂存失败：{exc}", "error")
        else:
            flash("裁决已暂存（尚未生成新版本，可继续调整后提交）。", "success")
        return redirect(url_for("review_detail", draft_id=draft_id))

    @app.route("/reviews/<int:draft_id>/submit", methods=["POST"])
    def review_submit(draft_id: int):
        decisions = _extract_decisions(request.form)
        try:
            new_vid = svc.submit_review_draft(draft_id, decisions)
        except InputError as exc:
            flash(f"提交失败，原版本及下载状态保持不变：{exc}", "error")
            return redirect(url_for("review_detail", draft_id=draft_id))
        flash(
            f"复核草稿 #{draft_id} 已提交，确认标签生成新版本 v{new_vid}："
            "患者归属、重复关系与配额保持不变，类别集合不变；旧版本及其清单已转为历史、"
            "下载停供。请在首页对新版本重新发起划分，成功后下载恢复（仍遵守永久侧别与精确配额）。",
            "success",
        )
        return redirect(url_for("index"))

    @app.route("/reviews/<int:draft_id>/discard", methods=["POST"])
    def review_discard(draft_id: int):
        try:
            svc.discard_draft(draft_id)
        except InputError as exc:
            flash(f"作废失败：{exc}", "error")
        else:
            flash(f"复核草稿 #{draft_id} 已作废，可重新上传复核 CSV。", "success")
        return redirect(url_for("reviews"))

    # ------------------------------------------------------------ 下载
    @app.route("/manifests/<int:manifest_id>/download/<side>")
    def download(manifest_id: int, side):
        if side not in ("train", "val"):
            abort(404)
        conn = get_db(svc.db_path)
        try:
            manifest = conn.execute(
                "SELECT m.*, v.status AS version_status FROM manifests m "
                "JOIN input_versions v ON v.id=m.version_id WHERE m.id=?",
                (manifest_id,),
            ).fetchone()
        finally:
            conn.close()
        if manifest is None or manifest["status"] != "published":
            abort(404)
        # 更正后旧清单仅作历史：非当前版本一律停供下载
        if manifest["version_status"] != "active":
            flash("该清单所属输入版本已被更正，仅作历史保留，当前下载停供。", "error")
            return redirect(url_for("history"))
        entries = [e for e in svc.get_manifest_entries(manifest_id) if e["side"] == side]
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["图像号", "患者号", "类别", "侧别", "连通分量"])
        for e in entries:
            writer.writerow(
                [e["image_no"], e["patient_no"], e["class_name"], e["side"], e["component_id"]]
            )
        filename = f"manifest_v{manifest['version_id']}_{side}.csv"
        return Response(
            buf.getvalue(),
            mimetype="text/csv; charset=utf-8",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    return app


def _decode(raw: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _extract_decisions(form) -> dict[str, str]:
    """从表单提取分歧项裁决：字段名 decision::<图像号>。"""
    return {
        key[len("decision::"):]: val
        for key, val in form.items()
        if key.startswith("decision::") and val != ""
    }
