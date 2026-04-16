from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.attachment import Attachment
from app.models.audit_log import AuditLog
from app.models.release import Release
from app.models.task import Task


def compute_project_stats(
    db: Session,
    project_ids: list[str],
    *,
    audit_days: int,
) -> tuple[datetime, dict[str, dict]]:
    """Compute common project stats used by both /admin/stats and /projects/{id}/stats."""

    now = datetime.now(UTC)
    today = now.date()

    unique_pids: list[str] = []
    seen: set[str] = set()
    for pid in project_ids:
        if pid in seen:
            continue
        seen.add(pid)
        unique_pids.append(pid)

    if not unique_pids:
        return now, {}

    tasks_by_status: dict[str, dict[str, int]] = defaultdict(dict)
    for pid, status_name, cnt in db.execute(
        select(Task.project_id, Task.status, func.count(Task.id))
        .where(Task.project_id.in_(unique_pids))
        .group_by(Task.project_id, Task.status)
    ).all():
        tasks_by_status[pid][status_name] = int(cnt)

    tasks_by_priority: dict[str, dict[str, int]] = defaultdict(dict)
    for pid, prio, cnt in db.execute(
        select(Task.project_id, Task.priority, func.count(Task.id))
        .where(Task.project_id.in_(unique_pids))
        .group_by(Task.project_id, Task.priority)
    ).all():
        tasks_by_priority[pid][prio] = int(cnt)

    backlog_totals = dict(
        (pid, int(cnt))
        for pid, cnt in db.execute(
            select(Task.project_id, func.count(Task.id))
            .where(Task.project_id.in_(unique_pids), Task.release_id.is_(None))
            .group_by(Task.project_id)
        ).all()
    )

    overdue_totals = dict(
        (pid, int(cnt))
        for pid, cnt in db.execute(
            select(Task.project_id, func.count(Task.id))
            .where(
                Task.project_id.in_(unique_pids),
                Task.due_date.is_not(None),
                Task.due_date < today,
                Task.status != "Done",
            )
            .group_by(Task.project_id)
        ).all()
    )

    missing_due_date_totals = dict(
        (pid, int(cnt))
        for pid, cnt in db.execute(
            select(Task.project_id, func.count(Task.id))
            .where(Task.project_id.in_(unique_pids), Task.due_date.is_(None))
            .group_by(Task.project_id)
        ).all()
    )

    attachment_rows = db.execute(
        select(Task.project_id, func.count(Attachment.id), func.coalesce(func.sum(Attachment.size_bytes), 0))
        .join(Task, Task.id == Attachment.task_id)
        .where(Task.project_id.in_(unique_pids))
        .group_by(Task.project_id)
    ).all()
    attachments_totals: dict[str, tuple[int, int]] = {pid: (int(cnt), int(sz)) for pid, cnt, sz in attachment_rows}

    audit_totals = dict(
        (pid, int(cnt))
        for pid, cnt in db.execute(
            select(AuditLog.project_id, func.count(AuditLog.id))
            .where(AuditLog.project_id.is_not(None), AuditLog.project_id.in_(unique_pids))
            .group_by(AuditLog.project_id)
        ).all()
    )

    since_7d = now - timedelta(days=7)
    audit_last_7d = dict(
        (pid, int(cnt))
        for pid, cnt in db.execute(
            select(AuditLog.project_id, func.count(AuditLog.id))
            .where(
                AuditLog.project_id.is_not(None),
                AuditLog.project_id.in_(unique_pids),
                AuditLog.created_at >= since_7d,
            )
            .group_by(AuditLog.project_id)
        ).all()
    )

    since_window = now - timedelta(days=audit_days)
    audit_last_window = dict(
        (pid, int(cnt))
        for pid, cnt in db.execute(
            select(AuditLog.project_id, func.count(AuditLog.id))
            .where(
                AuditLog.project_id.is_not(None),
                AuditLog.project_id.in_(unique_pids),
                AuditLog.created_at >= since_window,
            )
            .group_by(AuditLog.project_id)
        ).all()
    )

    # Releases (to include 0-count releases in the breakdown)
    releases_by_project: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for rid, pid, name in db.execute(
        select(Release.id, Release.project_id, Release.name)
        .where(Release.project_id.in_(unique_pids))
        .order_by(Release.name)
    ).all():
        releases_by_project[pid].append((rid, name))

    release_task_totals: dict[tuple[str, str | None], int] = {}
    for pid, rid, cnt in db.execute(
        select(Task.project_id, Task.release_id, func.count(Task.id))
        .where(Task.project_id.in_(unique_pids))
        .group_by(Task.project_id, Task.release_id)
    ).all():
        release_task_totals[(pid, rid)] = int(cnt)

    out: dict[str, dict] = {}
    for pid in unique_pids:
        by_status = dict(tasks_by_status.get(pid, {}))
        by_priority = dict(tasks_by_priority.get(pid, {}))

        backlog_total = int(backlog_totals.get(pid, 0))
        tasks_total = int(sum(by_status.values()))

        tasks_by_release: list[dict] = [
            {"release_id": None, "release_name": "Backlog", "tasks_total": backlog_total}
        ]
        for rid, name in releases_by_project.get(pid, []):
            tasks_by_release.append(
                {
                    "release_id": rid,
                    "release_name": name,
                    "tasks_total": int(release_task_totals.get((pid, rid), 0)),
                }
            )

        att_cnt, att_sz = attachments_totals.get(pid, (0, 0))

        out[pid] = {
            "tasks_total": tasks_total,
            "backlog_total": backlog_total,
            "overdue_total": int(overdue_totals.get(pid, 0)),
            "missing_due_date_total": int(missing_due_date_totals.get(pid, 0)),
            "tasks_by_status": by_status,
            "tasks_by_priority": by_priority,
            "tasks_by_release": tasks_by_release,
            "attachments_total": int(att_cnt),
            "attachments_bytes": int(att_sz),
            "audit_total": int(audit_totals.get(pid, 0)),
            "audit_last_7d": int(audit_last_7d.get(pid, 0)),
            "audit_last_window": int(audit_last_window.get(pid, 0)),
        }

    return now, out
