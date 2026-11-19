"""Google API tools: Gmail and Classroom."""
from __future__ import annotations

import logging
from typing import Annotated

from buddy.tools.registry import tool

logger = logging.getLogger(__name__)

_config = None


def init_google_tools(config) -> None:
    global _config
    _config = config


def _get_credentials():
    """Load Google OAuth credentials from token file."""
    try:
        from google.oauth2.credentials import Credentials

        if _config is None:
            return None
        token_path = _config.google_token_file
        if not token_path.exists():
            return None
        creds = Credentials.from_authorized_user_file(
            str(token_path), _config.google.scopes,
        )
        if creds and creds.expired and creds.refresh_token:
            from google.auth.transport.requests import Request
            creds.refresh(Request())
            with open(token_path, "w") as f:
                f.write(creds.to_json())
        return creds
    except Exception as e:
        logger.warning("Failed to load Google credentials: %s", e)
        return None


def _get_gmail_service():
    creds = _get_credentials()
    if not creds:
        return None
    from googleapiclient.discovery import build
    return build("gmail", "v1", credentials=creds)


def _get_classroom_service():
    creds = _get_credentials()
    if not creds:
        return None
    from googleapiclient.discovery import build
    return build("classroom", "v1", credentials=creds)


@tool
def check_email(
    max_results: Annotated[int, "Maximum number of emails to return"] = 5,
) -> str:
    """Check recent unread emails from Gmail."""
    service = _get_gmail_service()
    if not service:
        return "Error: Gmail not configured. Run 'buddy auth-google' first."

    try:
        results = service.users().messages().list(
            userId="me",
            q="is:unread",
            maxResults=max_results,
        ).execute()

        messages = results.get("messages", [])
        if not messages:
            return "No unread emails."

        lines = [f"You have {len(messages)} unread email(s):"]
        for msg_meta in messages:
            msg = service.users().messages().get(
                userId="me",
                id=msg_meta["id"],
                format="metadata",
                metadataHeaders=["Subject", "From"],
            ).execute()

            headers = {h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])}
            subject = headers.get("Subject", "(no subject)")
            sender = headers.get("From", "(unknown)")
            snippet = msg.get("snippet", "")[:100]
            lines.append(f"  - From: {sender}")
            lines.append(f"    Subject: {subject}")
            if snippet:
                lines.append(f"    Preview: {snippet}")

        return "\n".join(lines)
    except Exception as e:
        return f"Error checking email: {e}"


@tool
def classroom_assignments(
    status: Annotated[str, "Filter: pending, overdue, or all"] = "pending",
) -> str:
    """List assignments from Google Classroom."""
    service = _get_classroom_service()
    if not service:
        return "Error: Classroom not configured. Run 'buddy auth-google' first."

    try:
        # Get courses
        courses_result = service.courses().list(
            courseStates=["ACTIVE"],
        ).execute()
        courses = courses_result.get("courses", [])

        if not courses:
            return "No active courses found."

        import datetime

        now = datetime.datetime.now(datetime.timezone.utc)
        all_assignments: list[dict] = []

        for course in courses:
            course_id = course["id"]
            course_name = course.get("name", "Unknown")

            try:
                work_result = service.courses().courseWork().list(
                    courseId=course_id,
                    orderBy="dueDate desc",
                    pageSize=20,
                ).execute()

                for work in work_result.get("courseWork", []):
                    due_date = work.get("dueDate")
                    due_time = work.get("dueTime", {})
                    due_dt = None

                    if due_date:
                        y = due_date.get("year", 2024)
                        m = due_date.get("month", 1)
                        d = due_date.get("day", 1)
                        h = due_time.get("hours", 23)
                        mi = due_time.get("minutes", 59)
                        due_dt = datetime.datetime(y, m, d, h, mi, tzinfo=datetime.timezone.utc)

                    # Get submission status
                    try:
                        subs = service.courses().courseWork().studentSubmissions().list(
                            courseId=course_id,
                            courseWorkId=work["id"],
                            userId="me",
                        ).execute()
                        submissions = subs.get("studentSubmissions", [])
                        turned_in = any(
                            s.get("state") in ("TURNED_IN", "RETURNED")
                            for s in submissions
                        )
                    except Exception:
                        turned_in = False

                    if turned_in:
                        continue

                    is_overdue = due_dt is not None and due_dt < now
                    is_pending = not turned_in

                    if status == "overdue" and not is_overdue:
                        continue
                    if status == "pending" and not is_pending:
                        continue

                    all_assignments.append({
                        "course": course_name,
                        "title": work.get("title", "?"),
                        "due": due_dt.strftime("%Y-%m-%d %H:%M") if due_dt else "No due date",
                        "overdue": is_overdue,
                    })
            except Exception as e:
                logger.warning("Failed to get coursework for %s: %s", course_name, e)

        if not all_assignments:
            return f"No {status} assignments found."

        # Sort by due date
        all_assignments.sort(key=lambda a: a["due"])

        lines = [f"{status.capitalize()} assignments:"]
        for a in all_assignments:
            flag = " ⚠️ OVERDUE" if a["overdue"] else ""
            lines.append(f"  - [{a['course']}] {a['title']} (due: {a['due']}){flag}")

        return "\n".join(lines)
    except Exception as e:
        return f"Error getting assignments: {e}"
