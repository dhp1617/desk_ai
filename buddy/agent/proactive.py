"""Proactive schedule, dining, and email announcements."""
from __future__ import annotations

import asyncio
import datetime
import logging
import time
from typing import TYPE_CHECKING, Any

from buddy.config import Config
from buddy.memory.store import MemoryStore

if TYPE_CHECKING:
    from buddy.agent.state import BuddyAgent

logger = logging.getLogger(__name__)


class ProactiveManager:
    """Manages proactive notices (classes, meals, emails) with dedup and rate limits."""

    def __init__(self, config: Config, memory: MemoryStore):
        self.config = config
        self.memory = memory
        self.agent: BuddyAgent | None = None
        self._seen_notices: set[str] = set()
        self._last_notice_time: float = 0.0
        self._first_email_poll: bool = True
        self._known_email_ids: set[str] = set()
        self._running = False
        self._task: asyncio.Task | None = None

    def set_agent(self, agent: BuddyAgent) -> None:
        self.agent = agent

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _loop(self) -> None:
        while self._running:
            try:
                await self.check_all()
            except Exception as e:
                logger.error("Error in proactive check: %s", e)
            await asyncio.sleep(30.0)

    async def check_all(self) -> None:
        now = datetime.datetime.now()
        await self.check_classes(now)
        await self.check_meals(now)
        await self.check_emails()

    def can_dispatch_notice(self) -> bool:
        if not self.agent or not self.agent.is_idle():
            return False
        if (time.time() - self._last_notice_time) < self.config.notice.min_interval_s:
            return False
        return True

    async def dispatch_notice(self, key: str, text: str) -> bool:
        """Deliver notice if allowed; deduplicate by key and rate-limit."""
        if key in self._seen_notices:
            return False
        if not self.can_dispatch_notice():
            return False

        assert self.agent is not None
        self._seen_notices.add(key)
        self._last_notice_time = time.time()
        logger.info("Dispatching proactive notice [%s]: %s", key, text)
        await self.agent.speak_notice(text)
        return True

    async def check_classes(self, now: datetime.datetime) -> None:
        timetable = self.memory.get_timetable()
        if not timetable:
            return

        day_name = now.strftime("%A")
        classes = timetable.get(day_name, [])
        if not isinstance(classes, list):
            return

        lead_min = self.config.notice.class_remind_min
        target_delta = datetime.timedelta(minutes=lead_min)

        for item in classes:
            if not isinstance(item, dict):
                continue
            time_str = str(item.get("time", ""))
            try:
                parts = time_str.split(":")
                if len(parts) != 2:
                    continue
                ch, cm = int(parts[0]), int(parts[1])
                class_dt = now.replace(hour=ch, minute=cm, second=0, microsecond=0)
                diff = class_dt - now
                if datetime.timedelta(seconds=0) <= diff <= target_delta:
                    cname = item.get("class", item.get("name", "Class"))
                    room = item.get("room", "")
                    key = f"class:{day_name}:{time_str}:{cname}"
                    msg = f"Reminder: Your {cname} class starts in {lead_min} minutes at {time_str}"
                    if room:
                        msg += f" in room {room}."
                    else:
                        msg += "."
                    await self.dispatch_notice(key, msg)
            except Exception as e:
                logger.debug("Error checking class item %s: %s", item, e)

    async def check_meals(self, now: datetime.datetime) -> None:
        mess = self.memory.get_mess_menu()
        if not mess:
            return

        day_name = now.strftime("%A")
        day_menu = mess.get(day_name, {})
        if not isinstance(day_menu, dict):
            return

        meal_times = {
            "breakfast": "08:30",
            "lunch": "12:30",
            "snacks": "17:00",
            "dinner": "19:30",
        }

        lead_min = self.config.notice.meal_remind_min
        target_delta = datetime.timedelta(minutes=lead_min)

        for meal, time_str in meal_times.items():
            menu_item = day_menu.get(meal)
            if not menu_item:
                continue
            try:
                parts = time_str.split(":")
                mh, mm = int(parts[0]), int(parts[1])
                meal_dt = now.replace(hour=mh, minute=mm, second=0, microsecond=0)
                diff = meal_dt - now
                if datetime.timedelta(seconds=0) <= diff <= target_delta:
                    key = f"meal:{day_name}:{meal}:{time_str}"
                    msg = f"Reminder: {meal.capitalize()} starts in {lead_min} minutes. Today's menu is: {menu_item}."
                    await self.dispatch_notice(key, msg)
            except Exception as e:
                logger.debug("Error checking meal %s: %s", meal, e)

    async def check_emails(self) -> None:
        try:
            from buddy.tools.google_tools import _get_gmail_service
            service = _get_gmail_service()
            if not service:
                return

            loop = asyncio.get_running_loop()
            res = await loop.run_in_executor(
                None,
                lambda: service.users().messages().list(userId="me", q="is:unread", maxResults=5).execute()
            )
            messages = res.get("messages", [])
            current_ids = {m["id"] for m in messages}

            if self._first_email_poll:
                self._known_email_ids = current_ids
                self._first_email_poll = False
                return

            new_ids = current_ids - self._known_email_ids
            if new_ids:
                count = len(new_ids)
                key = f"email:{sorted(new_ids)[0]}"
                self._known_email_ids.update(new_ids)
                text = f"You have {count} new unread email{'s' if count > 1 else ''}."
                await self.dispatch_notice(key, text)
        except Exception as e:
            logger.debug("Proactive email check skipped or failed: %s", e)
