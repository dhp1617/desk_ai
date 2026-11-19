"""GTK4 floating orb overlay with Wayland layer-shell support."""
from __future__ import annotations

import logging
import math
import sys
from typing import Callable

logger = logging.getLogger(__name__)


class BuddyOrb:
    """GTK4 desktop floating orb reflecting assistant states and audio reactivity."""

    def __init__(
        self,
        size: int = 48,
        on_left_click: Callable[[], None] | None = None,
        on_right_click: Callable[[], None] | None = None,
    ):
        self.size = size
        self.on_left_click = on_left_click
        self.on_right_click = on_right_click

        self._state_name = "WAKE_LISTEN"
        self._level = 0.0
        self._rotation_angle = 0.0

        self._app = None
        self._window = None
        self._darea = None
        self._glib = None

    def set_state(self, state_name: str, level: float = 0.0) -> None:
        """Thread-safe update called by the agent thread."""
        self._state_name = state_name
        self._level = max(0.0, min(1.0, level))
        if self._glib and self._darea:
            self._glib.idle_add(self._darea.queue_draw)

    def run(self) -> None:
        """Launch the GTK4 application main loop."""
        try:
            import gi
            gi.require_version("Gtk", "4.0")
            from gi.repository import GLib, Gtk, Gdk
            self._glib = GLib
        except Exception as e:
            logger.error("Failed to load GTK4: %s. Orb GUI disabled.", e)
            return

        app = Gtk.Application(application_id="ai.desk.buddy.orb")

        def on_activate(application):
            window = Gtk.ApplicationWindow(application=application)
            window.set_default_size(self.size, self.size)
            window.set_title("Buddy")

            # Try gtk4-layer-shell for Wayland overlay
            try:
                gi.require_version("Gtk4LayerShell", "1.0")
                from gi.repository import Gtk4LayerShell
                if Gtk4LayerShell.is_supported():
                    Gtk4LayerShell.init_for_window(window)
                    Gtk4LayerShell.set_layer(window, Gtk4LayerShell.Layer.OVERLAY)
                    Gtk4LayerShell.set_anchor(window, Gtk4LayerShell.Edge.BOTTOM, True)
                    Gtk4LayerShell.set_anchor(window, Gtk4LayerShell.Edge.RIGHT, True)
                    Gtk4LayerShell.set_margin(window, Gtk4LayerShell.Edge.BOTTOM, 24)
                    Gtk4LayerShell.set_margin(window, Gtk4LayerShell.Edge.RIGHT, 24)
            except Exception:
                pass

            darea = Gtk.DrawingArea()
            darea.set_content_width(self.size)
            darea.set_content_height(self.size)
            darea.set_draw_func(self._draw_func)

            # Click gestures
            click_gesture = Gtk.GestureClick.new()
            click_gesture.set_button(0)  # Listen to all mouse buttons
            click_gesture.connect("pressed", self._on_clicked)
            darea.add_controller(click_gesture)

            window.set_child(darea)
            self._window = window
            self._darea = darea

            # Animation timer
            GLib.timeout_add(40, self._on_tick)
            window.present()

        app.connect("activate", on_activate)
        self._app = app
        app.run([])

    def _on_tick(self) -> bool:
        """Animation update loop."""
        self._rotation_angle = (self._rotation_angle + 0.1) % (2 * math.pi)
        if self._darea:
            self._darea.queue_draw()
        return True

    def _on_clicked(self, gesture, n_press, x, y):
        button = gesture.get_current_button()
        if button == 1:  # Left-click
            if self.on_left_click:
                self.on_left_click()
        elif button == 3:  # Right-click
            if self.on_right_click:
                self.on_right_click()

    def _draw_func(self, area, cr, width, height):
        cx = width / 2.0
        cy = height / 2.0
        radius = (min(width, height) / 2.0) - 4.0

        state = self._state_name

        if state == "MUTED":
            # Gray muted circle
            cr.set_source_rgba(0.5, 0.5, 0.5, 0.8)
            cr.arc(cx, cy, radius, 0, 2 * math.pi)
            cr.fill()
        elif state == "ERROR":
            # Red error circle
            cr.set_source_rgba(0.9, 0.2, 0.2, 0.9)
            cr.arc(cx, cy, radius, 0, 2 * math.pi)
            cr.fill()
        elif state == "CAPTURE":
            # Green pulsing with audio level
            scale = 1.0 + (self._level * 0.2)
            cr.set_source_rgba(0.1, 0.8, 0.3, 0.85)
            cr.arc(cx, cy, radius * scale, 0, 2 * math.pi)
            cr.fill()
        elif state == "SPEAKING":
            # Blue amplitude-reactive orb
            scale = 1.0 + (self._level * 0.3)
            cr.set_source_rgba(0.1, 0.5, 0.95, 0.9)
            cr.arc(cx, cy, radius * scale, 0, 2 * math.pi)
            cr.fill()
        elif state == "THINKING":
            # Blue core with spinning arc
            cr.set_source_rgba(0.2, 0.4, 0.8, 0.6)
            cr.arc(cx, cy, radius * 0.8, 0, 2 * math.pi)
            cr.fill()
            # Spinning arc
            cr.set_source_rgba(0.2, 0.7, 1.0, 0.9)
            cr.set_line_width(3.0)
            cr.arc(cx, cy, radius, self._rotation_angle, self._rotation_angle + 1.5)
            cr.stroke()
        elif state == "ALWAYS_LISTEN":
            # Bright pulsating blue
            cr.set_source_rgba(0.0, 0.6, 1.0, 0.95)
            cr.arc(cx, cy, radius, 0, 2 * math.pi)
            cr.fill()
        elif state == "FOLLOW_UP":
            # Amber / Warm Gold pulsating circle (User can speak without wake word!)
            pulse = 1.0 + 0.1 * math.sin(self._rotation_angle * 2)
            cr.set_source_rgba(1.0, 0.75, 0.1, 0.85)
            cr.arc(cx, cy, radius * pulse, 0, 2 * math.pi)
            cr.fill()
        else:  # WAKE_LISTEN / IDLE
            # Solid calm blue (Sleeping / Waiting for 'Buddy')
            cr.set_source_rgba(0.1, 0.4, 0.8, 0.75)
            cr.arc(cx, cy, radius, 0, 2 * math.pi)
            cr.fill()
