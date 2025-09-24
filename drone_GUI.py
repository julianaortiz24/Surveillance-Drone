import cv2
import os
import shutil
import zipfile
import tkinter as tk
from tkinter import Label, Button, Toplevel, Text, Scrollbar, END, Frame, Canvas, filedialog, simpledialog, messagebox
from PIL import Image, ImageTk
from datetime import datetime
import geocoder
import threading
import urllib.request
from io import BytesIO
import json
import winsound
import time

from recognize_juliana_2_6_25 import (
    setup_folders,
    load_known_faces,
    recognize_faces_in_frame
)
from route_planner import launch_route_planner

SETTINGS_FILE = "gui_settings.json"


class FaceRecognitionApp:
    def __init__(self, root):
        self.last_engagement_time = {}
        self.engagement_active = False 
        self.root = root
        self.root.title("Juliana Face Recognition")
        self.root.geometry("900x740")

        self.theme = self.load_theme()
        
        # Canvas + scrollable content  (store the canvas on self so we can theme it later)
        self.canvas = Canvas(self.root, highlightthickness=0, bg=self.get_bg())
        self.scrollable_frame = Frame(self.canvas, bg=self.get_bg())

        # Create a window for the frame and keep its TOP edge fixed, but center horizontally
        self._canvas_window_id = self.canvas.create_window(
            (0, 0), window=self.scrollable_frame, anchor="n"
        )

        self.canvas.pack(side="left", fill="both", expand=True)

        # Update scrollregion as the frame grows
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        # Re-center the frame horizontally and make the window span the full canvas width
        def _center_content(event=None):
            self.canvas.update_idletasks()
            cw = max(1, self.canvas.winfo_width())
            # make the inner window as wide as the canvas so left/right margins are equal
            self.canvas.itemconfigure(self._canvas_window_id, width=cw)
            # place the window's TOP-CENTER at the middle of the canvas
            self.canvas.coords(self._canvas_window_id, cw / 2, 0)

        self.canvas.bind("<Configure>", _center_content)

        # keep a reference so we can call it from apply_theme(), and run once now
        self._recenter_main = _center_content
        self._recenter_main()

        # Use the scrollable_frame as your main container
        self.main_frame = self.scrollable_frame

        self.widget_references = []
        
        self.gear_icon = self.load_gear_icon()

        self.video_label = Label(self.main_frame, bg=self.get_bg())
        self.video_label.pack(pady=(20, 10))

        self.control_btn = Button(
            self.main_frame,
            text="Start",
            command=self.toggle_camera,
            font=("Segoe UI", 20, "bold"),
            bg=self.get_btn_bg(),
            fg=self.get_fg(),
            activebackground=self.get_btn_bg(),
            activeforeground=self.get_fg(),
            width=12,
        )
        self.control_btn.pack(pady=(10, 10))

        self.status = Label(
            self.main_frame,
            text="Status: Stopped",
            fg="red",
            font=("Segoe UI", 14, "bold"),
            bg=self.get_bg(),
        )
        self.status.pack(pady=(0, 20))

        self.button_grid = Frame(self.main_frame, bg=self.get_bg())
        self.button_grid.pack()

        style = {
            "font": ("Segoe UI", 11, "bold"),
            "width": 16,
            "height": 2,
            "bg": self.get_btn_bg(),
            "fg": self.get_fg(),
            "activebackground": self.get_btn_bg(),
            "activeforeground": self.get_fg(),
        }

        self.upload_btn = Button(self.button_grid, text="Upload Pictures", command=self.upload_picture, **style)
        self.upload_btn.grid(row=0, column=0, padx=10, pady=10)

        self.summary_btn = Button(self.button_grid, text="Last Summary", command=self.show_summary, **style)
        self.summary_btn.grid(row=0, column=1, padx=10, pady=10)
        self.summary_btn.grid_remove()

        self.reload_btn = Button(self.button_grid, text="Reload Faces", command=self.reload_known_faces, **style)
        self.reload_btn.grid(row=1, column=0, padx=10, pady=10)

        self.previous_btn = Button(self.button_grid, text="All Summaries", command=self.show_previous_sessions, **style)
        self.previous_btn.grid(row=1, column=1, padx=10, pady=10)

        self.settings_btn = Button(self.root, image=self.gear_icon if self.gear_icon else None,
                                   text="⚙" if not self.gear_icon else "",
                                   font=("Segoe UI", 12), command=self.open_settings_window,
                                   bg=self.get_bg(), fg=self.get_fg(), borderwidth=0,
                                   activebackground=self.get_bg())
        self.settings_btn.place(x=10, y=10)

        self.plan_route_btn = Button(self.button_grid, text="Plan Route", command=self.plan_route, **style)
        self.plan_route_btn.grid(row=3, column=0, columnspan=2, pady=10)

        # Small overlay that sits on top of the video (bottom-right)
        self.engage_bar = Frame(self.root, bg=self.get_bg(), bd=0)

        self.target_label = Label(
            self.engage_bar, text="", font=("Segoe UI", 12, "bold"),
            fg="lime", bg=self.get_bg()
        )
        self.target_label.pack(side="left", padx=(0, 8))

        self.engage_button = Button(
            self.engage_bar, text="Engage", font=("Segoe UI", 10, "bold"),
            bg=self.get_btn_bg(), fg=self.get_fg(),
            activebackground=self.get_btn_bg(), activeforeground=self.get_fg(),
            command=self.engage_action
        )
        self.engage_button.pack(side="left")

        # hidden until a target is acquired
        self.engage_bar.place_forget()

        self.cap = None
        self.running = False
        setup_folders()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.known_faces = load_known_faces("known_faces", self.face_cascade)
        os.makedirs("session_logs", exist_ok=True)
        os.makedirs("session_snapshots", exist_ok=True)

        self.session_data = {}
        self.last_snapshot_time = datetime.min
        self.snapshot_folder_known = ""
        self.snapshot_folder_unknown = ""
        self.frame_lock = threading.Lock()
        self.latest_frame = None

        self.register_theme_widgets()
        self.apply_theme()
        self.render_frame_loop()

    def get_bg(self):
        return "#1e1e1e" if self.theme == "night" else "#f9f9f9"

    def get_fg(self):
        return "#ffffff" if self.theme == "night" else "#000000"

    def get_btn_bg(self):
        return "#333333" if self.theme == "night" else "#d3d3d3"

    def load_gear_icon(self):
        gear_file = "gear_dark.png" if self.theme == "night" else "gear_light.png"
        try:
            gear_image = Image.open(gear_file).resize((24, 24))
            return ImageTk.PhotoImage(gear_image)
        except:
            return None

    def load_theme(self):
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, "r") as f:
                settings = json.load(f)
            return settings.get("theme", "day")
        return "day"

    def save_theme(self, theme):
        with open(SETTINGS_FILE, "w") as f:
            json.dump({"theme": theme}, f)

    def register_theme_widgets(self):
        self.widget_references = [
            self.root, self.main_frame, self.video_label, self.control_btn, self.status,
            self.upload_btn, self.summary_btn, self.reload_btn,
            self.previous_btn, self.settings_btn, self.plan_route_btn,
            self.button_grid, self.engage_bar, self.target_label, self.engage_button
        ]

    def apply_theme(self):
        bg = self.get_bg()
        fg = self.get_fg()
        btn_bg = self.get_btn_bg()

        # 1) Root + MAIN CANVAS + main frame
        self.root.configure(bg=bg)
        if hasattr(self, "canvas") and self.canvas.winfo_exists():
            self.canvas.configure(bg=bg)
        if hasattr(self, "main_frame") and self.main_frame.winfo_exists():
            self.main_frame.configure(bg=bg)

        # 2) Known widgets
        for widget in self.widget_references:
            try:
                if not widget.winfo_exists():
                    continue
            except Exception:
                continue

            if isinstance(widget, (Label, Frame, tk.Tk)):
                try:
                    widget.configure(bg=bg)
                except tk.TclError:
                    pass
                if isinstance(widget, Label):
                    try:
                        widget.configure(fg=fg)
                    except tk.TclError:
                        pass

            elif isinstance(widget, Button):
                try:
                    widget.configure(bg=btn_bg, fg=fg,
                                     activebackground=btn_bg, activeforeground=fg)
                except tk.TclError:
                    pass
                # also theme the parent frame if needed
                try:
                    widget.master.configure(bg=bg)
                except tk.TclError:
                    pass

            elif isinstance(widget, Text):
                try:
                    widget.configure(bg=bg, fg=fg, insertbackground=fg,
                                     highlightthickness=0)
                except tk.TclError:
                    pass

        # 3) Buttons inside the grid
        for child in self.button_grid.winfo_children():
            if isinstance(child, Button):
                child.configure(bg=btn_bg, fg=fg,
                                activebackground=btn_bg, activeforeground=fg)

        # 4) Gear icon + its button background/fg
        self.gear_icon = self.load_gear_icon()
        if self.gear_icon:
            self.settings_btn.config(image=self.gear_icon, text="",
                                     bg=bg, activebackground=bg, fg=fg)
        else:
            self.settings_btn.config(image="", text="⚙",
                                     bg=bg, activebackground=bg, fg=fg)

        # 5) If an overlay panel is open, theme it (recursively)
        def _apply_theme_tree_local(widget):
            try:
                if isinstance(widget, Frame):
                    widget.configure(bg=bg)
                elif isinstance(widget, Label):
                    widget.configure(bg=bg, fg=fg)
                elif isinstance(widget, Button):
                    widget.configure(bg=btn_bg, fg=fg,
                                     activebackground=btn_bg, activeforeground=fg)
                elif isinstance(widget, Text):
                    widget.configure(bg=bg, fg=fg, insertbackground=fg,
                                     highlightthickness=0)
                elif isinstance(widget, Canvas):
                    widget.configure(bg=bg, highlightthickness=0, bd=0)
            except tk.TclError:
                pass

            for c in widget.winfo_children():
                _apply_theme_tree_local(c)

        if hasattr(self, "_panel") and self._panel and self._panel.winfo_exists():
            _apply_theme_tree_local(self._panel)

        # 6) Recenter the main content after palette change
        if hasattr(self, "_recenter_main") and callable(self._recenter_main):
            self.canvas.update_idletasks()
            self._recenter_main()  # positions the content using the real canvas width

    def open_settings_window(self):
        """Open Settings inside the main window using the overlay panel (no new window)."""
        body = self.open_panel("Settings")  # Back button is provided by open_panel()

        bg = self.get_bg()
        fg = self.get_fg()
        btn_bg = self.get_btn_bg()

        # Title inside the panel body (header already shows "Settings")
        tk.Label(
            body, text="Choose Theme:", font=("Segoe UI", 12, "bold"),
            bg=bg, fg=fg
        ).pack(pady=(16, 10))

        def set_theme(theme):
            self.theme = theme
            self.save_theme(theme)
            self.apply_theme()   # keep the panel open so user can see the change immediately

        # Day/Night buttons (styled with current theme colors)
        tk.Button(
            body, text="Day Mode", font=("Segoe UI", 10, "bold"), width=18,
            bg=btn_bg, fg=fg, activebackground=btn_bg, activeforeground=fg,
            command=lambda: set_theme("day")
        ).pack(pady=6)

        tk.Button(
            body, text="Night Mode", font=("Segoe UI", 10, "bold"), width=18,
            bg=btn_bg, fg=fg, activebackground=btn_bg, activeforeground=fg,
            command=lambda: set_theme("night")
        ).pack(pady=4)

    # --- Rest of the methods continue in next message due to length limit ---
    def toggle_camera(self):
        if not self.running:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.running = True
            self.status.config(text="Status: Running", fg="green")
            self.control_btn.config(text="Stop")
            self.summary_btn.grid_remove()
            self.previous_btn.grid_remove()
            # Reset engage overlay at start
            self.engagement_active = False
            self.target_label.config(text="")
            self.engage_bar.place_forget()

            now = datetime.now()
            timestamp = now.strftime('%Y-%m-%d_%H-%M-%S')
            self.snapshot_folder_known = f"session_snapshots/{timestamp}/known"
            self.snapshot_folder_unknown = f"session_snapshots/{timestamp}/unknown"
            os.makedirs(self.snapshot_folder_known, exist_ok=True)
            os.makedirs(self.snapshot_folder_unknown, exist_ok=True)

            self.session_data = {
                "start_time": now,
                "end_time": None,
                "total_faces": 0,
                "juliana_faces": 0,
                "unknown_faces": 0,
                "detected_names": {},
                "last_location": None,
                "last_location_time": None,
                "file_name": f"session_logs/session_{timestamp}.txt"
            }

            self.last_snapshot_time = datetime.min
            threading.Thread(target=self.frame_capture_loop, daemon=True).start()
        else:
            self.running = False
            self.status.config(text="Status: Stopped", fg="red")
            self.control_btn.config(text="Start")
            self.session_data["end_time"] = datetime.now()
            # Hide overlay when stopping
            self.engagement_active = False
            self.engage_bar.place_forget()

            if self.cap:
                self.cap.release()
                self.cap = None
            self.video_label.config(image='')

            self.save_summary_to_file()
            self.summary_btn.grid()
            self.previous_btn.grid()

    def frame_capture_loop(self):
        while self.running and self.cap:
            ret, frame = self.cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detections, unknown_count = recognize_faces_in_frame(
                frame, gray, self.face_cascade, self.known_faces
            )

            now = datetime.now()
            save_snapshot = (now - self.last_snapshot_time).total_seconds() >= 3
            coords = ["Unknown", "Unknown"]
            if save_snapshot:
                location = geocoder.ip('me')
                coords = location.latlng if location.ok else ["Unknown", "Unknown"]

            for (x, y, w, h, label, score) in detections:
                color = (0, 255, 0) if label == "Juliana" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                self.session_data["total_faces"] += 1
                if label == "Juliana":
                    self.session_data["juliana_faces"] += 1
                    folder = self.snapshot_folder_known

    # Play Windows system ping sound
                    winsound.PlaySound("C:\\Windows\\Media\\Windows Notify.wav",
                                       winsound.SND_FILENAME | winsound.SND_ASYNC)

    # Show on-screen alert only every 10 seconds
                    current_time = time.time()
                    last_time = self.last_engagement_time.get(label, 0)

                    if current_time - last_time > 10:
                        self.last_engagement_time[label] = current_time
                        self.root.after(0, lambda: self.display_target_acquired(label))

                else:
                    self.session_data["unknown_faces"] += 1
                    folder = self.snapshot_folder_unknown

                self.session_data["detected_names"][label] = self.session_data["detected_names"].get(label, 0) + 1

                if save_snapshot:
                    snapshot = frame.copy()
                    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
                    cv2.putText(snapshot, f"{label} @ {timestamp}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.putText(snapshot, f"Location: [{coords[0]}, {coords[1]}]", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    filename = f"{label}_{timestamp}.jpg"
                    filepath = os.path.join(folder, filename)
                    cv2.imwrite(filepath, snapshot)

                    self.session_data["last_location"] = coords
                    self.session_data["last_location_time"] = now.strftime("%H:%M:%S")

            if save_snapshot:
                self.last_snapshot_time = now

            with self.frame_lock:
                self.latest_frame = frame
    def render_frame_loop(self):
        if self.running:
            with self.frame_lock:
                frame = self.latest_frame.copy() if self.latest_frame is not None else None

            if frame is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

        if self.root.winfo_exists():
            self.root.after(30, self.render_frame_loop)

    def save_summary_to_file(self):
        data = self.session_data
        if not data["end_time"]:
            return

        duration = str(data["end_time"] - data["start_time"]).split(".")[0]
        accuracy = (data["juliana_faces"] / data["total_faces"] * 100) if data["total_faces"] else 0

        lines = [
            f"Date: {data['start_time'].strftime('%B %d, %Y')}",
            f"Session Duration: {duration}"
        ]

        if data["last_location"] and data["last_location_time"]:
            lat, lon = data["last_location"]
            lines.append(f"Last Known Location: [{lat}, {lon}] at {data['last_location_time']}")
        else:
            lines.append("Last Known Location: Unknown")

        lines.append(f"Facial Recognition Accuracy: {accuracy:.2f}%")
        lines.append("Detected Faces:")
        for name, count in data["detected_names"].items():
            lines.append(f"  - {name} ({count})")

        lines.append("\nSnapshots saved in:")
        lines.append(f"  - Known: {self.snapshot_folder_known}")
        lines.append(f"  - Unknown: {self.snapshot_folder_unknown}")

        with open(data["file_name"], "w") as f:
            f.write("\n".join(lines))
            
    def show_summary(self):
        """
        Open the most recent session summary.
        If the current run has not created self.session_data yet,
        fall back to the latest file in session_logs.
        """
        # Prefer the current session file if available
        try:
            filepath = self.session_data.get("file_name")
        except Exception:
            filepath = None

        if filepath and os.path.exists(filepath):
            self.show_summary_from_file(filepath)
            return

        # Fallback: latest summary from session_logs
        if os.path.exists("session_logs"):
            logs = sorted(
                [f for f in os.listdir("session_logs") if f.endswith(".txt")],
                reverse=True
            )
            if logs:
                latest = os.path.join("session_logs", logs[0])
                self.show_summary_from_file(latest)
                return

        messagebox.showinfo("No Summary", "No session summaries found yet. Run a session first.")

    def show_summary_from_file(self, filepath):
        if not os.path.exists(filepath):
            messagebox.showerror("Error", f"Summary file not found:\n{filepath}")
            return

        with open(filepath, "r") as f:
            content = f.read()

        body = self.open_panel("Session Summary")
        bg = self.get_bg()
        fg = self.get_fg()
        btn_bg = self.get_btn_bg()

        # two columns: text and map
        # two columns: text and map (equal split)
        body.grid_columnconfigure(0, weight=1, uniform="cols")
        body.grid_columnconfigure(1, weight=1, uniform="cols")

        # Summary text
        text_area = Text(body, wrap='word', height=16, font=("Segoe UI", 10),
                         bg=bg, fg=fg, insertbackground=fg)
        text_area.insert(END, content)
        text_area.configure(state='disabled')
        text_area.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # Map
        map_canvas = Canvas(body, height=300, bg=bg, highlightthickness=0, bd=0)
        map_canvas.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        try:
            latlon = content.split("Last Known Location: [")[1].split("]")[0]
            lat, lon = latlon.split(", ")
            mapbox_token = "pk.eyJ1IjoianVsaWFuYTItNCIsImEiOiJjbWJ1dmN0cjUwOXc2MmxteHFjYzd5Z3R4In0.nviVNGXAt_oYWn3pRClM9Q"
            map_url = (
                f"https://api.mapbox.com/styles/v1/mapbox/streets-v11/static/"
                f"pin-l+00ff00({lon},{lat})/{lon},{lat},14,0/400x300"
                f"?access_token={mapbox_token}"
            )
            img_data = urllib.request.urlopen(map_url).read()
            map_img = Image.open(BytesIO(img_data))
            map_tk = ImageTk.PhotoImage(map_img)
            Label(body, text=f"Last known location of Juliana",
                  font=("Segoe UI", 12, "bold"), bg=bg, fg=fg).grid(row=1, column=1, pady=(0, 10))
            map_canvas.create_image(0, 0, anchor="nw", image=map_tk)
            map_canvas.image = map_tk
        except:
            pass

        # Snapshots
        session_id = os.path.basename(filepath).replace("session_", "").replace(".txt", "")
        for idx, label in enumerate(["known", "unknown"]):
            dir_path = os.path.join("session_snapshots", session_id, label)
            frame = Frame(body, bg=bg)
            frame.grid(row=2, column=idx, padx=10, pady=5, sticky="nw")
            Label(frame, text=f"{label.capitalize()} Snapshots:",
                  font=("Segoe UI", 10, "bold"), bg=bg, fg=fg).pack(anchor="w")
            for img_file in os.listdir(dir_path) if os.path.exists(dir_path) else []:
                img_path = os.path.join(dir_path, img_file)
                try:
                    img = Image.open(img_path).resize((100, 100))
                    thumb = ImageTk.PhotoImage(img)
                    lbl = Label(frame, image=thumb, cursor="hand2", bg=bg)
                    lbl.image = thumb
                    lbl.pack(side="left", padx=5, pady=5)
                    lbl.bind("<Button-1>", lambda e, path=img_path: self.show_full_image(path))
                except:
                    continue

        Button(body, text="Export Full Session",
               command=lambda: self.export_session(filepath),
               font=("Segoe UI", 10, "bold"), bg=btn_bg, fg=fg,
               activebackground=btn_bg, activeforeground=fg
               ).grid(row=3, column=0, columnspan=2, pady=12)

    def show_full_image(self, path):
        top = Toplevel(self.root)
        top.title(os.path.basename(path))
        img = Image.open(path)
        photo = ImageTk.PhotoImage(img)
        lbl = Label(top, image=photo)
        lbl.image = photo
        lbl.pack()

    def export_session(self, filepath):
        session_id = os.path.basename(filepath).replace("session_", "").replace(".txt", "")
        dest = filedialog.asksaveasfilename(
            defaultextension=".zip",
            filetypes=[("Zip Files", "*.zip")],
            initialfile=f"session_{session_id}.zip"
        )
        if not dest:
            return

        temp_dir = f"temp_export_{session_id}"
        os.makedirs(temp_dir, exist_ok=True)
        shutil.copy(filepath, os.path.join(temp_dir, "session_summary.txt"))

        for label in ["known", "unknown"]:
            folder = os.path.join("session_snapshots", session_id, label)
            if os.path.exists(folder):
                shutil.copytree(folder, os.path.join(temp_dir, label), dirs_exist_ok=True)

        with zipfile.ZipFile(dest, 'w') as zipf:
            for foldername, _, filenames in os.walk(temp_dir):
                for filename in filenames:
                    zipf.write(os.path.join(foldername, filename),
                               os.path.relpath(os.path.join(foldername, filename), temp_dir))
        shutil.rmtree(temp_dir)
        messagebox.showinfo("Exported", f"Session exported to:\n{dest}")

    def upload_picture(self):
        filepath = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if not filepath:
            return
        name = simpledialog.askstring("Enter Name", "Name of the person:")
        if not name:
            return
        known_faces_dir = "known_faces"
        os.makedirs(known_faces_dir, exist_ok=True)
        existing = [f for f in os.listdir(known_faces_dir) if f.startswith(f"{name}_face_")]
        next_index = len(existing) + 1
        if next_index > 10:
            messagebox.showwarning("Limit Reached", f"{name} already has 10 face images.")
            return
        new_filename = f"{name}_face_{next_index}.jpg"
        save_path = os.path.join(known_faces_dir, new_filename)
        try:
            img = Image.open(filepath)
            img.save(save_path)
            messagebox.showinfo("Success", f"Image saved as {new_filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save image: {e}")

    def reload_known_faces(self):
        self.known_faces = load_known_faces("known_faces", self.face_cascade)
        messagebox.showinfo("Reloaded", "Known faces reloaded.")

    def show_previous_sessions(self):
        body = self.open_panel("All Summaries")
        bg = self.get_bg()
        fg = self.get_fg()
        btn_bg = self.get_btn_bg()

        logs = sorted(os.listdir("session_logs"), reverse=True)
        for f in logs:
            if f.endswith(".txt"):
                dt = datetime.strptime(
                    f.replace("session_", "").replace(".txt", ""),
                    "%Y-%m-%d_%H-%M-%S"
                )
                label = dt.strftime("%m/%d/%Y %H:%M")
                btn = Button(
                    body, text=label, width=40,
                    command=lambda file=f: self.show_summary_from_file(os.path.join("session_logs", file)),
                    bg=btn_bg, fg=fg, activebackground=btn_bg, activeforeground=fg,
                    font=("Segoe UI", 10, "bold")
                )
                btn.pack(pady=4, padx=10, anchor="w")

    def plan_route(self):
        body = self.open_panel("Drone Route Planner")
        bg = self.get_bg()
        fg = self.get_fg()
        btn_bg = self.get_btn_bg()

        route_color = "blue" if self.theme == "day" else "cyan"
        point_color = "black" if self.theme == "day" else "white"
        canvas_bg = "#ffffff" if self.theme == "day" else "#1e1e1e"

        canvas = Canvas(body, bg=canvas_bg, width=700, height=500, highlightthickness=0)
        canvas.pack(pady=10)

        try:
            base_drone_img = Image.open("drone_icon.png").resize((30, 30))
        except Exception as e:
            messagebox.showerror("Image Error", f"Could not load drone icon:\n{e}")
            return

        route_points, route_lines = [], []
        drone = None
        looping = False

        def draw_point(x, y):
            canvas.create_oval(x-3, y-3, x+3, y+3, fill=point_color, outline=point_color)

        def draw_line(p1, p2):
            line = canvas.create_line(p1[0], p1[1], p2[0], p2[1], fill=route_color, width=2)
            route_lines.append(line)

        def redraw_route():
            canvas.delete("all")
            for i, (x, y) in enumerate(route_points):
                draw_point(x, y)
                if i > 0:
                    draw_line(route_points[i-1], route_points[i])
            if looping and len(route_points) > 2:
                draw_line(route_points[-1], route_points[0])

        def on_click(event):
            route_points.append((event.x, event.y))
            redraw_route()

        def reset_route():
            nonlocal drone
            route_points.clear()
            for line in route_lines:
                canvas.delete(line)
            route_lines.clear()
            canvas.delete("all")
            drone = None

        def undo_last():
            if route_points:
                route_points.pop()
                redraw_route()

        def toggle_loop():
            nonlocal looping
            looping = not looping
            loop_btn.config(text="Loop: ON" if looping else "Loop: OFF")
            redraw_route()

        def simulate_drone():
            nonlocal drone
            if len(route_points) < 2:
                messagebox.showwarning("Not Enough Points", "Draw at least two points to simulate.")
                return
            if drone:
                canvas.delete(drone)

            path = route_points[:]
            if looping and len(path) >= 2:
                path.append(route_points[0])

            segments, angles = [], []
            for i in range(len(path) - 1):
                x1, y1 = path[i]
                x2, y2 = path[i + 1]
                dx, dy = x2 - x1, y2 - y1
                angle = -1 * (180 / 3.14159) * (dy / (abs(dx) + 1e-5))
                dist = (dx**2 + dy**2) ** 0.5
                steps = max(1, int(dist // 2))
                for j in range(steps):
                    t = j / steps
                    x = x1 + t * dx
                    y = y1 + t * dy
                    segments.append((x, y))
                    angles.append(angle)

            def move(i=0):
                nonlocal drone
                if i < len(segments):
                    x, y = segments[i]
                    angle = angles[i]
                    rotated_img = base_drone_img.rotate(angle, expand=True)
                    rotated_photo = ImageTk.PhotoImage(rotated_img)
                    if canvas.winfo_exists():
                        if drone:
                            canvas.delete(drone)
                        drone = canvas.create_image(x, y, image=rotated_photo)
                        canvas.image = rotated_photo
                        body.after(10, lambda: move(i + 1))
                elif looping:
                    simulate_drone()

            move()

        canvas.bind("<Button-1>", on_click)

        control = Frame(body, bg=bg)
        control.pack()
        Button(control, text="Simulate Route", command=simulate_drone,
               font=("Segoe UI", 11, "bold"), bg=btn_bg, fg=fg,
               activebackground=btn_bg, activeforeground=fg, width=16).grid(row=0, column=0, padx=10, pady=5)
        loop_btn = Button(control, text="Loop: OFF", command=toggle_loop,
                          font=("Segoe UI", 11, "bold"), bg=btn_bg, fg=fg,
                          activebackground=btn_bg, activeforeground=fg, width=16)
        loop_btn.grid(row=0, column=1, padx=10)
        Button(control, text="Undo Last Point", command=undo_last,
               font=("Segoe UI", 11, "bold"), bg=btn_bg, fg=fg,
               activebackground=btn_bg, activeforeground=fg, width=16).grid(row=1, column=0, padx=10, pady=5)
        Button(control, text="Reset Route", command=reset_route,
               font=("Segoe UI", 11, "bold"), bg=btn_bg, fg=fg,
               activebackground=btn_bg, activeforeground=fg, width=16).grid(row=1, column=1, padx=10)


    def prompt_engage(self, name):
        response = messagebox.askyesno("Target Acquired", f"{name} detected.\nWould you like to engage?")
        if response:
            print(f"Engagement confirmed on {name}.")  # Replace with future action
        else:
            print(f"{name} ignored.")

    def engage_action(self):
        print("Engagement confirmed.")  
        self.target_label.config(text="Engagement confirmed.")
        self.engage_button.pack_forget()
        self.engagement_active = False
        
    def display_target_acquired(self, name):
        """
        Show the Engage overlay anchored to the bottom-right of the live video.
        This is called on the main thread via root.after(...).
        """
        # Don't spam if it's already visible
        if self.engagement_active:
            return

        self.engagement_active = True
        self.target_label.config(text=f"Target Acquired: {name} — Awaiting engagement...")

        # Place the overlay relative to the video widget
        try:
            # bottom-right corner of the live video
            self.engage_bar.place(in_=self.video_label, relx=0.98, rely=0.98, anchor="se")
        except Exception:
            # Fallback: bottom-right of the root window
            self.engage_bar.place(relx=0.98, rely=0.98, anchor="se")

    def open_panel(self, title=""):
        """Create a centered, responsive overlay panel and return its scrollable body frame."""
        self.close_panel()  # ensure only one panel exists

        bg = self.get_bg()
        fg = self.get_fg()
        btn_bg = self.get_btn_bg()

        # let Tk compute current window size
        self.root.update_idletasks()
        root_w = max(700, self.root.winfo_width())
        root_h = max(500, self.root.winfo_height())

        # responsive panel size (fits inside the window with margins)
        panel_w = min(int(root_w * 0.92), 1100)
        panel_h = min(int(root_h * 0.9), 820)

        # centered overlay
        self._panel = tk.Frame(self.root, bg=bg, bd=2, relief="ridge")
        self._panel.place(relx=0.5, rely=0.5, anchor="center", width=panel_w, height=panel_h)

        # header (Back + title)
        header = tk.Frame(self._panel, bg=bg)
        header.pack(fill="x")

        tk.Button(
            header, text="◀ Back", command=self.close_panel,
            font=("Segoe UI", 10, "bold"), bg=btn_bg, fg=fg,
            activebackground=btn_bg, activeforeground=fg, width=10
        ).pack(side="left", padx=8, pady=8)

        tk.Label(
            header, text=title, font=("Segoe UI", 12, "bold"),
            bg=bg, fg=fg
        ).pack(side="left", padx=6)

        # scrollable body
        body_canvas = tk.Canvas(self._panel, bg=bg, highlightthickness=0)
        vsb = tk.Scrollbar(self._panel, orient="vertical", command=body_canvas.yview)
        body_canvas.configure(yscrollcommand=vsb.set)

        vsb.pack(side="right", fill="y")
        body_canvas.pack(side="left", fill="both", expand=True)

        # inner body frame
        body = tk.Frame(body_canvas, bg=bg)
        body_id = body_canvas.create_window((0, 0), window=body, anchor="nw")

        # keep scrollregion and inner width in sync
        def _on_body_config(_):
            body_canvas.configure(scrollregion=body_canvas.bbox("all"))
            # make inner body always match canvas width (no horizontal cutoff)
            body_canvas.itemconfigure(body_id, width=body_canvas.winfo_width())

        body.bind("<Configure>", _on_body_config)

        # mouse wheel scrolling inside the panel
        def _on_wheel(event):
            body_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        # Windows wheel
        for w in (body_canvas, body, self._panel):
            w.bind("<MouseWheel>", _on_wheel)

        # Linux wheel
        for w in (body_canvas, body, self._panel):
            w.bind("<Button-4>", lambda e: body_canvas.yview_scroll(-1, "units"))
            w.bind("<Button-5>", lambda e: body_canvas.yview_scroll(1, "units"))

        self._panel_body = body
        return body


    def close_panel(self):
        """Remove the overlay panel if present."""
        if hasattr(self, "_panel") and self._panel and self._panel.winfo_exists():
            self._panel.destroy()
            self._panel = None
            self._panel_body = None

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
