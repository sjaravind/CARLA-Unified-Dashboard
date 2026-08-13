import os
import csv
import time
import math
import cv2
import numpy as np
import joblib
import pandas as pd
import threading
import tkinter as tk
from PIL import Image, ImageTk
import customtkinter as ctk
from tkinter import filedialog
import carla

# Configure CustomTkinter Theme
ctk.set_appearance_mode("Dark")

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "safety_model.pkl")


# ==============================================================================
# SAFETY ADVISOR (merged from the former safety_advisor.py)
# ------------------------------------------------------------------------------
# Provides PredictiveSafetyMonitor, used below via `evaluate_intent(...)` -> dict.
#
#   1. Loads the ML model trained by train_safety_model.py (safety_model.pkl),
#      if present, to get a continuous "risk probability" for the current
#      driving state.
#   2. Falls back to pure rule-based logic if no model file is found yet, so
#      the dashboard always works even before you've trained anything.
#   3. Applies explicit weather physics adjustments:
#        - Rain  -> lower tyre grip -> earlier/softer braking advice, skid
#                   warning on hard braking, longer safe-following-distance.
#        - Fog   -> visibility-limited -> advisory speed cap, no braking-force
#                   change (fog doesn't affect grip, just what the driver sees).
#        - Sunny/Clear -> higher/normal grip -> baseline behaviour (note only).
#   4. Combines rule-based hazards (closing distance / TTC, lane drift) with
#      the ML risk score to decide an overall intervention level and advisory
#      throttle/steering limits, similar in spirit to real ADAS/AEB/LKA systems.
#
# This class never silently takes over the car -- it estimates and *advises*.
# It's on the dashboard code below to decide how much of throttle_limit /
# steer_gain / brake_assist to actually apply to the vehicle control commands.
# NOTE: kept in this same file (rather than a separate module) since it's
# only ever used together with the dashboard at runtime. The training script
# (train_safety_model.py) stays separate -- it doesn't need CARLA or tkinter,
# and only needs to run occasionally, not every time you drive.
# ==============================================================================

class PredictiveSafetyMonitor:
    def __init__(self, model_path: str = MODEL_PATH):
        self.model = None
        self.features = None
        self._prev_speed = 0.0
        self._prev_brake = 0.0

        if os.path.exists(model_path):
            try:
                bundle = joblib.load(model_path)
                self.model = bundle["model"]
                self.features = bundle["features"]
                print(f"[SafetyAdvisor] Loaded ML risk model from {model_path}")
            except Exception as e:
                print(f"[SafetyAdvisor] Could not load model ({e}); using rule-based fallback only.")
        else:
            print("[SafetyAdvisor] No trained model found yet -- running rule-based only. "
                  "Run train_safety_model.py once you have logged data.")

    # ------------------------------------------------------------------
    def _ml_risk_probability(self, speed_kmh, throttle, brake, steer,
                              rain, fog, num_vehicles, num_pedestrians,
                              accel, brake_jerk):
        if self.model is None:
            return None
        row = {
            "speed_kmh": speed_kmh, "throttle": throttle, "brake": brake,
            "steer": steer, "abs_steer": abs(steer), "rain": rain, "fog": fog,
            "accel": accel, "brake_jerk": brake_jerk,
            "brake_energy": brake * speed_kmh,
            "num_vehicles": num_vehicles, "num_pedestrians": num_pedestrians,
        }
        x = pd.DataFrame([[row[f] for f in self.features]], columns=self.features)
        try:
            return float(self.model.predict_proba(x)[0, 1])
        except Exception:
            return None

    # ------------------------------------------------------------------
    def evaluate_intent(self, pred_throttle, pred_brake, distance_to_lead,
                         radar_ttc, lane_offset, speed_kmh=0.0, steer=0.0,
                         rain=0, fog=0, num_vehicles=0, num_pedestrians=0):
        """
        Returns a dict consumed by the dashboard's `_update_gui_data`.
        Keeps the original keys (comfort_mode, intervention_level,
        throttle_limit, steer_gain, aeb_triggered, ttc_warning) and adds:
        risk_probability, brake_assist, weather_note, safe_follow_distance.
        """
        accel = speed_kmh - self._prev_speed
        brake_jerk = abs(pred_brake - self._prev_brake)
        self._prev_speed = speed_kmh
        self._prev_brake = pred_brake

        throttle_limit = 1.0
        steer_gain = 1.0
        brake_assist = 0.0
        aeb_triggered = False
        ttc_warning = False
        intervention_level = "None"
        weather_notes = []

        # --- 1. Weather-adjusted grip / visibility model -----------------
        grip_factor = 1.0
        if rain:
            grip_factor = 0.6
            weather_notes.append("Wet road: reduced tyre grip, longer stopping distance")
            if pred_brake > 0.7 and speed_kmh > 30:
                weather_notes.append("Hard braking on wet road -- skid risk")
                brake_assist = max(brake_assist, 0.3)
                intervention_level = "Moderate"

        if fog:
            weather_notes.append("Fog: reduced visibility")
            visibility_speed_cap = 60.0
            if speed_kmh > visibility_speed_cap:
                throttle_limit = min(throttle_limit, 0.5)
                weather_notes.append(f"Speed above safe visibility limit (~{visibility_speed_cap:.0f} km/h)")
                intervention_level = "Moderate"

        if not rain and not fog:
            weather_notes.append("Clear conditions: normal grip and visibility")

        speed_ms = speed_kmh / 3.6
        safe_follow_distance = (speed_ms ** 2) / (2 * 7.0 * grip_factor) + speed_ms * 1.0

        # --- 2. Closing-distance / TTC based forward-collision logic -----
        if radar_ttc < 2.0:
            aeb_triggered = True
            throttle_limit = 0.0
            brake_assist = 1.0
            intervention_level = "High"
        elif radar_ttc < 4.0:
            ttc_warning = True
            throttle_limit = min(throttle_limit, 0.5)
            if intervention_level == "None":
                intervention_level = "Moderate"
        if distance_to_lead < safe_follow_distance * 0.5:
            throttle_limit = min(throttle_limit, 0.3)
            if intervention_level == "None":
                intervention_level = "Moderate"

        # --- 3. Lane-keeping assist -----------------------------------
        if abs(lane_offset) > 1.0:
            steer_gain = 0.7
            if intervention_level == "None":
                intervention_level = "Moderate"
        elif abs(lane_offset) > 1.8:
            steer_gain = 0.5
            intervention_level = "High"

        # --- 4. ML risk score -------------------------------------------
        risk_probability = self._ml_risk_probability(
            speed_kmh, pred_throttle, pred_brake, steer, rain, fog,
            num_vehicles, num_pedestrians, accel, brake_jerk
        )
        if risk_probability is not None:
            if risk_probability > 0.75:
                intervention_level = "High"
                throttle_limit = min(throttle_limit, 0.4)
                steer_gain = min(steer_gain, 0.7)
            elif risk_probability > 0.45 and intervention_level == "None":
                intervention_level = "Moderate"
                throttle_limit = min(throttle_limit, 0.7)

        # --- 5. Comfort mode label ----------------------------------------
        if intervention_level == "High":
            comfort_mode = "Emergency Intervention"
        elif intervention_level == "Moderate":
            comfort_mode = "Cautious Assist"
        else:
            comfort_mode = "Comfort Cruise"

        return {
            "comfort_mode": comfort_mode,
            "intervention_level": intervention_level,
            "throttle_limit": throttle_limit,
            "steer_gain": steer_gain,
            "brake_assist": brake_assist,
            "aeb_triggered": aeb_triggered,
            "ttc_warning": ttc_warning,
            "risk_probability": risk_probability,
            "safe_follow_distance": safe_follow_distance,
            "weather_note": " | ".join(weather_notes),
        }


# ==============================================================================
# MAIN DASHBOARD APPLICATION
# ==============================================================================

class CarlaNativeDashboard(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("CARLA Smart Dashboard - Vice City Mode")
        self.geometry("1100x680")
        self.resizable(False, False)
        
        # Override background color to match screenshot space-dark style
        self.configure(fg_color="#080a0f")
        
        # State variables
        self.client = None
        self.world = None
        self.ego_vehicle = None
        self.carla_map = None
        self.camera_sensor = None
        self.radar_sensor = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        
        # Live telemetry states
        self.speed = 0.0
        self.throttle = 0.0
        self.brake = 0.0
        self.steering = 0.0
        self.lane_offset = 0.0
        self.distance_to_lead = 100.0
        self.radar_ttc = 99.9
        self.is_reverse = False
        self.weather_name = "Sunny"
        self.rain_flag = 0
        self.fog_flag = 0
        self.collision_flag = 0
        self.lane_invasion_flag = 0
        self.num_vehicles = 0
        self.num_pedestrians = 0
        
        # Safety / Adaptation configurations
        self.safety_monitor = PredictiveSafetyMonitor()
        self.lka_active = False
        self.throttle_limit = 1.0
        self.steer_gain = 1.0
        self.brake_assist = 0.0
        
        # Logging & History states
        self.active_trial_data = []
        self.log_file = "driving_data.csv"
        self.is_recording = False
        self.rows_saved = 0
        
        # Active camera view perspective
        self.camera_view = "Chase"
        
        # Keyboard controls for manual driving (Up/Down/Left/Right/Space)
        self.keys = {
            "Up": False,
            "Down": False,
            "Left": False,
            "Right": False,
            "space": False
        }
        
        # PIL/Image reference to prevent Garbage Collection
        self.current_tk_image = None
        self.driver_photo_tk = None
        self.driver_photo_path = None
        
        # Build UI layout
        self._build_ui()
        
        # Bind keyboard events
        self.bind("<KeyPress>", self._on_key_press)
        self.bind("<KeyRelease>", self._on_key_release)
        
        # Start periodic UI & control updater loop (20Hz = 50ms)
        self.update_loop()
        
    def _build_ui(self):
        # 1. Main Grid Configuration
        self.grid_columnconfigure(0, weight=3) # Visualizer column
        self.grid_columnconfigure(1, weight=1) # Controls column
        self.grid_rowconfigure(0, weight=1)
        
        # ======================================================================
        # LEFT SIDE: VISUALIZER & OVERLAYS FRAME
        # ======================================================================
        self.visualizer_frame = ctk.CTkFrame(self, corner_radius=15, border_width=2, border_color="#ff0066", fg_color="#0c0e14")
        self.visualizer_frame.grid(row=0, column=0, padx=15, pady=15, sticky="nsew")
        
        # Visualizer Camera View Label
        self.camera_label = ctk.CTkLabel(self.visualizer_frame, text="WAITING FOR CARLA CLIENT CONNECTION...", 
                                         font=("Consolas", 14), fg_color="#020308", text_color="#8892b0")
        self.camera_label.pack(fill="both", expand=True, padx=10, pady=(10, 5))
        
        # Bottom info bar inside visualizer frame
        self.info_panel = ctk.CTkFrame(self.visualizer_frame, fg_color="transparent")
        self.info_panel.pack(fill="x", side="bottom", padx=15, pady=(5, 15))
        
        # A. Driver view panel (replaces the old WASD key visualizer).
        # Shows a small "driver photo" (dummy steering wheel rig) plus a
        # live rotating steering-wheel indicator canvas driven by self.steering.
        # Input is still read from WASD internally (no joystick/wheel yet) --
        # this panel is purely visual, matching the intended hardware look.
        self.driver_frame = ctk.CTkFrame(self.info_panel, fg_color="transparent")
        self.driver_frame.pack(side="left")

        self.driver_photo_canvas = tk.Canvas(self.driver_frame, width=70, height=52,
                                              bg="#181c26", highlightthickness=0)
        self.driver_photo_canvas.grid(row=0, column=0, padx=(0, 8))
        self._draw_driver_photo_placeholder()
        self.driver_photo_canvas.bind("<Button-1>", lambda e: self.load_driver_photo())

        self.wheel_canvas = tk.Canvas(self.driver_frame, width=52, height=52,
                                       bg="#080a0f", highlightthickness=0)
        self.wheel_canvas.grid(row=0, column=1)
        self._draw_wheel_indicator(0.0)
        
        # B. Data Log Text Status
        self.log_status_frame = ctk.CTkFrame(self.info_panel, fg_color="transparent")
        self.log_status_frame.pack(side="left", padx=40)
        
        ctk.CTkLabel(self.log_status_frame, text="DATA LOG", font=("Inter", 11, "bold"), text_color="#ff9900").pack(anchor="w")
        self.lbl_saved_rows = ctk.CTkLabel(self.log_status_frame, text="0 rows saved", font=("Consolas", 11), text_color="#00f0ff")
        self.lbl_saved_rows.pack(anchor="w")
        self.lbl_log_file = ctk.CTkLabel(self.log_status_frame, text="→ driving_data.csv", font=("Consolas", 10), text_color="#8892b0")
        self.lbl_log_file.pack(anchor="w")
        
        # C. Connected status indicator dot
        self.conn_status_frame = ctk.CTkFrame(self.info_panel, fg_color="transparent")
        self.conn_status_frame.pack(side="right", padx=15)
        self.status_dot = ctk.CTkLabel(self.conn_status_frame, text="🔴 DISCONNECTED", font=("Inter", 11, "bold"), text_color="#ff0055")
        self.status_dot.pack()
        
        # ======================================================================
        # RIGHT SIDE: DASHBOARD STATS PANEL
        # ======================================================================
        self.right_panel = ctk.CTkScrollableFrame(self, width=280, corner_radius=0, fg_color="transparent")
        self.right_panel.grid(row=0, column=1, padx=(0, 15), pady=15, sticky="nsew")
        
        # Main neon styled titles
        self.title_label = ctk.CTkLabel(self.right_panel, text="SMART DRIVE", font=("Outfit", 24, "bold"), text_color="#ff0066")
        self.title_label.pack(pady=(5, 1))
        self.subtitle_label = ctk.CTkLabel(self.right_panel, text="CARLA DASHBOARD", font=("Inter", 10, "bold"), text_color="#00f0ff")
        self.subtitle_label.pack(pady=(0, 15))
        
        # --- SYSTEM SECTION ---
        ctk.CTkLabel(self.right_panel, text="- SYSTEM", font=("Inter", 11, "bold"), text_color="#8892b0").pack(anchor="w", padx=5, pady=(5, 2))
        
        self.btn_connect = ctk.CTkButton(self.right_panel, text="CONNECT TO CARLA", font=("Inter", 11, "bold"), height=30,
                                         fg_color="transparent", border_width=2, border_color="#1e7e34", text_color="#28a745", 
                                         hover_color="#0a2512", command=self.connect_carla)
        self.btn_connect.pack(fill="x", pady=4)
        
        self.btn_spawn = ctk.CTkButton(self.right_panel, text="SPAWN MY VEHICLE", font=("Inter", 11, "bold"), height=30,
                                       state="disabled", fg_color="transparent", border_width=2, border_color="#17a2b8", 
                                       text_color="#17a2b8", hover_color="#09212b", command=self.spawn_vehicle)
        self.btn_spawn.pack(fill="x", pady=4)
        
        # --- GAUGES SECTION (Dials Canvas) ---
        ctk.CTkLabel(self.right_panel, text="- GAUGES", font=("Inter", 11, "bold"), text_color="#8892b0").pack(anchor="w", padx=5, pady=(10, 2))
        
        # Custom Canvas for Semicircular speedometer and throttle arcs
        self.gauges_canvas = tk.Canvas(self.right_panel, width=260, height=105, bg="#080a0f", highlightthickness=0)
        self.gauges_canvas.pack(pady=5)
        self._draw_gauges_ui(0.0, 0.0) # Draw initial flat gauges
        
        # --- CONTROLS SECTION (Sliders) ---
        ctk.CTkLabel(self.right_panel, text="- CONTROLS", font=("Inter", 11, "bold"), text_color="#8892b0").pack(anchor="w", padx=5, pady=(10, 2))
        
        # Throttle Slider
        t_frame = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        t_frame.pack(fill="x", pady=3)
        ctk.CTkLabel(t_frame, text="THROTTLE", font=("Inter", 10, "bold"), text_color="#8892b0").pack(side="left")
        self.lbl_slider_throttle = ctk.CTkLabel(t_frame, text="0.00", font=("Consolas", 10, "bold"), text_color="#ffffff")
        self.lbl_slider_throttle.pack(side="right")
        self.slider_throttle = ctk.CTkSlider(self.right_panel, from_=0.0, to=1.0, progress_color="#00f0ff")
        self.slider_throttle.set(0.0)
        self.slider_throttle.pack(fill="x", pady=(0, 6))
        
        # Brake Slider
        b_frame = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        b_frame.pack(fill="x", pady=3)
        ctk.CTkLabel(b_frame, text="BRAKE", font=("Inter", 10, "bold"), text_color="#8892b0").pack(side="left")
        self.lbl_slider_brake = ctk.CTkLabel(b_frame, text="0.00", font=("Consolas", 10, "bold"), text_color="#ffffff")
        self.lbl_slider_brake.pack(side="right")
        self.slider_brake = ctk.CTkSlider(self.right_panel, from_=0.0, to=1.0, progress_color="#ff0066")
        self.slider_brake.set(0.0)
        self.slider_brake.pack(fill="x", pady=(0, 6))
        
        # Steer Slider
        s_frame = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        s_frame.pack(fill="x", pady=3)
        ctk.CTkLabel(s_frame, text="STEER", font=("Inter", 10, "bold"), text_color="#8892b0").pack(side="left")
        self.lbl_slider_steer = ctk.CTkLabel(s_frame, text="+0.00", font=("Consolas", 10, "bold"), text_color="#ffffff")
        self.lbl_slider_steer.pack(side="right")
        self.slider_steer = ctk.CTkSlider(self.right_panel, from_=-1.0, to=1.0, progress_color="#00f0ff")
        self.slider_steer.set(0.0)
        self.slider_steer.pack(fill="x", pady=(0, 8))
        
        # --- CAMERA VIEW SECTION (Buttons row) ---
        ctk.CTkLabel(self.right_panel, text="- CAMERA VIEW", font=("Inter", 11, "bold"), text_color="#8892b0").pack(anchor="w", padx=5, pady=(10, 2))
        self.cam_buttons_frame = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        self.cam_buttons_frame.pack(fill="x", pady=2)
        
        self.cam_buttons = {}
        for idx, view_name in enumerate(["Driver", "Chase", "Top"]):
            btn = ctk.CTkButton(self.cam_buttons_frame, text=view_name, font=("Inter", 10, "bold"), width=80, height=26,
                                fg_color="#181c26" if view_name != "Chase" else "#ff0066",
                                hover_color="#ff0055", command=lambda v=view_name: self.set_camera_view(v))
            btn.grid(row=0, column=idx, padx=2)
            self.cam_buttons[view_name] = btn
            
        # --- WEATHER SECTION (Grid buttons) ---
        ctk.CTkLabel(self.right_panel, text="- WEATHER", font=("Inter", 11, "bold"), text_color="#8892b0").pack(anchor="w", padx=5, pady=(10, 2))
        self.weather_grid_frame = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        self.weather_grid_frame.pack(fill="x", pady=2)
        
        self.weather_buttons = {}
        weather_list = ["Sunny", "Rainy", "Foggy"]
        for idx, w_name in enumerate(weather_list):
            btn = ctk.CTkButton(self.weather_grid_frame, text=w_name, font=("Inter", 10, "bold"), width=80, height=26,
                                fg_color="#181c26" if w_name != "Sunny" else "#00f0ff", text_color="#ffffff" if w_name != "Sunny" else "#000000",
                                hover_color="#00e0ee", command=lambda w=w_name: self.set_weather(w))
            btn.grid(row=0, column=idx, padx=2, pady=2)
            self.weather_buttons[w_name] = btn
            
        # --- TRIAL LOGGER ---
        ctk.CTkLabel(self.right_panel, text="- TRIAL LOGGER", font=("Inter", 11, "bold"), text_color="#8892b0").pack(anchor="w", padx=5, pady=(10, 2))
        self.actions_frame = ctk.CTkFrame(self.right_panel, corner_radius=10, border_width=1, border_color="#2c3144", fg_color="#0c0e14")
        self.actions_frame.pack(fill="x", pady=4)
        
        self.entry_filename = ctk.CTkEntry(self.actions_frame, placeholder_text="driving_data.csv", font=("Inter", 11), fg_color="#080a0f")
        self.entry_filename.insert(0, "driving_data.csv")
        self.entry_filename.pack(fill="x", padx=10, pady=5)
        
        self.btn_record = ctk.CTkButton(self.actions_frame, text="START DATA LOGGER", font=("Inter", 11, "bold"), height=30,
                                        fg_color="#6f42c1", hover_color="#5a32a3", command=self.toggle_logger)
        self.btn_record.pack(fill="x", padx=10, pady=(2, 10))
        
        # --- LIVE PARAMETERS STATUS ---
        ctk.CTkLabel(self.right_panel, text="- STATUS OVERLAYS", font=("Inter", 11, "bold"), text_color="#8892b0").pack(anchor="w", padx=5, pady=(10, 2))
        self.parameters_frame = ctk.CTkFrame(self.right_panel, corner_radius=10, border_width=1, border_color="#2c3144", fg_color="#0c0e14")
        self.parameters_frame.pack(fill="x", pady=4)
        
        self.param_labels = {}
        params_to_show = [
            ("Actuator Throttle", "0.0 %"),
            ("Actuator Brake", "0.0 %"),
            ("Actuator Steering", "0.0 %"),
            ("Active Gear", "Drive (D)"),
            ("Lane Offset", "0.00 m"),
            ("Time-to-Collision", "N/A"),
            ("Safe Follow Distance", "N/A"),
            ("AI Risk Score", "N/A"),
            ("AI Comfort Mode", "Comfort Cruise"),
            ("AI Safety Limit", "None"),
            ("AI Steering Gain", "1.00"),
            ("Weather Advisory", "Clear conditions"),
            ("System Threat", "Low")
        ]
        
        for name, default in params_to_show:
            row_frame = ctk.CTkFrame(self.parameters_frame, fg_color="transparent")
            row_frame.pack(fill="x", padx=12, pady=2)
            
            lbl_name = ctk.CTkLabel(row_frame, text=name, font=("Inter", 10), text_color="#8892b0")
            lbl_name.pack(side="left")
            
            lbl_val = ctk.CTkLabel(row_frame, text=default, font=("Consolas", 10, "bold"), text_color="#ffffff")
            lbl_val.pack(side="right")
            self.param_labels[name] = lbl_val
            
    def _draw_driver_photo_placeholder(self):
        """Placeholder shown until a real photo (dummy steering wheel rig /
        driver shot) is loaded. Click the thumbnail to load one."""
        c = self.driver_photo_canvas
        c.delete("all")
        c.create_rectangle(1, 1, 69, 51, outline="#2c3144")
        c.create_oval(20, 12, 50, 42, outline="#8892b0", width=2)  # dummy wheel icon
        c.create_line(35, 12, 35, 42, fill="#8892b0")
        c.create_line(20, 27, 50, 27, fill="#8892b0")
        c.create_text(35, 47, text="click to load photo", fill="#8892b0", font=("Inter", 6))

    def load_driver_photo(self):
        """Lets the user pick a real photo (driver + dummy steering wheel)
        to display in place of the placeholder. Call again any time to swap it."""
        path = filedialog.askopenfilename(
            title="Select driver photo",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            img = Image.open(path).convert("RGB").resize((70, 52))
            self.driver_photo_tk = ImageTk.PhotoImage(img)
            self.driver_photo_path = path
            self.driver_photo_canvas.delete("all")
            self.driver_photo_canvas.create_image(0, 0, anchor="nw", image=self.driver_photo_tk)
        except Exception as e:
            tk.messagebox.showerror("Photo Load Error", f"Could not load image:\n{e}")

    def _draw_wheel_indicator(self, steer_value):
        """Small rotating steering-wheel graphic driven by the current steer
        value (-1.0 .. 1.0), standing in for a physical wheel/joystick readout."""
        c = self.wheel_canvas
        c.delete("all")
        cx, cy, r = 26, 26, 20
        angle = steer_value * 90.0  # +/-90 degrees of visual rotation per full lock
        c.create_oval(cx - r, cy - r, cx + r, cy + r, outline="#00f0ff", width=3)
        for offset_deg in (0, 120, 240):
            a = math.radians(angle + offset_deg)
            x = cx + r * math.sin(a)
            y = cy - r * math.cos(a)
            c.create_line(cx, cy, x, y, fill="#00f0ff", width=2)
        c.create_oval(cx - 4, cy - 4, cx + 4, cy + 4, fill="#00f0ff", outline="")

    def _draw_gauges_ui(self, speed_kmh, throttle_val):
        # Clear existing drawings
        self.gauges_canvas.delete("all")
        
        # A. Speedometer Gauge (Left side)
        # Background arc
        self.gauges_canvas.create_arc(15, 12, 105, 102, start=30, extent=120, style='arc', width=7, outline='#181c26')
        # Value arc
        speed_extent = min(120.0, (speed_kmh / 100.0) * 120.0)
        self.gauges_canvas.create_arc(15, 12, 105, 102, start=150, extent=-speed_extent, style='arc', width=7, outline='#00f0ff')
        # Text values
        self.gauges_canvas.create_text(60, 52, text=f"{speed_kmh:.0f}", fill="#ffffff", font=("Outfit", 18, "bold"))
        self.gauges_canvas.create_text(60, 75, text="km/h", fill="#8892b0", font=("Inter", 9, "bold"))
        
        # B. Throttle Gauge (Right side)
        # Background arc
        self.gauges_canvas.create_arc(150, 12, 240, 102, start=30, extent=120, style='arc', width=7, outline='#181c26')
        # Value arc
        throttle_extent = min(120.0, throttle_val * 120.0)
        self.gauges_canvas.create_arc(150, 12, 240, 102, start=150, extent=-throttle_extent, style='arc', width=7, outline='#ff0066')
        # Text values
        self.gauges_canvas.create_text(195, 52, text=f"{throttle_val*100:.0f}", fill="#ffffff", font=("Outfit", 18, "bold"))
        self.gauges_canvas.create_text(195, 75, text="THROTTLE %", fill="#8892b0", font=("Inter", 9, "bold"))
        
    # ==========================================================================
    # CARLA CLIENT COMMUNICATIONS
    # ==========================================================================
    def connect_carla(self):
        try:
            self.btn_connect.configure(text="CONNECTING...", state="disabled")
            self.update_idletasks()
            
            self.client = carla.Client("localhost", 2000)
            self.client.set_timeout(5.0)
            self.world = self.client.get_world()
            self.carla_map = self.world.get_map()
            
            self.btn_connect.configure(text="CONNECTED!", fg_color="transparent", border_color="#28a745", text_color="#28a745")
            self.btn_spawn.configure(state="normal")
            self.status_dot.configure(text="🟢 CONNECTED", text_color="#00ff66")
            
        except Exception as e:
            self.btn_connect.configure(text="CONNECT FAILED", state="normal", fg_color="transparent", border_color="#dc3545", text_color="#dc3545")
            self.status_dot.configure(text="🔴 DISCONNECTED", text_color="#ff0055")
            tk.messagebox.showerror("Connection Error", f"Failed to connect to CARLA Server:\n{e}")
            
    def spawn_vehicle(self):
        try:
            self._cleanup_sensors()
            if self.ego_vehicle and self.ego_vehicle.is_alive:
                self.ego_vehicle.destroy()
            
            blueprint_library = self.world.get_blueprint_library()
            ego_bp = blueprint_library.find('vehicle.tesla.model3')
            ego_bp.set_attribute('role_name', 'ego')
            
            spawn_points = self.carla_map.get_spawn_points()
            spawn_point = np.random.choice(spawn_points) if spawn_points else carla.Transform()
            
            self.ego_vehicle = self.world.spawn_actor(ego_bp, spawn_point)
            
            # Setup Sensors
            self._setup_camera()
            self._setup_radar()
            self._setup_collision()
            
            self.btn_spawn.configure(text="SPAWNED!", fg_color="transparent", border_color="#138496", text_color="#17a2b8")
            
        except Exception as e:
            tk.messagebox.showerror("Spawn Error", f"Failed to spawn Tesla vehicle:\n{e}")
            
    def _setup_camera(self):
        blueprint_library = self.world.get_blueprint_library()
        cam_bp = blueprint_library.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', '640')
        cam_bp.set_attribute('image_size_y', '360')
        cam_bp.set_attribute('fov', '90')
        cam_bp.set_attribute('sensor_tick', '0.033') # 30 FPS stream
        
        self.update_camera_transform(cam_bp)
        
    def update_camera_transform(self, blueprint=None):
        transforms = {
            "Chase": carla.Transform(carla.Location(x=-5.5, y=0.0, z=2.8), carla.Rotation(pitch=-15.0)),
            "Driver": carla.Transform(carla.Location(x=0.8, y=-0.3, z=1.2), carla.Rotation(pitch=-5.0)),
            "Top": carla.Transform(carla.Location(x=0.0, y=0.0, z=15.0), carla.Rotation(pitch=-90.0))
        }
        
        if self.camera_sensor and self.camera_sensor.is_alive:
            self.camera_sensor.destroy()
            self.camera_sensor = None
            
        if not self.ego_vehicle:
            return
            
        if not blueprint:
            blueprint = self.world.get_blueprint_library().find('sensor.camera.rgb')
            blueprint.set_attribute('image_size_x', '640')
            blueprint.set_attribute('image_size_y', '360')
            blueprint.set_attribute('fov', '90')
            blueprint.set_attribute('sensor_tick', '0.033')
            
        self.camera_sensor = self.world.spawn_actor(blueprint, transforms[self.camera_view], attach_to=self.ego_vehicle)
        self.camera_sensor.listen(self._on_camera_frame)
        
    def _setup_radar(self):
        blueprint_library = self.world.get_blueprint_library()
        radar_bp = blueprint_library.find('sensor.other.radar')
        radar_bp.set_attribute('horizontal_fov', '30')
        radar_bp.set_attribute('vertical_fov', '5')
        radar_bp.set_attribute('range', '100')
        radar_transform = carla.Transform(carla.Location(x=2.0, z=1.0))
        self.radar_sensor = self.world.spawn_actor(radar_bp, radar_transform, attach_to=self.ego_vehicle)
        self.radar_sensor.listen(self._on_radar_data)
        
    def _setup_collision(self):
        blueprint_library = self.world.get_blueprint_library()
        col_bp = blueprint_library.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(col_bp, carla.Transform(), attach_to=self.ego_vehicle)
        self.collision_sensor.listen(self._on_collision)

        lane_bp = blueprint_library.find('sensor.other.lane_invasion')
        self.lane_invasion_sensor = self.world.spawn_actor(lane_bp, carla.Transform(), attach_to=self.ego_vehicle)
        self.lane_invasion_sensor.listen(self._on_lane_invasion)
        
    # ==========================================================================
    # SENSOR DATA EVENTS & PROCESSING
    # ==========================================================================
    def _on_camera_frame(self, image):
        try:
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            bgr = array[:, :, :3].copy()
            
            # --- HUD OVERLAYS ON CAMERA IMAGE ---
            cv2.putText(bgr, f"GEAR: {'R' if self.is_reverse else 'N' if self.speed < 0.1 else 'D'}", (20, 340), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 240, 255), 1, cv2.LINE_AA)
            cv2.putText(bgr, f"{self.speed * 3.6:.1f} km/h", (80, 340), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 240, 255), 1, cv2.LINE_AA)
            
            # Render HUD log stats on frame bottom right
            log_text = f"• REC {self.rows_saved} rows" if self.is_recording else ""
            cv2.putText(bgr, log_text, (image.width - 120, 340), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
            
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            
            self.after(0, self._render_tk_image, pil_image)
            
        except Exception as e:
            import traceback
            print(f"CAMERA CALLBACK ERROR: {e}")
            traceback.print_exc()
            
    def _render_tk_image(self, pil_image):
        self.current_tk_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(640, 360))
        self.camera_label.configure(image=self.current_tk_image, text="")
        
    def _on_radar_data(self, radar_data):
        min_dist = 100.0
        target_rel_speed = 0.0
        for detection in radar_data:
            if abs(detection.azimuth) < 0.1:
                if detection.depth < min_dist:
                    min_dist = detection.depth
                    target_rel_speed = detection.velocity
                    
        self.distance_to_lead = min_dist
        if target_rel_speed < -0.1:
            self.radar_ttc = min_dist / abs(target_rel_speed)
        else:
            self.radar_ttc = 99.9
            
    def _on_collision(self, event):
        # Latched for one CSV row / GUI refresh so the event doesn't get missed
        # between two logging ticks; the flag is cleared right after it's read.
        self.collision_flag = 1

    def _on_lane_invasion(self, event):
        self.lane_invasion_flag = 1
        
    # ==========================================================================
    # CONTROL LOGIC & INPUT BINDINGS
    # ==========================================================================
    def _on_key_press(self, event):
        if event.keysym in self.keys:
            self.keys[event.keysym] = True
        if event.keysym.lower() == "r":
            self.is_reverse = not self.is_reverse
            print(f"Gear Shift: Reverse={'Active' if self.is_reverse else 'Inactive'}")
            
    def _on_key_release(self, event):
        if event.keysym in self.keys:
            self.keys[event.keysym] = False
            
    def update_loop(self):
        try:
            if self.world and self.ego_vehicle and self.ego_vehicle.is_alive:
                v = self.ego_vehicle.get_velocity()
                self.speed = math.sqrt(v.x**2 + v.y**2 + v.z**2)
                
                location = self.ego_vehicle.get_location()
                waypoint = self.carla_map.get_waypoint(location)
                wp_loc = waypoint.transform.location
                wp_to_veh = location - wp_loc
                wp_forward = waypoint.transform.get_forward_vector()
                cross_z = (wp_forward.x * wp_to_veh.y) - (wp_forward.y * wp_to_veh.x)
                dist = math.sqrt(wp_to_veh.x**2 + wp_to_veh.y**2)
                self.lane_offset = dist if cross_z > 0 else -dist
                
                # 2. Map manual controls from keys
                manual_throttle = 1.0 if self.keys["Up"] else 0.0
                manual_brake = 1.0 if (self.keys["Down"] or self.keys["space"]) else 0.0
                
                manual_steer = 0.0
                if self.keys["Left"]:
                    manual_steer = -0.55
                elif self.keys["Right"]:
                    manual_steer = 0.55
                    
                # 2b. Count nearby vehicles/pedestrians (cheap, done every tick;
                # for larger worlds this could be throttled to every N ticks).
                actors = self.world.get_actors()
                self.num_vehicles = len(actors.filter('vehicle.*')) - 1  # exclude ego
                self.num_pedestrians = len(actors.filter('walker.pedestrian.*'))

                # 3. Evaluate safety limits (advisory -- see safety_advisor.py)
                adaptation = self.safety_monitor.evaluate_intent(
                    pred_throttle=manual_throttle,
                    pred_brake=manual_brake,
                    distance_to_lead=self.distance_to_lead,
                    radar_ttc=self.radar_ttc,
                    lane_offset=self.lane_offset,
                    speed_kmh=self.speed * 3.6,
                    steer=manual_steer,
                    rain=self.rain_flag,
                    fog=self.fog_flag,
                    num_vehicles=max(self.num_vehicles, 0),
                    num_pedestrians=self.num_pedestrians
                )
                
                # Reset safety variables to ensure zero overrides
                self.throttle_limit = 1.0
                self.brake_assist = 0.0
                self.steer_gain = 1.0
                self.lka_active = False
                
                self.steering = manual_steer
                self.throttle = manual_throttle
                self.brake = manual_brake
                
                # Apply control inputs to actuators (100% manual control)
                control = carla.VehicleControl(
                    throttle=float(manual_throttle),
                    steer=float(manual_steer),
                    brake=float(manual_brake),
                    reverse=self.is_reverse
                )
                self.ego_vehicle.apply_control(control)
                
                # Teleport spectator to track vehicle
                try:
                    spectator = self.world.get_spectator()
                    transform = self.ego_vehicle.get_transform()
                    forward = transform.get_forward_vector()
                    spec_loc = location - forward * 10.0 + carla.Location(z=4.0)
                    spec_rot = carla.Rotation(pitch=-20.0, yaw=transform.rotation.yaw, roll=0.0)
                    spectator.set_transform(carla.Transform(spec_loc, spec_rot))
                except Exception:
                    pass
                
                # 4. Save record to local logs array if recording is active
                if self.is_recording:
                    self._record_telemetry_row()
                    
                # 5. Update GUI elements
                self._update_gui_data(adaptation)
                
        except Exception as e:
            print(f"Error in control loop: {e}")
            
        self.after(50, self.update_loop)
        
    def _update_gui_data(self, adaptation):
        speed_kmh = self.speed * 3.6
        
        # 1. Update sliders
        self.slider_throttle.set(self.throttle)
        self.lbl_slider_throttle.configure(text=f"{self.throttle:.2f}")
        
        self.slider_brake.set(self.brake)
        self.lbl_slider_brake.configure(text=f"{self.brake:.2f}")
        
        self.slider_steer.set(self.steering)
        self.lbl_slider_steer.configure(text=f"{self.steering*100:+.0f} %")
        
        # 2. Update semicircular gauges
        self._draw_gauges_ui(speed_kmh, self.throttle)
        
        # 3. Update the rotating steering-wheel indicator (replaces WASD highlights)
        self._draw_wheel_indicator(self.steering)

        # 4. Update data log labels
        self.lbl_saved_rows.configure(text=f"{self.rows_saved} rows saved")
        
        # 5. Update All Parameters Status table
        self.param_labels["Actuator Throttle"].configure(text=f"{self.throttle * 100:.1f} %")
        self.param_labels["Actuator Brake"].configure(text=f"{self.brake * 100:.1f} %")
        self.param_labels["Actuator Steering"].configure(text=f"{self.steering * 100:+.1f} %")
        
        gear_text = "Reverse (R)" if self.is_reverse else "Drive (D)"
        self.param_labels["Active Gear"].configure(text=gear_text)
        
        self.param_labels["Lane Offset"].configure(text=f"{self.lane_offset:.2f} m")
        
        ttc_text = f"{self.radar_ttc:.1f} s" if self.radar_ttc < 99.0 else "N/A"
        self.param_labels["Time-to-Collision"].configure(text=ttc_text)
        
        follow_dist = adaptation.get("safe_follow_distance")
        self.param_labels["Safe Follow Distance"].configure(
            text=f"{follow_dist:.1f} m" if follow_dist is not None else "N/A")
        
        risk = adaptation.get("risk_probability")
        if risk is not None:
            risk_color = "#00ff66" if risk < 0.45 else ("#ff9900" if risk < 0.75 else "#ff0055")
            self.param_labels["AI Risk Score"].configure(text=f"{risk*100:.0f} %", text_color=risk_color)
        else:
            self.param_labels["AI Risk Score"].configure(text="N/A (untrained)", text_color="#8892b0")
        
        self.param_labels["AI Comfort Mode"].configure(text=adaptation["comfort_mode"])
        
        intervention = adaptation["intervention_level"]
        self.param_labels["AI Safety Limit"].configure(text=f"Capped at {int(adaptation['throttle_limit']*100)}%" if intervention != "None" else "None")
        self.param_labels["AI Steering Gain"].configure(text=f"{adaptation['steer_gain']:.2f}")
        
        weather_note = adaptation.get("weather_note", "")
        self.param_labels["Weather Advisory"].configure(text=weather_note[:34] if weather_note else "Clear conditions")
        
        threat_level = "LOW"
        threat_color = "#00ff66"
        if adaptation["intervention_level"] == "High" or adaptation["aeb_triggered"]:
            threat_level = "CRITICAL"
            threat_color = "#ff0055"
        elif adaptation["intervention_level"] == "Moderate" or adaptation["ttc_warning"]:
            threat_level = "WARNING"
            threat_color = "#ff9900"
            
        self.param_labels["System Threat"].configure(text=threat_level, text_color=threat_color)
        
    # ==========================================================================
    # USER CONTROLS ACTIONS
    # ==========================================================================
    def set_camera_view(self, val):
        self.camera_view = val
        
        # Visual indicator update on buttons
        for name, btn in self.cam_buttons.items():
            if name == val:
                btn.configure(fg_color="#ff0066")
            else:
                btn.configure(fg_color="#181c26")
                
        if self.world and self.ego_vehicle and self.ego_vehicle.is_alive:
            self.update_camera_transform()
            print(f"Camera perspective set to: {val}")
            
    def set_weather(self, val):
        self.weather_name = val
        # rain/fog flags feed both the CSV log schema and the safety model's
        # weather-aware grip/visibility adjustments.
        self.rain_flag = 1 if val == "Rainy" else 0
        self.fog_flag = 1 if val == "Foggy" else 0
        
        # Visual indicator update on buttons
        for name, btn in self.weather_buttons.items():
            if name == val:
                btn.configure(fg_color="#00f0ff", text_color="#000000")
            else:
                btn.configure(fg_color="#181c26", text_color="#ffffff")
                
        if self.world:
            presets = {
                "Sunny": carla.WeatherParameters.ClearNoon,
                "Rainy": carla.WeatherParameters.HardRainNoon,
                "Foggy": carla.WeatherParameters(
                    cloudiness=80.0,
                    precipitation=0.0,
                    precipitation_deposits=0.0,
                    wind_intensity=10.0,
                    sun_azimuth_angle=0.0,
                    sun_altitude_angle=45.0,
                    fog_density=80.0,
                    fog_distance=2.0,
                    fog_falloff=0.2,
                    wetness=0.0
                )
            }
            self.world.set_weather(presets[val])
            print(f"CARLA Weather set to: {val}")
            
    def toggle_logger(self):
        self.is_recording = not self.is_recording
        if self.is_recording:
            custom_name = self.entry_filename.get().strip()
            if not custom_name.endswith(".csv"):
                custom_name += ".csv"
            self.log_file = custom_name if custom_name else "driving_data.csv"
            
            self.entry_filename.configure(state="disabled")
            self.btn_record.configure(text="STOP DATA LOGGER", fg_color="#dc3545", hover_color="#bd2130")
            
            self.lbl_log_file.configure(text=f"→ {self.log_file}")
            
            # Continuous logging: if the file already exists (e.g. from a
            # previous session), keep appending to it and carry the running
            # row count forward instead of resetting/overwriting it.
            if os.path.exists(self.log_file):
                with open(self.log_file, "r", newline="") as f:
                    self.rows_saved = max(sum(1 for _ in f) - 1, 0)  # minus header
            else:
                self.rows_saved = 0
                with open(self.log_file, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "timestamp", "speed_kmh", "throttle", "brake", "steer",
                        "weather", "rain", "fog", "num_vehicles", "num_pedestrians",
                        "camera_view", "collision", "lane_invasion",
                        "pos_x", "pos_y", "pos_z", "yaw"
                    ])
        else:
            self.entry_filename.configure(state="normal")
            self.btn_record.configure(text="START DATA LOGGER", fg_color="#6f42c1", hover_color="#5a32a3")
            
    def _record_telemetry_row(self):
        try:
            location = self.ego_vehicle.get_location() if self.ego_vehicle else None
            yaw = self.ego_vehicle.get_transform().rotation.yaw if self.ego_vehicle else 0.0
            with open(self.log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    time.strftime("%H:%M.%S"),
                    round(self.speed * 3.6, 2),      # speed_kmh
                    round(self.throttle, 2),
                    round(self.brake, 2),
                    round(self.steering, 2),
                    self.weather_name,
                    self.rain_flag,
                    self.fog_flag,
                    max(self.num_vehicles, 0),
                    self.num_pedestrians,
                    self.camera_view,
                    self.collision_flag,
                    self.lane_invasion_flag,
                    round(location.x, 3) if location else 0.0,
                    round(location.y, 3) if location else 0.0,
                    round(location.z, 3) if location else 0.0,
                    round(yaw, 2)
                ])
            self.rows_saved += 1
            # Hazard flags are per-event -- clear them once logged so the next
            # row doesn't keep reporting a collision/lane-invasion that already happened.
            self.collision_flag = 0
            self.lane_invasion_flag = 0
        except Exception:
            pass
            
    # ==========================================================================
    # CLEANUP & SHUTDOWN ROUTINES
    # ==========================================================================
    def _cleanup_sensors(self):
        sensors = [self.camera_sensor, self.radar_sensor, self.collision_sensor, self.lane_invasion_sensor]
        for sensor in sensors:
            if sensor and sensor.is_alive:
                sensor.destroy()
        self.camera_sensor = None
        self.radar_sensor = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        
    def destroy(self):
        print("Cleaning up vehicle and sensors...")
        self._cleanup_sensors()
        if self.ego_vehicle and self.ego_vehicle.is_alive:
            self.ego_vehicle.destroy()
        super().destroy()

if __name__ == "__main__":
    app = CarlaNativeDashboard()
    app.mainloop()