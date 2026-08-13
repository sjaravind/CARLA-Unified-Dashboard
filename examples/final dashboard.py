import os
import csv
import time
import math
import cv2
import numpy as np
import threading
import tkinter as tk
from collections import deque
from PIL import Image, ImageTk
import customtkinter as ctk
import carla

from adaptation import PredictiveSafetyMonitor

# Configure CustomTkinter Theme
ctk.set_appearance_mode("Dark")

class CarlaNativeDashboard(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Window Configuration
        self.title("CARLA Minimal Telemetry Dashboard")
        self.geometry("980x480")
        self.resizable(False, False)
        self.configure(fg_color="#07080b")
        
        # Client states
        self.client = None
        self.world = None
        self.carla_map = None
        self.ego_vehicle = None
        self.camera_sensor = None
        self.radar_sensor = None
        self.collision_sensor = None
        self.traffic_vehicles = []
        
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
        
        # Smooth keyboard control interpolation states
        self.keyboard_throttle = 0.0
        self.keyboard_reverse = 0.0
        self.keyboard_space_brake = 0.0
        self.keyboard_steer = 0.0
        
        # Safety / Adaptation configurations
        self.safety_monitor = PredictiveSafetyMonitor()
        self.lka_active = False
        self.throttle_limit = 1.0
        self.steer_gain = 1.0
        self.brake_assist = 0.0
        self.sensitivity_mode = "Standard"
        
        # Logging & History states
        self.active_trial_data = []
        self.log_file = "driving_data.csv"
        self.is_recording = True
        self.rows_saved = 0
        self.camera_view = "Chase"
        
        # Keyboard controls state
        self.keys = {
            "Up": False,
            "Down": False,
            "Left": False,
            "Right": False,
            "space": False
        }
        
        # Bind keys
        self.bind("<KeyPress>", self._on_key_press)
        self.bind("<KeyRelease>", self._on_key_release)
        
        # Create UI
        self._build_ui()
        
        # Start main loop
        self.after(50, self.update_loop)

    def _build_ui(self):
        # Configure Grid: 2 columns
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=7) # Left Panel (Camera + Alerts)
        self.grid_columnconfigure(1, weight=3) # Right Panel (Telemetry + Settings)
        
        # ======================================================================
        # LEFT PANEL: VISUALIZER & OVERLAYS FRAME
        # ======================================================================
        self.visualizer_frame = ctk.CTkFrame(self, corner_radius=15, border_width=2, border_color="#181c26", fg_color="#0c0e14")
        self.visualizer_frame.grid(row=0, column=0, padx=15, pady=15, sticky="nsew")
        
        # Visualizer Camera View Label
        self.camera_label = ctk.CTkLabel(self.visualizer_frame, text="WAITING FOR CARLA CLIENT CONNECTION...", 
                                         font=("Consolas", 14), fg_color="#020308", text_color="#8892b0")
        self.camera_label.pack(fill="both", expand=True, padx=10, pady=(5, 5))
        
        # Bottom info bar inside visualizer frame
        self.info_panel = ctk.CTkFrame(self.visualizer_frame, fg_color="transparent")
        self.info_panel.pack(fill="x", side="bottom", padx=15, pady=(5, 10))
        
        # Data Log Text Status
        self.log_status_frame = ctk.CTkFrame(self.info_panel, fg_color="transparent")
        self.log_status_frame.pack(side="left", padx=15)
        
        ctk.CTkLabel(self.log_status_frame, text="DATA LOGGER", font=("Inter", 10, "bold"), text_color="#ff9900").pack(anchor="w")
        self.lbl_saved_rows = ctk.CTkLabel(self.log_status_frame, text="0 rows saved", font=("Consolas", 11), text_color="#00f0ff")
        self.lbl_saved_rows.pack(anchor="w")
        self.lbl_log_file = ctk.CTkLabel(self.log_status_frame, text="→ driving_data.csv", font=("Consolas", 10), text_color="#8892b0")
        self.lbl_log_file.pack(anchor="w")
        
        # Connected status indicator
        self.conn_status_frame = ctk.CTkFrame(self.info_panel, fg_color="transparent")
        self.conn_status_frame.pack(side="right", padx=15)
        self.status_dot = ctk.CTkLabel(self.conn_status_frame, text="🔴 DISCONNECTED", font=("Inter", 11, "bold"), text_color="#ff0055")
        self.status_dot.pack()
        
        # ======================================================================
        # RIGHT PANEL: SYSTEM & GAUGES SIDEBAR
        # ======================================================================
        self.right_panel = ctk.CTkFrame(self, fg_color="transparent")
        self.right_panel.grid(row=0, column=1, padx=(0, 15), pady=15, sticky="nsew")
        
        # Connect & Spawn Actions Row
        self.actions_frame = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        self.actions_frame.pack(fill="x", pady=(5, 10))
        
        self.btn_connect = ctk.CTkButton(self.actions_frame, text="CONNECT", font=("Inter", 11, "bold"), height=30,
                                         fg_color="transparent", border_width=2, border_color="#28a745", text_color="#28a745", 
                                         hover_color="#0a2512", command=self.connect_carla)
        self.btn_connect.pack(side="left", fill="x", expand=True, padx=(0, 4))
        
        self.btn_spawn = ctk.CTkButton(self.actions_frame, text="SPAWN", font=("Inter", 11, "bold"), height=30,
                                       state="disabled", fg_color="transparent", border_width=2, border_color="#17a2b8", 
                                       text_color="#17a2b8", hover_color="#09212b", command=self.spawn_vehicle)
        self.btn_spawn.pack(side="right", fill="x", expand=True, padx=(4, 0))
        
        # Telemetry Canvas (Steering Wheel + Throttle/Brake bar animations)
        self.telemetry_canvas = tk.Canvas(self.right_panel, width=260, height=270, bg="#0c0e14", highlightthickness=1, highlightbackground="#222533")
        self.telemetry_canvas.pack(pady=5)
        
        # Camera & Weather Selectors row
        self.selectors_row = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        self.selectors_row.pack(fill="x", pady=5)
        
        self.camera_select = ctk.CTkOptionMenu(self.selectors_row, values=["Chase", "Driver", "Top"], 
                                               fg_color="#181c26", button_color="#2c3144", button_hover_color="#ff9f00",
                                               command=self.set_camera_view, height=25, font=("Inter", 10, "bold"))
        self.camera_select.set("Chase")
        self.camera_select.pack(side="left", fill="x", expand=True, padx=(0, 4))
        
        self.weather_select = ctk.CTkOptionMenu(self.selectors_row, values=["Sunny", "Rainy", "Foggy"],
                                                fg_color="#181c26", button_color="#2c3144", button_hover_color="#ff9f00",
                                                command=self.set_weather, height=25, font=("Inter", 10, "bold"))
        self.weather_select.set("Sunny")
        self.weather_select.pack(side="right", fill="x", expand=True, padx=(4, 0))
        
        # Traffic row
        self.traffic_row = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        self.traffic_row.pack(fill="x", pady=5)
        
        self.btn_spawn_traffic = ctk.CTkButton(self.traffic_row, text="Spawn Traffic", font=("Inter", 10, "bold"), height=25,
                                               fg_color="#181c26", hover_color="#ff9f00", command=self.spawn_traffic)
        self.btn_spawn_traffic.pack(side="left", fill="x", expand=True, padx=(0, 4))
        
        self.btn_clear_traffic = ctk.CTkButton(self.traffic_row, text="Clear Traffic", font=("Inter", 10, "bold"), height=25,
                                               fg_color="#181c26", hover_color="#ff0055", command=self.clear_traffic)
        self.btn_clear_traffic.pack(side="right", fill="x", expand=True, padx=(4, 0))
        
        # Logger filename input
        self.logger_row = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        self.logger_row.pack(fill="x", pady=5)
        
        self.entry_filename = ctk.CTkEntry(self.logger_row, placeholder_text="driving_data.csv", font=("Inter", 11), height=25, fg_color="#080a0f")
        self.entry_filename.insert(0, "driving_data.csv")
        self.entry_filename.pack(side="left", fill="x", expand=True, padx=(0, 4))
        
        self.btn_record = ctk.CTkButton(self.logger_row, text="REC", font=("Inter", 10, "bold"), height=25, width=60,
                                        fg_color="#6f42c1", hover_color="#5a32a3", command=self.toggle_logger)
        self.btn_record.pack(side="right", padx=(4, 0))
        
        # Initial canvas draws
        self._draw_telemetry_ui(0.0, 0.0, 0.0, 0.0)

    def _draw_telemetry_ui(self, speed_kmh, steer_val, throttle_val, brake_val):
        self.telemetry_canvas.delete("all")
        
        # 1. Digital Speedometer (Center Top)
        self.telemetry_canvas.create_text(130, 30, text="{:.0f}".format(speed_kmh), fill="#ffffff", font=("Outfit", 26, "bold"))
        self.telemetry_canvas.create_text(130, 48, text="km/h", fill="#8892b0", font=("Inter", 8, "bold"))
        
        # 2. Interactive Steering Wheel (Center Middle)
        cx, cy = 130, 125
        r = 45
        theta = steer_val * (math.pi / 2) # steering rotation mapping
        
        # Outer Rim
        self.telemetry_canvas.create_oval(cx - r, cy - r, cx + r, cy + r, width=7, outline='#1c2130')
        self.telemetry_canvas.create_arc(cx - r, cy - r, cx + r, cy + r, start=int(math.degrees(-theta) + 30), extent=120, style='arc', width=7, outline='#00f0ff')
        
        # Spokes
        x_left = cx + r * math.cos(theta + math.pi)
        y_left = cy + r * math.sin(theta + math.pi)
        self.telemetry_canvas.create_line(cx, cy, x_left, y_left, width=5, fill='#8892b0')
        
        x_right = cx + r * math.cos(theta)
        y_right = cy + r * math.sin(theta)
        self.telemetry_canvas.create_line(cx, cy, x_right, y_right, width=5, fill='#8892b0')
        
        x_bottom = cx + r * math.cos(theta + math.pi/2)
        y_bottom = cy + r * math.sin(theta + math.pi/2)
        self.telemetry_canvas.create_line(cx, cy, x_bottom, y_bottom, width=5, fill='#8892b0')
        
        # Center Hub
        self.telemetry_canvas.create_oval(cx - 9, cy - 9, cx + 9, cy + 9, fill='#0c0e14', outline='#ff0066', width=2)
        
        # 12 o'clock center marker
        x_mark_start = cx + (r - 7) * math.cos(theta - math.pi/2)
        y_mark_start = cy + (r - 7) * math.sin(theta - math.pi/2)
        x_mark_end = cx + r * math.cos(theta - math.pi/2)
        y_mark_end = cy + r * math.sin(theta - math.pi/2)
        self.telemetry_canvas.create_line(x_mark_start, y_mark_start, x_mark_end, y_mark_end, width=4, fill='#ff0066')
        
        # Steering readout text
        steer_pct = abs(steer_val) * 100
        if abs(steer_val) < 0.02:
            steer_text = "CENTERED"
            steer_color = "#00ff66"
        elif steer_val < 0:
            steer_text = "LEFT {:.0f}%".format(steer_pct)
            steer_color = "#00f0ff"
        else:
            steer_text = "RIGHT {:.0f}%".format(steer_pct)
            steer_color = "#ff0066"
        self.telemetry_canvas.create_text(cx, cy + 56, text=steer_text, fill=steer_color, font=("Consolas", 9, "bold"))
        
        # 3. Cool Throttle & Brake Animation Bars (Bottom)
        bar_w = 16
        bar_h = 50
        y_bottom = 245
        
        # Throttle Bar (Left Side)
        tx = 60
        self.telemetry_canvas.create_text(tx + bar_w/2, y_bottom - bar_h - 10, text="THR", fill="#8892b0", font=("Inter", 8, "bold"))
        # Track
        self.telemetry_canvas.create_rectangle(tx, y_bottom - bar_h, tx + bar_w, y_bottom, fill="#1c2130", outline="")
        # Level Animation
        level_t = throttle_val * bar_h
        if level_t > 0:
            self.telemetry_canvas.create_rectangle(tx, y_bottom - level_t, tx + bar_w, y_bottom, fill="#00ff66", outline="")
        # Percent value
        self.telemetry_canvas.create_text(tx + bar_w/2, y_bottom + 12, text="{:.0f}%".format(throttle_val*100), fill="#ffffff", font=("Consolas", 9, "bold"))
        
        # Brake Bar (Right Side)
        bx = 180
        self.telemetry_canvas.create_text(bx + bar_w/2, y_bottom - bar_h - 10, text="BRK", fill="#8892b0", font=("Inter", 8, "bold"))
        # Track
        self.telemetry_canvas.create_rectangle(bx, y_bottom - bar_h, bx + bar_w, y_bottom, fill="#1c2130", outline="")
        # Level Animation
        level_b = brake_val * bar_h
        if level_b > 0:
            self.telemetry_canvas.create_rectangle(bx, y_bottom - level_b, bx + bar_w, y_bottom, fill="#ff0055", outline="")
        # Percent value
        self.telemetry_canvas.create_text(bx + bar_w/2, y_bottom + 12, text="{:.0f}%".format(brake_val*100), fill="#ffffff", font=("Consolas", 9, "bold"))

    # ==========================================================================
    # CARLA CLIENT COMMUNICATIONS
    # ==========================================================================
    def connect_carla(self):
        try:
            self.client = carla.Client("127.0.0.1", 2000)
            self.client.set_timeout(4.0)
            self.world = self.client.get_world()
            self.carla_map = self.world.get_map()
            
            self.btn_connect.configure(text="CONNECTED", fg_color="transparent", border_color="#28a745", text_color="#28a745")
            self.btn_spawn.configure(state="normal")
            self.status_dot.configure(text="🟢 CONNECTED", text_color="#28a745")
            print("Connected to CARLA simulator mapping town maps.")
            
        except Exception as e:
            print("Connection failure: " + str(e))
            tk.messagebox.showerror("Error", "Failed to connect to CARLA:\n" + str(e))

    def spawn_vehicle(self):
        if not self.world:
            return
            
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
            
            self.btn_spawn.configure(text="SPAWNED", fg_color="transparent", border_color="#17a2b8", text_color="#17a2b8")
            print("Ego vehicle spawned.")
            
        except Exception as e:
            tk.messagebox.showerror("Spawn Error", "Failed to spawn Tesla vehicle:\n" + str(e))

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
            
        self.camera_sensor = self.world.spawn_actor(blueprint, transforms[self.camera_view], attach_to=self.ego_vehicle)
        self.camera_sensor.listen(self._on_camera_frame)

    def _setup_radar(self):
        blueprint_library = self.world.get_blueprint_library()
        radar_bp = blueprint_library.find('sensor.other.radar')
        radar_bp.set_attribute('horizontal_fov', '30')
        radar_bp.set_attribute('vertical_fov', '10')
        radar_bp.set_attribute('range', '100')
        radar_bp.set_attribute('sensor_tick', '0.05')
        
        self.radar_sensor = self.world.spawn_actor(radar_bp, carla.Transform(carla.Location(x=2.0, z=0.5)), attach_to=self.ego_vehicle)
        self.radar_sensor.listen(self._on_radar_data)

    def _setup_collision(self):
        blueprint_library = self.world.get_blueprint_library()
        col_bp = blueprint_library.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(col_bp, carla.Transform(), attach_to=self.ego_vehicle)
        self.collision_sensor.listen(self._on_collision)

    def _on_camera_frame(self, image):
        try:
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            bgr = array[:, :, :3].copy()
            
            # Simple minimal overlay
            cv2.putText(bgr, "GEAR: {}".format('R' if self.is_reverse else 'N' if self.speed < 0.1 else 'D'), (20, 340), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 240, 255), 1, cv2.LINE_AA)
            cv2.putText(bgr, "{:.1f} km/h".format(self.speed * 3.6), (80, 340), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 240, 255), 1, cv2.LINE_AA)
            
            log_text = "• REC {} rows".format(self.rows_saved) if self.is_recording else ""
            cv2.putText(bgr, log_text, (image.width - 120, 340), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
            
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            self.after(0, self._render_tk_image, pil_image)
            
        except Exception as e:
            print("Camera rendering error: " + str(e))

    def _render_tk_image(self, pil_image):
        tk_img = ImageTk.PhotoImage(pil_image)
        self.camera_label.configure(image=tk_img, text="")
        self.camera_label.image = tk_img

    def _on_radar_data(self, radar_data):
        min_dist = 100.0
        min_ttc = 99.9
        
        for detect in radar_data:
            dist = detect.depth
            vel = detect.velocity
            if dist < min_dist:
                min_dist = dist
                if vel < -0.1: # target closing in
                    min_ttc = dist / abs(vel)
                    
        self.distance_to_lead = min_dist
        self.radar_ttc = min_ttc

    def _on_collision(self, event):
        pass

    def _on_key_press(self, event):
        if event.keysym in self.keys:
            self.keys[event.keysym] = True

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
                
                # 2. Smoothly interpolate manual controls from keyboard inputs
                # Throttle smoothing (simulates progressive accelerator pedal)
                if self.keys["Up"]:
                    self.keyboard_throttle = min(1.0, self.keyboard_throttle + 0.03)
                else:
                    self.keyboard_throttle = max(0.0, self.keyboard_throttle - 0.05)
                    
                # Reverse throttle smoothing (Down key)
                if self.keys["Down"]:
                    self.keyboard_reverse = min(1.0, self.keyboard_reverse + 0.03)
                else:
                    self.keyboard_reverse = max(0.0, self.keyboard_reverse - 0.05)
                    
                # Spacebar Brake smoothing (simulates progressive brake pedal)
                if self.keys["space"]:
                    self.keyboard_space_brake = min(1.0, self.keyboard_space_brake + 0.06)
                else:
                    self.keyboard_space_brake = max(0.0, self.keyboard_space_brake - 0.10)
                    
                # Steering smoothing with auto-centering (prevents sudden tire slips)
                if self.keys["Left"]:
                    self.keyboard_steer = max(-0.55, self.keyboard_steer - 0.02)
                elif self.keys["Right"]:
                    self.keyboard_steer = min(0.55, self.keyboard_steer + 0.02)
                else:
                    # Return steering to center when keys are released
                    if self.keyboard_steer > 0.02:
                        self.keyboard_steer = max(0.0, self.keyboard_steer - 0.04)
                    elif self.keyboard_steer < -0.02:
                        self.keyboard_steer = min(0.0, self.keyboard_steer + 0.04)
                    else:
                        self.keyboard_steer = 0.0
                        
                # Direct control mapping: no auto-shifting or gear selector logic
                if self.keys["Up"]:
                    manual_throttle = self.keyboard_throttle
                    manual_brake = 0.0
                    self.is_reverse = False
                elif self.keys["Down"]:
                    manual_throttle = self.keyboard_reverse
                    manual_brake = 0.0
                    self.is_reverse = True
                else:
                    manual_throttle = 0.0
                    manual_brake = 0.0
                    
                # If space is pressed, override/add brake force
                if self.keys["space"]:
                    manual_brake = self.keyboard_space_brake
                    
                manual_steer = self.keyboard_steer
                    
                # 3. Evaluate safety limits (no-op monitor - pure manual control)
                adaptation = self.safety_monitor.evaluate_intent(
                    distance_to_lead=self.distance_to_lead,
                    radar_ttc=self.radar_ttc,
                    lane_offset=self.lane_offset,
                    actual_throttle=manual_throttle,
                    actual_brake=manual_brake,
                    sensitivity_mode=self.sensitivity_mode
                )
                
                # Apply controls directly
                self.steering = manual_steer
                self.throttle = manual_throttle
                self.brake = manual_brake
                
                # Send control actuator packet to CARLA vehicle
                control = carla.VehicleControl(
                    throttle=self.throttle,
                    steer=self.steering,
                    brake=self.brake,
                    hand_brake=False,
                    reverse=self.is_reverse
                )
                self.ego_vehicle.apply_control(control)
                
                # Auto telemetry logging
                if self.is_recording and (self.throttle > 0.01 or self.brake > 0.01 or abs(self.steering) > 0.01):
                    self._record_telemetry_row()
                    
                # Update GUI variables and canvas renders
                self._update_gui_data(adaptation)
                
        except Exception as e:
            print("Error in control loop: " + str(e))
            
        self.after(50, self.update_loop)

    def _update_gui_data(self, adaptation):
        speed_kmh = self.speed * 3.6
        
        # Update canvas drawings
        self._draw_telemetry_ui(speed_kmh, self.steering, self.throttle, self.brake)
                                    
        # Update logger stats label
        self.lbl_saved_rows.configure(text="{} rows saved".format(self.rows_saved))
        
        pass

    def set_camera_view(self, val):
        self.camera_view = val
        if self.world and self.ego_vehicle:
            self.update_camera_transform()
            print("Camera View set to: " + str(val))

    def set_weather(self, val):
        self.weather_name = val
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
                    fog_density=40.0,
                    fog_distance=15.0,
                    fog_falloff=0.2,
                    wetness=0.0
                )
            }
            self.world.set_weather(presets[val])
            print("CARLA Weather set to: " + str(val))
            
    def change_sensitivity(self, val):
        self.sensitivity_mode = val
        print("AI Safety sensitivity set to: " + str(val))
            
    def toggle_logger(self):
        self.is_recording = not self.is_recording
        if self.is_recording:
            custom_name = self.entry_filename.get().strip()
            if not custom_name.endswith(".csv") and custom_name:
                custom_name += ".csv"
            self.log_file = custom_name if custom_name else "driving_data.csv"
            
            self.entry_filename.configure(state="disabled")
            self.btn_record.configure(text="REC", fg_color="#dc3545", hover_color="#bd2130")
            self.rows_saved = 0
            self.lbl_log_file.configure(text="→ " + str(self.log_file))
            
            if not os.path.exists(self.log_file):
                with open(self.log_file, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Timestamp", "Speed (m/s)", "Throttle", "Brake", "Steer", "Lane Offset (m)", "Distance to Lead (m)"])
        else:
            self.entry_filename.configure(state="normal")
            self.btn_record.configure(text="REC", fg_color="#6f42c1", hover_color="#5a32a3")
            
    def _record_telemetry_row(self):
        try:
            if not os.path.exists(self.log_file):
                with open(self.log_file, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Timestamp", "Speed (m/s)", "Throttle", "Brake", "Steer", "Lane Offset (m)", "Distance to Lead (m)"])
                    
            with open(self.log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    round(self.speed, 2),
                    round(self.throttle, 2),
                    round(self.brake, 2),
                    round(self.steering, 2),
                    round(self.lane_offset, 2),
                    round(self.distance_to_lead, 2)
                ])
            self.rows_saved += 1
        except Exception:
            pass
            
    def spawn_traffic(self):
        if not self.world or not self.client:
            tk.messagebox.showerror("Error", "Connect to CARLA first!")
            return
            
        try:
            self.clear_traffic()
            traffic_manager = self.client.get_trafficmanager(8000)
            traffic_manager.set_global_distance_to_leading_vehicle(4.0)
            
            blueprint_library = self.world.get_blueprint_library()
            vehicle_blueprints = blueprint_library.filter('vehicle.*')
            
            spawn_points = self.carla_map.get_spawn_points()
            count = min(15, len(spawn_points) - 5)
            
            # Spawn loop
            for _ in range(count):
                bp = np.random.choice(vehicle_blueprints)
                spawn_pt = np.random.choice(spawn_points)
                
                vehicle = self.world.try_spawn_actor(bp, spawn_pt)
                if vehicle:
                    vehicle.set_autopilot(True)
                    self.traffic_vehicles.append(vehicle)
                    
            print("Spawned " + str(count) + " traffic vehicles successfully.")
            self.btn_spawn_traffic.configure(text="Spawned", fg_color="#ff9f00", text_color="#000000")
            
        except Exception as e:
            self.btn_spawn_traffic.configure(text="Spawn Traffic", state="normal", fg_color="#181c26", text_color="#ffffff")
            tk.messagebox.showerror("Traffic Error", "Failed to spawn traffic:\n" + str(e))

    def clear_traffic(self):
        if not self.traffic_vehicles:
            return
        print("Clearing " + str(len(self.traffic_vehicles)) + " traffic vehicles...")
        for vehicle in self.traffic_vehicles:
            if vehicle and vehicle.is_alive:
                vehicle.destroy()
        self.traffic_vehicles = []
        self.btn_spawn_traffic.configure(text="Spawn Traffic", state="normal", fg_color="#181c26", text_color="#ffffff")
        print("Traffic cleared.")

    def _cleanup_sensors(self):
        sensors = [self.camera_sensor, self.radar_sensor, self.collision_sensor]
        for sensor in sensors:
            if sensor and sensor.is_alive:
                sensor.destroy()
        self.camera_sensor = None
        self.radar_sensor = None
        self.collision_sensor = None
        
    def destroy(self):
        print("Cleaning up vehicle and sensors...")
        self.clear_traffic()
        self._cleanup_sensors()
        if self.ego_vehicle and self.ego_vehicle.is_alive:
            self.ego_vehicle.destroy()
        super().destroy()

if __name__ == "__main__":
    app = CarlaNativeDashboard()
    app.mainloop()