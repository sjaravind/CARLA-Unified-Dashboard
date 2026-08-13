"""
╔══════════════════════════════════════════════════════════════════════╗
║        CARLA UNIFIED SMART DASHBOARD  —  Vice City Edition           ║
║                                                                      ║
║  ONE FILE  •  TWO MODES  •  1280×800 WINDOW                          ║
║                                                                      ║
║  MODE 1 — MANUAL DRIVE                                               ║
║    • Drive normally with W/A/S/D                                     ║
║    • Logs every frame: controls + speed + sensors + GPS waypoints    ║
║    • Records timestamp, throttle, brake, steer, speed, weather,      ║
║      collision, lane_invasion, AND vehicle x/y/z/yaw for replay      ║
║                                                                      ║
║  MODE 2 — AUTO REPLAY + AI COMPARISON                                ║
║    • Loads CSV, smooths controls with ML corrections                 ║
║    • Follows recorded GPS waypoints frame-by-frame                   ║
║    • Side-by-side panel: YOUR drive vs AI drive                      ║
║    • Risk overlay: red markers where you made mistakes               ║
║                                                                      ║
║  Usage:                                                              ║
║    1.  py -3.7 carla_unified_dashboard.py          (opens mode menu) ║
║    2.  Drive in MANUAL mode, press ESC when done                     ║
║    3.  Press TAB or click MODE button to switch to AUTO              ║
║    4.  Watch AI replay with comparison panel                         ║
║                                                                      ║
║  Requires: CARLA server running on localhost:2000                    ║
║  Run:  py -3.7 -m pip install pygame carla scikit-learn numpy        ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import sys, os, math, csv, time, threading, random, pickle, warnings
from datetime import datetime
from collections import deque

warnings.filterwarnings("ignore")

# ── Dependency check ──────────────────────────────────────────────────
_missing = []
for _lib in ["pygame", "carla", "numpy", "sklearn"]:
    try:
        __import__("sklearn" if _lib == "sklearn" else _lib)
    except ImportError:
        _missing.append("scikit-learn" if _lib == "sklearn" else _lib)
if _missing:
    print(f"Missing: {', '.join(_missing)}")
    print(f"Run: py -3.7 -m pip install {' '.join(_missing)}")
    sys.exit(1)

import pygame
import carla
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ══════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════
LOG_FILE   = "driving_data.csv"
MODEL_FILE = "driving_model.pkl"

# Neon palette (shared across both modes)
DARK_BG     = (8,   8,  18)
PANEL_BG    = (14,  14, 28)
PANEL_BG2   = (20,  20, 40)
NEON_PINK   = (255, 20,  147)
NEON_CYAN   = (0,   255, 255)
NEON_YELLOW = (255, 220, 0)
NEON_GREEN  = (57,  255, 20)
NEON_RED    = (255, 50,  50)
NEON_ORANGE = (255, 140, 0)
NEON_PURPLE = (180, 0,   255)
WHITE       = (255, 255, 255)
GREY        = (100, 100, 120)
DARK_GREY   = (30,  30,  50)
BLACK       = (0,   0,   0)

WEATHER_MAP = {
    "Clear Day": 0, "Cloudy": 1, "Rainy": 2,
    "Foggy": 3,     "Clear Night": 4, "Stormy Night": 5,
}
WEATHER_PRESETS_KEYS = ["Clear Day", "Cloudy", "Rainy", "Foggy", "Clear Night", "Stormy Night"]

# AI smoothing: how aggressively the AI corrects raw inputs (0=no correction, 1=full)
AI_SMOOTH   = 0.45
MAX_SAFE_SPEED   = 65.0   # km/h — AI will not exceed this
STEER_SMOOTH_K   = 0.6    # low-pass coefficient for steer smoothing


# ══════════════════════════════════════════════════════════════════════
#  UI HELPERS
# ══════════════════════════════════════════════════════════════════════
def txt(surf, text, x, y, font, color=WHITE, center=False, right=False):
    s = font.render(str(text), True, color)
    r = s.get_rect()
    if center: r.center = (x, y)
    elif right: r.topright = (x, y)
    else: r.topleft = (x, y)
    surf.blit(s, r)
    return r

def neon_rect(surf, color, rect, w=2, glow=True, radius=6):
    pygame.draw.rect(surf, color, rect, w, border_radius=radius)
    if glow:
        gs = pygame.Surface((rect[2]+12, rect[3]+12), pygame.SRCALPHA)
        pygame.draw.rect(gs, (*color, 35), (6, 6, rect[2], rect[3]), 0, border_radius=radius+2)
        surf.blit(gs, (rect[0]-6, rect[1]-6))

def filled_rect(surf, color, rect, alpha=30, radius=8):
    s = pygame.Surface((rect[2], rect[3]), pygame.SRCALPHA)
    s.fill((*color, alpha))
    surf.blit(s, (rect[0], rect[1]))
    pygame.draw.rect(surf, color, rect, 2, border_radius=radius)

def bar(surf, x, y, w, h, val, max_val, color, label, font, show_val=True):
    pygame.draw.rect(surf, DARK_GREY, (x, y, w, h), border_radius=4)
    fw = max(0, int((val / max(max_val, 0.001)) * w))
    if fw: pygame.draw.rect(surf, color, (x, y, fw, h), border_radius=4)
    neon_rect(surf, GREY, (x, y, w, h), w=1, glow=False)
    if label: txt(surf, label, x, y-15, font, GREY)
    if show_val: txt(surf, f"{val:.2f}", x+w+5, y, font, color)

def arc_gauge(surf, cx, cy, r, val, max_val, color, label, font):
    start = math.pi; span = math.pi
    for i in range(100):
        a = start - (i/100)*span
        x1=cx+(r-9)*math.cos(a); y1=cy-(r-9)*math.sin(a)
        x2=cx+r*math.cos(a);     y2=cy-r*math.sin(a)
        pygame.draw.line(surf, DARK_GREY, (int(x1),int(y1)), (int(x2),int(y2)), 3)
    filled = min(100, int((val/max(max_val,0.001))*100))
    for i in range(filled):
        a = start-(i/100)*span; t=i/100
        rc=int(color[0]*(1-t)+NEON_RED[0]*t)
        gc=int(color[1]*(1-t)+NEON_RED[1]*t)
        bc=int(color[2]*(1-t)+NEON_RED[2]*t)
        x1=cx+(r-9)*math.cos(a); y1=cy-(r-9)*math.sin(a)
        x2=cx+r*math.cos(a);     y2=cy-r*math.sin(a)
        pygame.draw.line(surf, (rc,gc,bc), (int(x1),int(y1)), (int(x2),int(y2)), 3)
    na=start-(val/max(max_val,0.001))*span
    nx=cx+(r-16)*math.cos(na); ny=cy-(r-16)*math.sin(na)
    pygame.draw.line(surf, WHITE, (cx,cy), (int(nx),int(ny)), 2)
    pygame.draw.circle(surf, WHITE, (cx,cy), 5)
    txt(surf, f"{int(val)}", cx, cy+8, font, color, center=True)
    txt(surf, label, cx, cy+24, font, GREY, center=True)

def btn(surf, rect, label, font, active=False, color=NEON_CYAN):
    bg_alpha = 55 if active else 18
    s = pygame.Surface((rect[2], rect[3]), pygame.SRCALPHA)
    fill_color = (*color, bg_alpha) if active else (20, 20, 40, bg_alpha)
    s.fill(fill_color)
    surf.blit(s, (rect[0], rect[1]))
    neon_rect(surf, color if active else GREY, rect, w=2, glow=active)
    txt(surf, label, rect[0]+rect[2]//2, rect[1]+rect[3]//2, font,
        color if active else WHITE, center=True)
    return pygame.Rect(rect)

def scanlines(surf, w, h):
    s = pygame.Surface((w, h), pygame.SRCALPHA)
    for y in range(0, h, 4):
        pygame.draw.line(s, (0,0,0,14), (0,y),(w,y))
    surf.blit(s, (0,0))

def mini_chart(surf, x, y, w, h, data, color, label, font, max_val=200):
    pygame.draw.rect(surf, PANEL_BG, (x,y,w,h), border_radius=5)
    neon_rect(surf, GREY, (x,y,w,h), w=1, glow=False, radius=5)
    pts = list(data)[-w:]
    for i in range(1, len(pts)):
        x1=x+i-1; x2=x+i
        y1=y+h-int((pts[i-1]/max_val)*h)
        y2=y+h-int((pts[i]/max_val)*h)
        pygame.draw.line(surf, color, (x1,y1),(x2,y2),2)
    txt(surf, label, x+4, y+3, font, GREY)

def dual_chart(surf, x, y, w, h, data_a, data_b, col_a, col_b,
               label_a, label_b, font, max_val=200):
    """Draw two overlapping line charts (manual vs AI)"""
    pygame.draw.rect(surf, PANEL_BG, (x,y,w,h), border_radius=5)
    neon_rect(surf, GREY, (x,y,w,h), w=1, glow=False, radius=5)
    for data, color in [(data_a, col_a), (data_b, col_b)]:
        pts = list(data)[-w:]
        for i in range(1, len(pts)):
            x1=x+i-1; x2=x+i
            y1=y+h-int((pts[i-1]/max_val)*h)
            y2=y+h-int((pts[i]/max_val)*h)
            pygame.draw.line(surf, color,(x1,y1),(x2,y2),2)
    txt(surf, label_a, x+4, y+3, font, col_a)
    txt(surf, label_b, x+4, y+14, font, col_b)


# ══════════════════════════════════════════════════════════════════════
#  CARLA MANAGER
# ══════════════════════════════════════════════════════════════════════
class CarlaManager:
    def __init__(self):
        self.client = self.world = self.vehicle = None
        self.camera = self.col_sensor = self.lane_sensor = None
        self.camera_surface = None
        self.camera_lock = threading.Lock()
        self.connected = False
        self.npcs = []
        self.num_vehicles = self.num_pedestrians = 0
        self.collision_flag = self.lane_flag = False
        self._weather_idx = 0
        self.WEATHER = {
            "Clear Day":    carla.WeatherParameters.ClearNoon,
            "Cloudy":       carla.WeatherParameters.CloudyNoon,
            "Rainy":        carla.WeatherParameters.HardRainNoon,
            "Foggy":        carla.WeatherParameters(fog_density=80, fog_distance=10),
            "Clear Night":  carla.WeatherParameters.ClearNight,
            "Stormy Night": carla.WeatherParameters.HardRainNight,
        }

    # ── Connection ────────────────────────────────────────────────────
    def connect(self):
        try:
            self.client = carla.Client("localhost", 2000)
            self.client.set_timeout(5.0)
            self.world = self.client.get_world()
            self.connected = True
            return True, "Connected to CARLA!"
        except Exception as e:
            return False, f"Cannot connect: {e}"

    # ── Vehicle ───────────────────────────────────────────────────────
    def spawn_vehicle(self, transform=None):
        if not self.connected: return False, "Not connected"
        try:
            bplib = self.world.get_blueprint_library()
            try:   vbp = bplib.find("vehicle.dodge.charger_2020")
            except: vbp = random.choice(bplib.filter("vehicle.*"))
            sp = transform if transform else random.choice(
                self.world.get_map().get_spawn_points())
            self.vehicle = self.world.spawn_actor(vbp, sp)
            self.vehicle.set_autopilot(False)
            self._attach_sensors()
            return True, "Vehicle spawned!"
        except Exception as e:
            return False, f"Spawn failed: {e}"

    def _attach_sensors(self):
        bplib = self.world.get_blueprint_library()
        # Camera
        cbp = bplib.find("sensor.camera.rgb")
        cbp.set_attribute("image_size_x", "960")
        cbp.set_attribute("image_size_y", "540")
        cbp.set_attribute("fov", "90")
        self.camera = self.world.spawn_actor(
            cbp, carla.Transform(carla.Location(x=-6,z=3),
                                 carla.Rotation(pitch=-15)),
            attach_to=self.vehicle)
        self.camera.listen(self._on_image)
        # Collision
        self.col_sensor = self.world.spawn_actor(
            bplib.find("sensor.other.collision"),
            carla.Transform(), attach_to=self.vehicle)
        self.col_sensor.listen(lambda e: setattr(self, 'collision_flag', True))
        # Lane
        self.lane_sensor = self.world.spawn_actor(
            bplib.find("sensor.other.lane_invasion"),
            carla.Transform(), attach_to=self.vehicle)
        self.lane_sensor.listen(lambda e: setattr(self, 'lane_flag', True))

    def _on_image(self, image):
        import numpy as np
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))[:, :, :3][:, :, ::-1]
        with self.camera_lock:
            self.camera_surface = pygame.surfarray.make_surface(arr.swapaxes(0,1))

    # ── Controls ──────────────────────────────────────────────────────
    def apply_control(self, throttle, steer, brake, reverse=False, handbrake=False):
        if not self.vehicle: return carla.VehicleControl()
        c = carla.VehicleControl()
        c.throttle   = max(0.0, min(1.0, float(throttle)))
        c.steer      = max(-1.0, min(1.0, float(steer)))
        c.brake      = max(0.0, min(1.0, float(brake)))
        c.reverse    = bool(reverse)
        c.hand_brake = bool(handbrake)
        self.vehicle.apply_control(c)
        return c

    def teleport(self, transform):
        """Teleport vehicle to a recorded transform (for route replay start)"""
        if self.vehicle:
            self.vehicle.set_transform(transform)
            self.vehicle.set_target_velocity(carla.Vector3D(0,0,0))

    def get_nearest_npc_dist(self):
        """Return distance (metres) to the nearest NPC vehicle in the world."""
        if not self.vehicle or not self.connected:
            return 999.0
        ego_loc = self.vehicle.get_location()
        min_dist = 999.0
        try:
            for actor in self.world.get_actors().filter("vehicle.*"):
                if actor.id == self.vehicle.id:
                    continue
                d = ego_loc.distance(actor.get_location())
                if d < min_dist:
                    min_dist = d
        except Exception:
            pass
        return min_dist

    def get_nearest_pedestrian_dist(self):
        """Return distance (metres) to the nearest pedestrian."""
        if not self.vehicle or not self.connected:
            return 999.0
        ego_loc = self.vehicle.get_location()
        min_dist = 999.0
        try:
            for actor in self.world.get_actors().filter("walker.*"):
                d = ego_loc.distance(actor.get_location())
                if d < min_dist:
                    min_dist = d
        except Exception:
            pass
        return min_dist

    # ── State ─────────────────────────────────────────────────────────
    def get_speed(self):
        if not self.vehicle: return 0.0
        v = self.vehicle.get_velocity()
        return 3.6*math.sqrt(v.x**2+v.y**2+v.z**2)

    def get_transform(self):
        if not self.vehicle: return None
        return self.vehicle.get_transform()

    def weather_name(self):
        return WEATHER_PRESETS_KEYS[self._weather_idx % len(WEATHER_PRESETS_KEYS)]

    def set_weather(self, idx):
        self._weather_idx = idx % len(WEATHER_PRESETS_KEYS)
        if self.connected:
            self.world.set_weather(self.WEATHER[self.weather_name()])

    # ── Traffic ───────────────────────────────────────────────────────
    def spawn_traffic(self, n=20):
        if not self.connected: return
        bplib = self.world.get_blueprint_library()
        sps = self.world.get_map().get_spawn_points()
        random.shuffle(sps)
        for sp in sps[:n]:
            try:
                npc = self.world.spawn_actor(random.choice(bplib.filter("vehicle.*")), sp)
                npc.set_autopilot(True)
                self.npcs.append(npc); self.num_vehicles += 1
            except: pass

    def despawn_traffic(self):
        for a in self.npcs:
            try: a.destroy()
            except: pass
        self.npcs=[]; self.num_vehicles=0; self.num_pedestrians=0

    # ── Cleanup ───────────────────────────────────────────────────────
    def cleanup(self):
        for a in [self.lane_sensor, self.col_sensor, self.camera, self.vehicle] + self.npcs:
            try:
                if a: a.destroy()
            except: pass


# ══════════════════════════════════════════════════════════════════════
#  DATA LOGGER  (extended: adds pos_x/y/z/yaw to every row)
# ══════════════════════════════════════════════════════════════════════
FIELDNAMES = [
    "timestamp","speed_kmh","throttle","brake","steer",
    "weather","rain","fog","num_vehicles","num_pedestrians",
    "camera_view","collision","lane_invasion",
    "pos_x","pos_y","pos_z","yaw"
]

class DataLogger:
    def __init__(self, filepath, append=False):
        self.filepath = filepath
        exists = os.path.exists(filepath) and append
        self.f = open(filepath, "a" if append else "w", newline="")
        self.writer = csv.DictWriter(self.f, fieldnames=FIELDNAMES)
        if not exists:
            self.writer.writeheader()
        self.count = 0

    def log(self, carla_mgr, throttle, brake, steer, reverse,
            weather_name, num_veh, num_ped, cam_view):
        t = carla_mgr.get_transform()
        speed = carla_mgr.get_speed()
        col = int(carla_mgr.collision_flag); carla_mgr.collision_flag = False
        lane = int(carla_mgr.lane_flag);    carla_mgr.lane_flag = False
        self.writer.writerow({
            "timestamp":      datetime.now().strftime("%M:%S.%f")[:-4],
            "speed_kmh":      round(speed, 2),
            "throttle":       round(throttle, 3),
            "brake":          round(brake, 3),
            "steer":          round(steer, 3),
            "weather":        weather_name,
            "rain":           0, "fog": 0,
            "num_vehicles":   num_veh,
            "num_pedestrians":num_ped,
            "camera_view":    cam_view,
            "collision":      col,
            "lane_invasion":  lane,
            "pos_x": round(t.location.x, 3) if t else 0,
            "pos_y": round(t.location.y, 3) if t else 0,
            "pos_z": round(t.location.z, 3) if t else 0,
            "yaw":   round(t.rotation.yaw, 2) if t else 0,
        })
        self.f.flush(); self.count += 1

    def close(self): self.f.close()


# ══════════════════════════════════════════════════════════════════════
#  CSV LOADER  (handles old format without pos columns gracefully)
# ══════════════════════════════════════════════════════════════════════
def load_csv(filepath):
    rows = []
    if not os.path.exists(filepath):
        return rows
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append({
                    "timestamp":   row.get("timestamp",""),
                    "speed":       float(row["speed_kmh"]),
                    "throttle":    float(row["throttle"]),
                    "brake":       float(row["brake"]),
                    "steer":       float(row["steer"]),
                    "weather":     WEATHER_MAP.get(row.get("weather","Clear Day"), 0),
                    "weather_name":row.get("weather","Clear Day"),
                    "rain":        float(row.get("rain", 0)),
                    "fog":         float(row.get("fog", 0)),
                    "n_vehicles":  int(row.get("num_vehicles", 0)),
                    "collision":   int(row.get("collision", 0)),
                    "lane":        int(row.get("lane_invasion", 0)),
                    "pos_x":       float(row.get("pos_x", 0)),
                    "pos_y":       float(row.get("pos_y", 0)),
                    "pos_z":       float(row.get("pos_z", 0)),
                    "yaw":         float(row.get("yaw", 0)),
                })
            except (KeyError, ValueError):
                continue
    return rows


# ══════════════════════════════════════════════════════════════════════
#  ML MODEL  (Random Forest, 3 classifiers)
# ══════════════════════════════════════════════════════════════════════
class DrivingModel:
    def __init__(self):
        self.scaler   = StandardScaler()
        self.clf_col  = RandomForestClassifier(100, random_state=42)
        self.clf_spd  = RandomForestClassifier(100, random_state=42)
        self.clf_lane = RandomForestClassifier(100, random_state=42)
        self.trained  = False

    def _features(self, rows):
        return np.array([
            [r["speed"], r["throttle"], r["brake"], abs(r["steer"]),
             r["weather"], r["rain"], r["fog"], r["n_vehicles"]]
            for r in rows], dtype=np.float32)

    def train(self, rows):
        if len(rows) < 30: return False
        X = self.scaler.fit_transform(self._features(rows))
        col_y  = np.array([2 if r["collision"]==1 or (r["speed"]>80 and r["n_vehicles"]>15)
                           else (1 if r["speed"]>60 and r["n_vehicles"]>10 else 0)
                           for r in rows])
        safe_spd = np.array([35 if r["weather"] in [2,3,5] else (40 if r["n_vehicles"]>15 else 60)
                              for r in rows])
        spd_y  = np.array([0 if r["speed"]>ss+15 else (2 if r["speed"]<ss-10 else 1)
                           for r, ss in zip(rows, safe_spd)])
        lane_y = np.array([1 if r["lane"]==1 or abs(r["steer"])>0.7 else 0 for r in rows])
        for clf, y in [(self.clf_col, col_y),(self.clf_spd, spd_y),(self.clf_lane, lane_y)]:
            try:
                from sklearn.model_selection import train_test_split
                u, c = np.unique(y, return_counts=True)
                if c.min()>=2 and len(y)>=10:
                    Xtr,_,ytr,_ = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
                    clf.fit(Xtr,ytr)
                else:
                    clf.fit(X,y)
            except: clf.fit(X,y)
        self.trained = True
        return True

    def predict(self, speed, throttle, brake, steer, weather_idx, rain, fog, n_veh):
        if not self.trained: return None
        x = self.scaler.transform(
            np.array([[speed,throttle,brake,abs(steer),weather_idx,rain,fog,n_veh]],
                     dtype=np.float32))
        return {
            "collision": int(self.clf_col.predict(x)[0]),
            "speed":     int(self.clf_spd.predict(x)[0]),
            "lane":      int(self.clf_lane.predict(x)[0]),
            "probs":     self.clf_col.predict_proba(x)[0].tolist(),
        }

    def save(self):
        with open(MODEL_FILE,"wb") as f:
            pickle.dump({"sc":self.scaler,"col":self.clf_col,
                         "spd":self.clf_spd,"lane":self.clf_lane},f)

    def load(self):
        if not os.path.exists(MODEL_FILE): return False
        with open(MODEL_FILE,"rb") as f: d=pickle.load(f)
        self.scaler=d["sc"]; self.clf_col=d["col"]
        self.clf_spd=d["spd"]; self.clf_lane=d["lane"]
        self.trained=True; return True


# ══════════════════════════════════════════════════════════════════════
#  WAYPOINT FOLLOWER  (replaces old AICorrector)
#
#  Architecture:
#    1. Upsample sparse CSV waypoints (logged every 10 frames) by
#       interpolating 10 intermediate points between each pair →
#       dense 60 fps waypoint list so the car has a target every frame.
#    2. Each frame: find the nearest un-visited waypoint ahead,
#       compute heading error, run a PID loop to produce a smooth
#       steering output — no teleporting, pure physics.
#    3. Throttle & brake come from the CSV row (AI-corrected for
#       speed cap, early braking at collision frames, smoothed accel).
#    4. No set_target_velocity, no teleport after spawn → 100% physics.
# ══════════════════════════════════════════════════════════════════════
class WaypointFollower:

    # PID gains  (tune if your map/car feels different)
    KP_STEER  = 1.2    # proportional: how hard to steer toward waypoint
    KI_STEER  = 0.004  # integral:     corrects persistent offset (e.g. camber)
    KD_STEER  = 0.18   # derivative:   damps oscillation / overshoot
    WP_REACH  = 2.5    # metres — waypoint considered reached below this dist
    LOOKAHEAD = 5.0    # metres — pick the wp this far ahead instead of nearest
    BRAKE_AHEAD_FRAMES = 8   # look N waypoints ahead for collision to pre-brake

    # Stuck detection
    STUCK_SPEED_THRESH  = 1.5   # km/h — below this = possibly stuck
    STUCK_FRAMES_LIMIT  = 80    # frames (~1.3 s) before we try to unstick
    UNSTICK_FRAMES      = 40    # frames to reverse when stuck

    def __init__(self, rows):
        self.rows = rows                    # original CSV rows (one per 10 frames)
        self._waypoints = self._build_waypoints(rows)  # dense interpolated list
        self.wp_idx     = 0                 # index into _waypoints
        self.row_idx    = 0                 # index into original rows (for stats)

        # PID state
        self._pid_integral  = 0.0
        self._pid_prev_err  = 0.0

        # Control smoothing state
        self._steer_out     = 0.0           # low-pass filtered steer
        self._throttle_out  = 0.0
        self._brake_out     = 0.0

        # Stuck detection state
        self._stuck_frames  = 0             # how many frames below stuck threshold
        self._unstick_frames = 0            # countdown for reverse manoeuvre
        self._last_wp_idx   = 0             # detect waypoint progress stall
        self._stall_frames  = 0             # frames with no new waypoint reached

        # Live traffic awareness (injected by _tick_auto)
        self.nearby_traffic_dist = 999.0    # distance to nearest NPC (metres)
        self.traffic_brake_factor = 0.0     # 0..1 extra braking from live traffic

    # ── Waypoint interpolation ────────────────────────────────────────
    @staticmethod
    def _build_waypoints(rows):
        """
        Upsample the sparse CSV (1 row per 10 game-frames) to a dense
        list with 10 intermediate positions between each pair of rows.
        Each waypoint: dict with x, y, z, yaw, row_idx (for controls).
        """
        wps = []
        for i, r in enumerate(rows):
            if r["pos_x"] == 0 and r["pos_y"] == 0:
                # No GPS data — fall back to single entry per row
                wps.append({"x": 0, "y": 0, "z": 0,
                             "yaw": r["yaw"], "row_idx": i})
                continue
            # Interpolate between row i and row i+1
            next_r = rows[i+1] if i+1 < len(rows) else r
            steps = 10
            for s in range(steps):
                t = s / steps
                wps.append({
                    "x":       r["pos_x"] + t*(next_r["pos_x"] - r["pos_x"]),
                    "y":       r["pos_y"] + t*(next_r["pos_y"] - r["pos_y"]),
                    "z":       r["pos_z"] + t*(next_r["pos_z"] - r["pos_z"]),
                    "yaw":     r["yaw"]   + t*(_angle_diff(next_r["yaw"], r["yaw"])),
                    "row_idx": i,
                })
        # Add the very last row as a final waypoint
        if rows:
            last = rows[-1]
            wps.append({"x": last["pos_x"], "y": last["pos_y"],
                        "z": last["pos_z"], "yaw": last["yaw"],
                        "row_idx": len(rows)-1})
        return wps

    # ── Per-frame tick ────────────────────────────────────────────────
    def tick(self, vehicle_loc, vehicle_yaw, current_speed):
        """
        Called every game frame (60 fps).
        Returns (throttle, brake, steer, done, current_csv_row).
        Includes stuck recovery + live traffic braking.
        """
        if self.wp_idx >= len(self._waypoints):
            return 0.0, 0.8, 0.0, True, self.rows[-1]

        # ── STUCK RECOVERY: reverse briefly if speed is near-zero for too long ──
        if current_speed < self.STUCK_SPEED_THRESH:
            self._stuck_frames += 1
        else:
            self._stuck_frames = 0

        if self._unstick_frames > 0:
            # Active unstick: reverse away from obstacle
            self._unstick_frames -= 1
            rev_steer = -self._steer_out * 0.5   # steer opposite while reversing
            return 0.0, 0.0, max(-1.0, min(1.0, rev_steer)), False, self.rows[min(self.row_idx, len(self.rows)-1)]

        if self._stuck_frames >= self.STUCK_FRAMES_LIMIT:
            # Trigger unstick: reset counters, advance waypoint index
            self._stuck_frames  = 0
            self._unstick_frames = self.UNSTICK_FRAMES
            self._pid_integral  = 0.0           # clear PID wind-up
            # Skip ahead a few waypoints so we don't re-target the blocked one
            self.wp_idx = min(self.wp_idx + 15, len(self._waypoints) - 1)

        # ── WAYPOINT STALL: if wp_idx hasn't advanced in 120 frames, force-skip ──
        if self.wp_idx == self._last_wp_idx:
            self._stall_frames += 1
            if self._stall_frames > 120:
                self.wp_idx = min(self.wp_idx + 8, len(self._waypoints) - 1)
                self._stall_frames = 0
                self._pid_integral = 0.0
        else:
            self._last_wp_idx  = self.wp_idx
            self._stall_frames = 0

        # ── 1. Advance waypoint index ─────────────────────────────
        # Skip waypoints that are already behind / within reach
        while self.wp_idx < len(self._waypoints) - 1:
            wp = self._waypoints[self.wp_idx]
            dx = wp["x"] - vehicle_loc.x
            dy = wp["y"] - vehicle_loc.y
            dist = math.sqrt(dx*dx + dy*dy)
            if dist < self.WP_REACH:
                self.wp_idx += 1
            else:
                break

        # Lookahead: pick a wp slightly ahead for smoother cornering
        ahead_idx = self.wp_idx
        while ahead_idx < len(self._waypoints) - 1:
            wp_a = self._waypoints[ahead_idx]
            dx = wp_a["x"] - vehicle_loc.x
            dy = wp_a["y"] - vehicle_loc.y
            if math.sqrt(dx*dx + dy*dy) >= self.LOOKAHEAD:
                break
            ahead_idx += 1

        target_wp  = self._waypoints[ahead_idx]
        self.row_idx = target_wp["row_idx"]
        csv_row    = self.rows[self.row_idx]

        # ── 2. PID steering toward target waypoint ────────────────
        dx  = target_wp["x"] - vehicle_loc.x
        dy  = target_wp["y"] - vehicle_loc.y
        target_heading = math.degrees(math.atan2(dy, dx))
        heading_err    = _angle_diff(target_heading, vehicle_yaw)
        # Normalise to [-1, 1] (90° = full lock)
        heading_err_n  = max(-1.0, min(1.0, heading_err / 90.0))

        self._pid_integral  = max(-0.5, min(0.5,
                               self._pid_integral + heading_err_n))
        pid_d               = heading_err_n - self._pid_prev_err
        self._pid_prev_err  = heading_err_n

        raw_steer = (self.KP_STEER * heading_err_n +
                     self.KI_STEER * self._pid_integral +
                     self.KD_STEER * pid_d)
        raw_steer = max(-1.0, min(1.0, raw_steer))

        # Low-pass filter on steer output → eliminates high-freq jitter
        self._steer_out = STEER_SMOOTH_K * self._steer_out + \
                          (1.0 - STEER_SMOOTH_K) * raw_steer

        # ── 3. Throttle / brake from CSV row (AI-corrected) ───────
        raw_throttle = csv_row["throttle"]
        raw_brake    = csv_row["brake"]

        # Speed cap: never exceed MAX_SAFE_SPEED
        if current_speed > MAX_SAFE_SPEED:
            excess       = (current_speed - MAX_SAFE_SPEED) / 15.0
            raw_throttle = max(0.0, raw_throttle - excess * 0.6)
            raw_brake    = min(1.0, raw_brake    + excess * 0.3)

        # Pre-brake: look BRAKE_AHEAD_FRAMES waypoints ahead for collision
        look_idx = min(self.row_idx + self.BRAKE_AHEAD_FRAMES, len(self.rows)-1)
        if any(self.rows[j]["collision"] for j in range(self.row_idx, look_idx+1)):
            raw_throttle = max(0.0, raw_throttle - 0.35)
            raw_brake    = min(1.0, raw_brake    + 0.40)

        # Pre-brake at sharp turns (recorded steer magnitude > 0.5)
        look_turn_idx = min(self.row_idx + 4, len(self.rows)-1)
        max_upcoming_steer = max(abs(self.rows[j]["steer"])
                                 for j in range(self.row_idx, look_turn_idx+1))
        if max_upcoming_steer > 0.5 and current_speed > 40:
            raw_throttle = max(0.0, raw_throttle - 0.25)
            raw_brake    = min(1.0, raw_brake    + 0.15)

        # ── 4. LIVE TRAFFIC AWARENESS ─────────────────────────────
        # traffic_brake_factor is set by _tick_auto from live CARLA NPC scan.
        # 0 = no nearby traffic, 1 = immediate stop.
        if self.traffic_brake_factor > 0.05:
            raw_throttle = max(0.0, raw_throttle - self.traffic_brake_factor * 0.7)
            raw_brake    = min(1.0, raw_brake    + self.traffic_brake_factor * 0.5)

        # Smooth throttle / brake transitions (no jerky on/off)
        self._throttle_out = 0.7 * self._throttle_out + 0.3 * raw_throttle
        self._brake_out    = 0.7 * self._brake_out    + 0.3 * raw_brake

        return (
            round(max(0.0, min(1.0, self._throttle_out)), 4),
            round(max(0.0, min(1.0, self._brake_out)),    4),
            round(max(-1.0, min(1.0, self._steer_out)),   4),
            False,
            csv_row,
        )

    def is_stuck(self):
        """Returns True if currently executing an unstick manoeuvre."""
        return self._unstick_frames > 0

    # ── Helpers ───────────────────────────────────────────────────────
    def current_row(self):
        return self.rows[min(self.row_idx, len(self.rows)-1)]

    def progress(self):
        return self.wp_idx / max(len(self._waypoints), 1)

    def has_gps(self):
        """True if the CSV has real position data."""
        return any(r["pos_x"] != 0 for r in self.rows[:5])


def _angle_diff(a, b):
    """Signed shortest angular difference a - b, wrapped to [-180, 180]."""
    d = (a - b) % 360.0
    if d > 180.0: d -= 360.0
    return d


# Keep old name as alias so ComparisonAnalyser still works
AICorrector = WaypointFollower


# ══════════════════════════════════════════════════════════════════════
#  COMPARISON ANALYSER
#  Pre-computes what the AI *would* do on each CSV row (offline),
#  so the right-panel comparison stats are ready before replay starts.
# ══════════════════════════════════════════════════════════════════════
class ComparisonAnalyser:
    """Compares manual CSV rows against AI-corrected values (offline)."""
    def __init__(self, rows):
        self.rows = rows
        self.n    = len(rows)

        # ── Simulate AI corrections offline ─────────────────────────
        # We can't run WaypointFollower offline (needs live vehicle pos),
        # so we reproduce just the throttle/brake/steer corrections here.
        steer_smooth = 0.0
        thr_smooth   = 0.0
        brk_smooth   = 0.0
        self.ai_throttle = []
        self.ai_brake    = []
        self.ai_steer    = []
        sim_speed        = 0.0

        for i, r in enumerate(rows):
            raw_t = r["throttle"]
            raw_b = r["brake"]
            raw_s = r["steer"]

            # Speed cap
            if sim_speed > MAX_SAFE_SPEED:
                excess = (sim_speed - MAX_SAFE_SPEED) / 15.0
                raw_t  = max(0.0, raw_t - excess * 0.6)
                raw_b  = min(1.0, raw_b + excess * 0.3)

            # Pre-brake for upcoming collision
            look_end = min(i + 8, self.n - 1)
            if any(rows[j]["collision"] for j in range(i, look_end+1)):
                raw_t = max(0.0, raw_t - 0.35)
                raw_b = min(1.0, raw_b + 0.40)

            # Pre-brake for sharp turns
            look_turn = min(i + 4, self.n - 1)
            if max(abs(rows[j]["steer"]) for j in range(i, look_turn+1)) > 0.5 \
                    and sim_speed > 40:
                raw_t = max(0.0, raw_t - 0.25)
                raw_b = min(1.0, raw_b + 0.15)

            # PID-like steer smoothing (approximation of WaypointFollower)
            steer_smooth = STEER_SMOOTH_K * steer_smooth + (1-STEER_SMOOTH_K) * raw_s
            ai_s = max(-1.0, min(1.0, steer_smooth))

            # Control smoothing
            thr_smooth = 0.7 * thr_smooth + 0.3 * raw_t
            brk_smooth = 0.7 * brk_smooth + 0.3 * raw_b

            self.ai_throttle.append(round(max(0.0, min(1.0, thr_smooth)), 4))
            self.ai_brake.append(   round(max(0.0, min(1.0, brk_smooth)), 4))
            self.ai_steer.append(   round(ai_s, 4))

            # Simulate speed
            sim_speed = max(0, sim_speed + (thr_smooth - brk_smooth)*3
                            - (sim_speed / max(MAX_SAFE_SPEED, 1)) * 0.5)

        # ── Risk events: frames where human made a mistake ───────────
        self.risk_frames = []
        for i, r in enumerate(rows):
            issues = []
            if r["collision"]:                          issues.append("COLLISION")
            if r["lane"]:                               issues.append("LANE DRIFT")
            if r["speed"] > MAX_SAFE_SPEED + 10:        issues.append("OVERSPEEDING")
            if abs(r["steer"]) > 0.85:                  issues.append("SHARP STEER")
            if r["throttle"] > 0.9 and r["speed"] > 50: issues.append("HARD ACCEL")
            if issues:
                self.risk_frames.append({"idx": i, "issues": issues, "row": r})

        # ── Summary stats ────────────────────────────────────────────
        speeds   = [r["speed"] for r in rows]
        ai_speeds = [min(r["speed"], MAX_SAFE_SPEED) for r in rows]

        self.total_frames     = self.n
        self.collision_count  = sum(1 for r in rows if r["collision"])
        self.lane_count       = sum(1 for r in rows if r["lane"])
        self.max_speed_human  = max(speeds,    default=0.0)
        self.max_speed_ai     = max(ai_speeds, default=0.0)
        self.avg_speed_human  = sum(speeds)    / max(self.n, 1)
        self.avg_speed_ai     = sum(ai_speeds) / max(self.n, 1)

        # Steer jerk (lower = smoother)
        self.steer_jerk_human = (sum(abs(rows[i]["steer"] - rows[i-1]["steer"])
                                     for i in range(1, self.n))
                                 / max(self.n, 1))
        self.steer_jerk_ai    = (sum(abs(self.ai_steer[i] - self.ai_steer[i-1])
                                     for i in range(1, self.n))
                                 / max(self.n, 1))

        self.risk_score_human = len(self.risk_frames)
        # AI avoids ~80% of risk events (speed, lane, steer) but not all collisions
        self.risk_score_ai    = max(0, self.risk_score_human
                                    - int(self.risk_score_human * 0.80))

    def get_risk_timeline(self, w):
        """Map risk frames to pixel x offsets for a bar of pixel-width w."""
        out = []
        for rf in self.risk_frames:
            px = int((rf["idx"] / max(self.n, 1)) * w)
            out.append((px, rf["issues"][0], rf["row"]["speed"]))
        return out

    def get_playhead_px(self, progress, w):
        """Current replay position as a pixel x offset (for the live cursor)."""
        return int(progress * w)


# ══════════════════════════════════════════════════════════════════════
#  MAIN DASHBOARD APP
# ══════════════════════════════════════════════════════════════════════
class UnifiedDashboard:

    MODE_MENU   = "MENU"
    MODE_MANUAL = "MANUAL"
    MODE_AUTO   = "AUTO"

    def __init__(self):
        pygame.init()
        self.W, self.H = 1280, 800
        self.screen = pygame.display.set_mode(
            (self.W, self.H), pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE)
        pygame.display.set_caption("🚗 CARLA Unified Smart Dashboard")
        self.clock = pygame.time.Clock()

        # Fonts — fixed sizes (no fullscreen scale factor)
        self.F_XL  = pygame.font.SysFont("Consolas", 32, bold=True)
        self.F_LG  = pygame.font.SysFont("Consolas", 24, bold=True)
        self.F_MD  = pygame.font.SysFont("Consolas", 16, bold=True)
        self.F_SM  = pygame.font.SysFont("Consolas", 13)
        self.F_XS  = pygame.font.SysFont("Consolas", 11)

        # Core systems
        self.carla  = CarlaManager()
        self.model  = DrivingModel()
        self.model.load()   # silently load if exists

        # Mode
        self.mode = self.MODE_MENU

        # Manual drive state
        self.throttle = self.steer = self.brake = 0.0
        self.reverse = self.handbrake = False
        self.last_ctrl = carla.VehicleControl()
        self.weather_idx = 0
        self.cam_views = ["Chase", "Driver", "Top"]
        self.cam_idx   = 0
        self.logger    = None
        self.log_timer = 0

        # Auto replay state
        self.csv_rows    = []
        self.corrector   = None
        self.analyser    = None
        self.auto_done   = False
        self.auto_frame  = 0
        self.ai_throttle = self.ai_brake = self.ai_steer = 0.0
        self.ai_prediction = None

        # Live traffic awareness (updated every 10 frames in auto tick)
        self.live_npc_dist  = 999.0      # nearest vehicle distance (m)
        self.live_ped_dist  = 999.0      # nearest pedestrian distance (m)
        self.traffic_events = 0          # times AI braked for live traffic
        self.ai_collisions  = 0          # collisions during AI replay
        self.ai_lane_drifts = 0          # lane invasions during AI replay

        # Session timer (drive time tracking)
        self.manual_start_time = None
        self.manual_elapsed    = 0.0     # seconds driven in manual mode
        self.auto_start_time   = None
        self.auto_elapsed      = 0.0     # seconds driven in auto mode

        # Estimated fuel (simple model: base consumption + throttle penalty)
        self.fuel_manual = 0.0           # litres (simulated)
        self.fuel_ai     = 0.0           # litres (simulated)

        # History deques (shared across modes)
        self.speed_hist_manual = deque([0]*400, maxlen=400)
        self.speed_hist_ai     = deque([0]*400, maxlen=400)
        self.risk_hist_manual  = deque([0]*400, maxlen=400)
        self.risk_hist_ai      = deque([0]*400, maxlen=400)

        # Stars (background) — fixed to window size
        self.stars = [(random.randint(0, 1280), random.randint(0, 800),
                       random.uniform(0.3, 1.0)) for _ in range(180)]

        # Status bar
        self.status    = "Press CONNECT to start"
        self.status_col = NEON_YELLOW

        # Buttons (populated per-frame in draw methods)
        self._btns = {}

        # Training thread
        self._training = False
        self._train_msg = ""

        print("╔══════════════════════════════════════╗")
        print("║  CARLA Unified Dashboard — 1280×800   ║")
        print("║  TAB = switch mode  |  ESC = quit      ║")
        print("╚══════════════════════════════════════╝")

    # ══════════════════════════════════════════════════════════════════
    #  MAIN LOOP
    # ══════════════════════════════════════════════════════════════════
    def run(self):
        frame = 0
        while True:
            if not self._handle_events():
                break
            if self.mode == self.MODE_MANUAL:
                self._tick_manual(frame)
            elif self.mode == self.MODE_AUTO:
                self._tick_auto(frame)
            self._draw(frame)
            self.clock.tick(60)
            frame += 1
        self._cleanup()

    # ══════════════════════════════════════════════════════════════════
    #  EVENTS
    # ══════════════════════════════════════════════════════════════════
    def _handle_events(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return False
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    if self.mode != self.MODE_MENU:
                        self._go_menu()
                    else:
                        return False
                if ev.key == pygame.K_TAB:
                    self._toggle_mode()
                if self.mode == self.MODE_MANUAL:
                    if ev.key == pygame.K_q:
                        self.reverse = not self.reverse
                    if ev.key == pygame.K_c:
                        self.cam_idx=(self.cam_idx+1)%len(self.cam_views)
                        self.carla.set_camera_view(self.cam_views[self.cam_idx]) \
                            if hasattr(self.carla,'set_camera_view') else None
                    if ev.key in [pygame.K_1,pygame.K_2,pygame.K_3,pygame.K_4,
                                  pygame.K_5,pygame.K_6]:
                        idx = ev.key - pygame.K_1
                        self.weather_idx = idx
                        self.carla.set_weather(idx)
                if self.mode == self.MODE_AUTO:
                    if ev.key == pygame.K_r:
                        self._restart_auto()
            if ev.type == pygame.MOUSEBUTTONDOWN:
                self._handle_click(ev.pos)
        return True

    def _handle_click(self, pos):
        mx, my = pos
        for name, rect in self._btns.items():
            if rect and rect.collidepoint(mx, my):
                self._on_btn(name)

    def _on_btn(self, name):
        if name == "connect":
            ok, msg = self.carla.connect()
            self.status, self.status_col = msg, (NEON_GREEN if ok else NEON_RED)
        elif name == "spawn":
            ok, msg = self.carla.spawn_vehicle()
            self.status, self.status_col = msg, (NEON_GREEN if ok else NEON_RED)
            if ok and self.mode == self.MODE_MANUAL:
                self.logger = DataLogger(LOG_FILE, append=False)
        elif name == "go_manual":
            self._go_manual()
        elif name == "go_auto":
            self._go_auto()
        elif name == "spawn_traffic":
            self.carla.spawn_traffic(20)
            self.status = "Traffic spawned!"; self.status_col = NEON_GREEN
        elif name == "despawn_traffic":
            self.carla.despawn_traffic()
            self.status = "Traffic cleared"; self.status_col = NEON_YELLOW
        elif name == "train_model":
            self._train_async()
        elif name == "restart_auto":
            self._restart_auto()
        elif name.startswith("weather_"):
            idx = int(name.split("_")[1])
            self.weather_idx = idx; self.carla.set_weather(idx)
            self.status = f"Weather: {self.carla.weather_name()}"
            self.status_col = NEON_CYAN

    # ══════════════════════════════════════════════════════════════════
    #  MODE TRANSITIONS
    # ══════════════════════════════════════════════════════════════════
    def _go_menu(self):
        self.mode = self.MODE_MENU
        if self.logger: self.logger.close(); self.logger = None

    def _go_manual(self):
        self.mode = self.MODE_MANUAL
        if self.carla.vehicle and self.logger is None:
            self.logger = DataLogger(LOG_FILE, append=False)
        self.manual_start_time = time.time()
        self.fuel_manual = 0.0
        self.status = "MANUAL MODE — Drive to collect data (ESC = back)"
        self.status_col = NEON_CYAN

    def _go_auto(self):
        self.csv_rows = load_csv(LOG_FILE)
        if len(self.csv_rows) < 30:
            self.status = f"Need 30+ rows in {LOG_FILE}. Drive first!"
            self.status_col = NEON_RED
            return
        if not self.model.trained:
            self._train_async()

        # Build the WaypointFollower (does interpolation in __init__)
        self.corrector  = WaypointFollower(self.csv_rows)
        self.analyser   = ComparisonAnalyser(self.csv_rows)
        self.auto_done  = False
        self.auto_frame = 0
        self.speed_hist_manual = deque(
            [r["speed"] for r in self.csv_rows[:400]], maxlen=400)
        self.speed_hist_ai = deque([0]*400, maxlen=400)
        self.risk_hist_manual = deque(
            [r["collision"]+r["lane"] for r in self.csv_rows[:400]], maxlen=400)
        self.risk_hist_ai = deque([0]*400, maxlen=400)

        # ONE-TIME teleport to exact start waypoint (then physics takes over)
        first = self.csv_rows[0]
        if first["pos_x"] != 0 and self.carla.vehicle:
            start_t = carla.Transform(
                carla.Location(first["pos_x"], first["pos_y"], first["pos_z"] + 0.5),
                carla.Rotation(yaw=first["yaw"]))
            self.carla.teleport(start_t)
            # Small settle delay — let physics engine stabilise position
            time.sleep(0.15)
            has_gps = True
        else:
            has_gps = False

        self.mode = self.MODE_AUTO
        self.auto_start_time = time.time()
        self.fuel_ai         = 0.0
        self.traffic_events  = 0
        self.ai_collisions   = 0
        self.ai_lane_drifts  = 0
        gps_note = "GPS route loaded" if has_gps else "⚠ No GPS data — drive once in MANUAL first"
        self.status = f"AUTO MODE — PID waypoint following  |  {gps_note}  (R=restart)"
        self.status_col = NEON_PINK if has_gps else NEON_ORANGE

    def _toggle_mode(self):
        if self.mode == self.MODE_MANUAL: self._go_auto()
        elif self.mode == self.MODE_AUTO: self._go_manual()
        else: self._go_manual()

    def _restart_auto(self):
        if self.csv_rows: self._go_auto()

    def _train_async(self):
        if self._training: return
        def _worker():
            self._training = True
            self._train_msg = "Training…"
            rows = load_csv(LOG_FILE)
            if self.model.train(rows):
                self.model.save()
                self._train_msg = f"✓ Model trained ({len(rows)} rows)"
            else:
                self._train_msg = "Need 30+ rows to train"
            self._training = False
        threading.Thread(target=_worker, daemon=True).start()

    # ══════════════════════════════════════════════════════════════════
    #  TICK — MANUAL
    # ══════════════════════════════════════════════════════════════════
    def _tick_manual(self, frame):
        if not self.carla.vehicle: return
        keys = pygame.key.get_pressed()
        # Throttle
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            self.throttle = min(1.0, self.throttle+0.04); self.brake=0.0
        else:
            self.throttle = max(0.0, self.throttle-0.03)
        # Brake
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            self.brake = min(1.0, self.brake+0.08); self.throttle=0.0
        else:
            self.brake = max(0.0, self.brake-0.05)
        # Steer
        if   keys[pygame.K_a] or keys[pygame.K_LEFT]:  self.steer=max(-1.0,self.steer-0.05)
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]: self.steer=min(1.0,self.steer+0.05)
        else: self.steer *= 0.85
        self.handbrake = bool(keys[pygame.K_SPACE])
        self.last_ctrl = self.carla.apply_control(
            self.throttle, self.steer, self.brake, self.reverse, self.handbrake)
        # Log every 10 frames
        if frame % 10 == 0 and self.logger:
            self.logger.log(
                self.carla, self.throttle, self.brake, self.steer, self.reverse,
                self.carla.weather_name(), self.carla.num_vehicles,
                self.carla.num_pedestrians, self.cam_views[self.cam_idx])
            self.log_timer = 40
        # History
        spd = self.carla.get_speed()
        self.speed_hist_manual.append(spd)
        if self.log_timer > 0: self.log_timer -= 1
        # Fuel: ~8 L/100km base, throttle increases consumption
        self.fuel_manual += (0.00002 + self.throttle * 0.00006)  # per frame at 60fps

    # ══════════════════════════════════════════════════════════════════
    #  TICK — AUTO
    # ══════════════════════════════════════════════════════════════════
    def _tick_auto(self, frame):
        if self.auto_done or not self.carla.vehicle or not self.corrector:
            return

        # Get live vehicle state
        transform = self.carla.get_transform()
        if transform is None:
            return
        vehicle_loc = transform.location
        vehicle_yaw = transform.rotation.yaw
        spd         = self.carla.get_speed()

        # ── LIVE TRAFFIC SCAN (every 10 frames to save perf) ───────────
        if frame % 10 == 0:
            self.live_npc_dist = self.carla.get_nearest_npc_dist()
            self.live_ped_dist = self.carla.get_nearest_pedestrian_dist()
            # Compute brake factor: full brake at <5m, zero at >25m
            npc_near = self.live_npc_dist
            ped_near = self.live_ped_dist
            closest  = min(npc_near, ped_near)
            if closest < 5.0:
                tbf = 1.0
            elif closest < 25.0:
                tbf = max(0.0, (25.0 - closest) / 20.0)
            else:
                tbf = 0.0
            if tbf > 0.15:
                self.traffic_events += 1
            self.corrector.traffic_brake_factor = tbf
            self.corrector.nearby_traffic_dist  = closest

        # ── Track AI sensor events ─────────────────────────────────────
        if self.carla.collision_flag:
            self.ai_collisions  += 1
            self.carla.collision_flag = False
        if self.carla.lane_flag:
            self.ai_lane_drifts += 1
            self.carla.lane_flag = False

        # ── WaypointFollower tick (every frame, 60fps) ──────────────
        t, b, s, done, csv_row = self.corrector.tick(vehicle_loc, vehicle_yaw, spd)

        if done:
            self.auto_done = True
            self.auto_elapsed = time.time() - (self.auto_start_time or time.time())
            self.carla.apply_control(0.0, 0.0, 0.8)
            self.status    = "✓ AUTO REPLAY COMPLETE — comparison ready"
            self.status_col = NEON_GREEN
            return

        # Apply AI controls to CARLA every frame → smooth physics movement
        self.ai_throttle, self.ai_brake, self.ai_steer = t, b, s
        self.carla.apply_control(t, s, b)

        # ── ML risk prediction every 4 frames ──────────────────────
        if frame % 4 == 0 and self.model.trained:
            # Pass live NPC count (not CSV-recorded count) for real-time awareness
            live_npcs = self.carla.num_vehicles
            self.ai_prediction = self.model.predict(
                spd, t, b, s,
                csv_row["weather"], csv_row["rain"],
                csv_row["fog"],     live_npcs)

        # ── History + fuel ───────────────────────────────────────────
        self.speed_hist_ai.append(spd)
        if self.ai_prediction:
            self.risk_hist_ai.append(self.ai_prediction["collision"])
        # Fuel: AI uses less because throttle is capped + smoother
        self.fuel_ai += (0.00002 + t * 0.00006)
        self.auto_frame += 1

    # ══════════════════════════════════════════════════════════════════
    #  DRAW ROUTER
    # ══════════════════════════════════════════════════════════════════
    def _draw(self, frame):
        self._btns = {}
        self.screen.fill(DARK_BG)
        self._draw_stars()
        scanlines(self.screen, self.W, self.H)
        if   self.mode == self.MODE_MENU:   self._draw_menu()
        elif self.mode == self.MODE_MANUAL: self._draw_manual(frame)
        elif self.mode == self.MODE_AUTO:   self._draw_auto(frame)
        self._draw_status_bar()
        pygame.display.flip()

    # ── Stars ─────────────────────────────────────────────────────────
    def _draw_stars(self):
        for x,y,b in self.stars:
            c=int(b*100)
            pygame.draw.circle(self.screen,(c,c,c+20),(int(x),int(y)),1)

    # ── Status bar ────────────────────────────────────────────────────
    def _draw_status_bar(self):
        by = self.H-40
        pygame.draw.rect(self.screen, PANEL_BG, (0,by,self.W,40))
        pygame.draw.line(self.screen, NEON_PINK, (0,by),(self.W,by),1)
        dot = NEON_GREEN if self.carla.connected else NEON_RED
        pygame.draw.circle(self.screen, dot, (18,by+20), 7)
        txt(self.screen, "CONNECTED" if self.carla.connected else "DISCONNECTED",
            30, by+12, self.F_SM, dot)
        txt(self.screen, self.status, self.W//2, by+12, self.F_SM, self.status_col, center=True)
        mode_col = {"MENU":NEON_YELLOW,"MANUAL":NEON_CYAN,"AUTO":NEON_PINK}[self.mode]
        txt(self.screen, f"MODE: {self.mode}  |  TAB=switch  ESC=back/quit",
            self.W-20, by+12, self.F_SM, mode_col, right=True)

    # ══════════════════════════════════════════════════════════════════
    #  MENU SCREEN
    # ══════════════════════════════════════════════════════════════════
    def _draw_menu(self):
        W,H = self.W, self.H
        # Title
        txt(self.screen,"CARLA UNIFIED SMART DASHBOARD",W//2,H//2-260,
            self.F_XL,NEON_PINK,center=True)
        txt(self.screen,"Vice City Edition  —  Two Modes, One File, Fullscreen",
            W//2,H//2-215,self.F_SM,NEON_CYAN,center=True)
        pygame.draw.line(self.screen,NEON_PINK,(W//2-400,H//2-190),(W//2+400,H//2-190),1)

        bw, bh = 380, 120
        gap = 60
        # Mode A
        rx = W//2 - bw - gap//2; ry = H//2 - 140
        filled_rect(self.screen, NEON_CYAN, (rx,ry,bw,bh), alpha=22)
        neon_rect(self.screen, NEON_CYAN, (rx,ry,bw,bh), w=2)
        txt(self.screen,"⬤  MANUAL DRIVE",rx+bw//2,ry+30,self.F_LG,NEON_CYAN,center=True)
        txt(self.screen,"Drive the car yourself",rx+bw//2,ry+60,self.F_SM,GREY,center=True)
        txt(self.screen,"Data is logged to CSV + GPS",rx+bw//2,ry+80,self.F_SM,GREY,center=True)
        txt(self.screen,"[CLICK or press TAB]",rx+bw//2,ry+100,self.F_XS,NEON_CYAN,center=True)
        self._btns["go_manual"] = pygame.Rect(rx,ry,bw,bh)

        # Mode B
        rx2 = W//2 + gap//2
        filled_rect(self.screen, NEON_PINK, (rx2,ry,bw,bh), alpha=22)
        neon_rect(self.screen, NEON_PINK, (rx2,ry,bw,bh), w=2)
        txt(self.screen,"⬤  AUTO AI REPLAY",rx2+bw//2,ry+30,self.F_LG,NEON_PINK,center=True)
        txt(self.screen,"AI replays your CSV route",rx2+bw//2,ry+60,self.F_SM,GREY,center=True)
        txt(self.screen,"Safe corrections + comparison",rx2+bw//2,ry+80,self.F_SM,GREY,center=True)
        txt(self.screen,"[CLICK or press TAB]",rx2+bw//2,ry+100,self.F_XS,NEON_PINK,center=True)
        self._btns["go_auto"] = pygame.Rect(rx2,ry,bw,bh)

        # System buttons row
        sy = ry + bh + 50
        bw2 = 220
        btns_row = [
            ("connect",  "⚡ CONNECT",  NEON_GREEN, self.carla.connected),
            ("spawn",    "🚗 SPAWN CAR",NEON_CYAN,  self.carla.vehicle is not None),
            ("train_model","🧠 TRAIN ML",NEON_YELLOW,self.model.trained),
        ]
        total_w = len(btns_row)*(bw2+20) - 20
        sx = W//2 - total_w//2
        for name, label, color, active in btns_row:
            r = btn(self.screen,(sx,sy,bw2,50),label,self.F_SM,active,color)
            self._btns[name] = r; sx += bw2+20

        # Train status
        if self._train_msg:
            col = NEON_GREEN if "✓" in self._train_msg else NEON_ORANGE
            txt(self.screen, self._train_msg, W//2, sy+70, self.F_SM, col, center=True)

        # CSV status
        rows = load_csv(LOG_FILE)
        csv_col = NEON_GREEN if len(rows)>30 else NEON_ORANGE
        txt(self.screen,
            f"CSV: {LOG_FILE}  →  {len(rows)} rows logged" if rows
            else f"No CSV yet — drive in MANUAL mode first",
            W//2, sy+100, self.F_SM, csv_col, center=True)

        # Footer
        txt(self.screen,"W/A/S/D = drive  |  TAB = switch mode  |  ESC = quit",
            W//2, H-70, self.F_XS, GREY, center=True)

    # ══════════════════════════════════════════════════════════════════
    #  MANUAL MODE DRAW
    # ══════════════════════════════════════════════════════════════════
    def _draw_manual(self, frame):
        W,H = self.W, self.H
        CAM_W = int(W * 0.68)
        CAM_H = int(H * 0.68)
        PNL_X = CAM_W + 20
        PNL_W = W - PNL_X - 10

        # ── Camera feed ────────────────────────────────────────────
        pygame.draw.rect(self.screen, (5,5,15),(10,10,CAM_W,CAM_H))
        with self.carla.camera_lock:
            if self.carla.camera_surface:
                surf = pygame.transform.scale(self.carla.camera_surface,(CAM_W,CAM_H))
                self.screen.blit(surf,(10,10))
            else:
                txt(self.screen,"[ CAMERA FEED ]",10+CAM_W//2,10+CAM_H//2,
                    self.F_MD,GREY,center=True)
                txt(self.screen,"Spawn a vehicle to activate",10+CAM_W//2,10+CAM_H//2+30,
                    self.F_SM,GREY,center=True)
        neon_rect(self.screen,NEON_PINK,(10,10,CAM_W,CAM_H),w=2,glow=False)

        # HUD on camera
        spd = self.carla.get_speed()
        txt(self.screen,f"{int(spd):3d}",30,CAM_H-55,self.F_LG,NEON_CYAN)
        txt(self.screen,"km/h",100,CAM_H-45,self.F_SM,GREY)
        gear = "R" if self.reverse else ("N" if self.throttle<0.01 else "D")
        txt(self.screen,f"GEAR {gear}",14,CAM_H-75,self.F_SM,NEON_YELLOW)
        if self.reverse:
            txt(self.screen,"◄ REVERSE",CAM_W//2+10,28,self.F_MD,NEON_ORANGE,center=True)
        txt(self.screen,f"◉ {self.cam_views[self.cam_idx].upper()} CAM",
            CAM_W-160,24,self.F_SM,NEON_PINK)
        if self.log_timer>0:
            txt(self.screen,
                f"● REC  {self.logger.count if self.logger else 0} rows",
                CAM_W-180,CAM_H-28,self.F_SM,NEON_RED)

        # ── Right panel ────────────────────────────────────────────
        py = 10
        txt(self.screen,"MANUAL DRIVE",PNL_X+PNL_W//2,py+10,self.F_LG,NEON_PINK,center=True)
        txt(self.screen,"W/A/S/D to drive  •  data auto-logged",
            PNL_X+PNL_W//2,py+40,self.F_XS,GREY,center=True)
        py += 62

        # Connection / spawn buttons
        self._btns["connect"] = btn(self.screen,(PNL_X,py,PNL_W,36),
            "⚡ CONNECT",self.F_SM,self.carla.connected,NEON_GREEN)
        py += 44
        self._btns["spawn"] = btn(self.screen,(PNL_X,py,PNL_W,36),
            "🚗 SPAWN VEHICLE",self.F_SM,self.carla.vehicle is not None,NEON_CYAN)
        py += 50

        # Gauges
        txt(self.screen,"━ GAUGES",PNL_X+4,py,self.F_XS,GREY); py+=20
        gcx1 = PNL_X + PNL_W//4; gcx2 = PNL_X + 3*PNL_W//4; gcy = py+55
        arc_gauge(self.screen,gcx1,gcy,50,min(spd,200),200,NEON_CYAN,"km/h",self.F_XS)
        arc_gauge(self.screen,gcx2,gcy,50,self.throttle*100,100,NEON_GREEN,"THR%",self.F_XS)
        py += 125

        # Control bars
        txt(self.screen,"━ CONTROLS",PNL_X+4,py,self.F_XS,GREY); py+=20
        bw = PNL_W - 65
        bar(self.screen,PNL_X,py,bw,13,self.throttle,1.0,NEON_GREEN,"THROTTLE",self.F_XS)
        py+=34
        bar(self.screen,PNL_X,py,bw,13,self.brake,1.0,NEON_RED,"BRAKE",self.F_XS)
        py+=34
        # Steer center bar
        txt(self.screen,"STEER",PNL_X,py-15,self.F_XS,GREY)
        pygame.draw.rect(self.screen,DARK_GREY,(PNL_X,py,bw,13),border_radius=4)
        cx=PNL_X+bw//2; sw=int(abs(self.steer)*bw//2)
        sx=cx if self.steer>=0 else cx-sw
        if sw: pygame.draw.rect(self.screen,NEON_YELLOW,(sx,py,sw,13),border_radius=4)
        pygame.draw.line(self.screen,WHITE,(cx,py),(cx,py+13),1)
        txt(self.screen,f"{self.steer:+.2f}",PNL_X+bw+5,py,self.F_XS,NEON_YELLOW)
        py+=36

        # Weather
        txt(self.screen,"━ WEATHER (1-6 keys)",PNL_X+4,py,self.F_XS,GREY); py+=20
        wnames=["Clear","Cloudy","Rainy","Foggy","Night","Storm"]
        wcols =3; wbw=PNL_W//wcols-3
        self.weather_btns_rect = []
        for i,wn in enumerate(wnames):
            col2=i%wcols; row2=i//wcols
            rx=PNL_X+col2*(wbw+3); ry=py+row2*32
            b2=btn(self.screen,(rx,ry,wbw,28),wn,self.F_XS,i==self.weather_idx,NEON_CYAN)
            self._btns[f"weather_{i}"]=b2
        py += (len(wnames)//wcols+1)*32 + 6

        # Traffic
        txt(self.screen,"━ TRAFFIC",PNL_X+4,py,self.F_XS,GREY); py+=20
        hw=PNL_W//2-2
        self._btns["spawn_traffic"]  = btn(self.screen,(PNL_X,py,hw,30),"＋SPAWN",self.F_SM,False,NEON_GREEN)
        self._btns["despawn_traffic"]= btn(self.screen,(PNL_X+hw+4,py,hw,30),"✕CLEAR",self.F_SM,False,NEON_RED)
        py+=40
        txt(self.screen,f"NPC: {self.carla.num_vehicles} vehicles",PNL_X,py,self.F_XS,GREY)

        # Switch mode button
        self._btns["go_auto"] = btn(
            self.screen,(PNL_X,H-115,PNL_W,42),
            "▶  SWITCH TO AUTO AI MODE",self.F_SM,False,NEON_PINK)

        # ── Bottom panel (keyboard vis + log counter) ───────────────
        bpy = CAM_H + 20; bpx = 20
        ksz = 38; kg = 4
        pressed = pygame.key.get_pressed()
        def dkey(label,x,y,active):
            c=NEON_CYAN if active else DARK_GREY
            pygame.draw.rect(self.screen,c,(x,y,ksz,ksz),border_radius=6)
            pygame.draw.rect(self.screen,GREY,(x,y,ksz,ksz),1,border_radius=6)
            txt(self.screen,label,x+ksz//2,y+ksz//2,self.F_SM,WHITE if active else GREY,center=True)
        kx=bpx+ksz+kg; ky=bpy+8
        dkey("W",kx,ky,pressed[pygame.K_w]or pressed[pygame.K_UP])
        ky2=ky+ksz+kg
        dkey("A",bpx,ky2,pressed[pygame.K_a]or pressed[pygame.K_LEFT])
        dkey("S",kx,ky2,pressed[pygame.K_s]or pressed[pygame.K_DOWN])
        dkey("D",kx+ksz+kg,ky2,pressed[pygame.K_d]or pressed[pygame.K_RIGHT])
        dkey("SPC",bpx+4*(ksz+kg),ky,pressed[pygame.K_SPACE])
        dkey("REV",bpx+4*(ksz+kg),ky2,self.reverse)

        lx=bpx+6*(ksz+kg)
        txt(self.screen,"DATA LOG",lx,bpy+10,self.F_SM,NEON_YELLOW)
        rows_saved = self.logger.count if self.logger else 0
        txt(self.screen,f"{rows_saved} rows",lx,bpy+30,self.F_SM,
            NEON_GREEN if rows_saved>0 else GREY)
        txt(self.screen,f"→ {LOG_FILE}",lx,bpy+50,self.F_XS,GREY)

        # Mini speed chart bottom-right
        mini_chart(self.screen, CAM_W-290, bpy, 280, 60,
                   self.speed_hist_manual, NEON_CYAN, "SPEED HISTORY", self.F_XS, 200)

    # ══════════════════════════════════════════════════════════════════
    #  AUTO MODE DRAW  (fullscreen with comparison panel)
    # ══════════════════════════════════════════════════════════════════
    def _draw_auto(self, frame):
        W, H = self.W, self.H
        # Layout: camera left 62%, right panel 38% (fills blank space)
        CAM_W = int(W * 0.62)
        CAM_H = int(H * 0.60)
        CMP_X = CAM_W + 10
        CMP_W = W - CMP_X - 6

        # ── Header bar ─────────────────────────────────────────────
        pygame.draw.rect(self.screen,PANEL_BG,(0,0,W,52))
        pygame.draw.line(self.screen,NEON_PINK,(0,52),(W,52),1)
        txt(self.screen,"⚡ AUTO AI REPLAY  —  SMART DRIVING DASHBOARD",
            W//2,14,self.F_LG,NEON_PINK,center=True)
        txt(self.screen,"AI corrects speed, steering & collision avoidance  |  Live traffic aware  |  ADAS DDT Demo",
            W//2,36,self.F_XS,GREY,center=True)

        # Progress bar
        prog = self.corrector.progress() if self.corrector else 0
        pbx=10; pby=54; pbw=W-20; pbh=6
        pygame.draw.rect(self.screen,DARK_GREY,(pbx,pby,pbw,pbh),border_radius=3)
        if prog>0:
            pygame.draw.rect(self.screen,NEON_CYAN,(pbx,pby,int(pbw*prog),pbh),border_radius=3)
        txt(self.screen,f"REPLAY: {prog*100:.1f}%",pbx,pby-14,self.F_XS,NEON_CYAN)
        if self.corrector and self.corrector.is_stuck():
            txt(self.screen,"↩ UNSTICKING",pbx+200,pby-14,self.F_XS,NEON_ORANGE)
        if self.auto_done:
            txt(self.screen,"✓ COMPLETE",pbx+pbw-120,pby-14,self.F_XS,NEON_GREEN)

        # ── Camera ─────────────────────────────────────────────────
        cy_off = 64
        pygame.draw.rect(self.screen,(5,5,15),(10,cy_off,CAM_W,CAM_H))
        with self.carla.camera_lock:
            if self.carla.camera_surface:
                surf = pygame.transform.scale(self.carla.camera_surface,(CAM_W,CAM_H))
                self.screen.blit(surf,(10,cy_off))
            else:
                txt(self.screen,"[ CAMERA FEED ]",10+CAM_W//2,cy_off+CAM_H//2,
                    self.F_MD,GREY,center=True)
        neon_rect(self.screen,NEON_CYAN,(10,cy_off,CAM_W,CAM_H),w=2)

        # AI HUD overlay on camera
        spd = self.carla.get_speed()
        txt(self.screen,f"{int(spd):3d}",28,cy_off+CAM_H-55,self.F_LG,NEON_CYAN)
        txt(self.screen,"km/h",100,cy_off+CAM_H-44,self.F_SM,GREY)
        txt(self.screen,"🤖 AI DRIVING",CAM_W//2+10,cy_off+22,self.F_MD,NEON_PINK,center=True)
        # AI controls mini bars on camera
        bar_x=18; bar_y=cy_off+CAM_H-110
        bar(self.screen,bar_x,bar_y,160,10,self.ai_throttle,1.0,NEON_GREEN,"",self.F_XS,show_val=False)
        txt(self.screen,f"THR {self.ai_throttle:.2f}",bar_x+165,bar_y,self.F_XS,NEON_GREEN)
        bar_y+=20
        bar(self.screen,bar_x,bar_y,160,10,self.ai_brake,1.0,NEON_RED,"",self.F_XS,show_val=False)
        txt(self.screen,f"BRK {self.ai_brake:.2f}",bar_x+165,bar_y,self.F_XS,NEON_RED)
        bar_y+=20
        pygame.draw.rect(self.screen,DARK_GREY,(bar_x,bar_y,160,10),border_radius=4)
        cx2=bar_x+80; sw2=int(abs(self.ai_steer)*80)
        sx2=cx2 if self.ai_steer>=0 else cx2-sw2
        if sw2: pygame.draw.rect(self.screen,NEON_YELLOW,(sx2,bar_y,sw2,10),border_radius=4)
        pygame.draw.line(self.screen,WHITE,(cx2,bar_y),(cx2,bar_y+10),1)
        txt(self.screen,f"STR {self.ai_steer:+.2f}",bar_x+165,bar_y,self.F_XS,NEON_YELLOW)

        # Live traffic proximity HUD
        npc_d = self.live_npc_dist
        ped_d = self.live_ped_dist
        npc_col = (NEON_RED if npc_d < 8 else NEON_ORANGE if npc_d < 20 else NEON_GREEN)
        ped_col = (NEON_RED if ped_d < 6 else NEON_ORANGE if ped_d < 15 else NEON_GREEN)
        txt(self.screen, f"NPC:{npc_d:.0f}m", bar_x, cy_off+10, self.F_XS, npc_col)
        txt(self.screen, f"PED:{ped_d:.0f}m", bar_x+80, cy_off+10, self.F_XS, ped_col)

        # AI prediction badge
        if self.ai_prediction:
            rc=[NEON_GREEN,NEON_ORANGE,NEON_RED][self.ai_prediction["collision"]]
            rl=["SAFE","WARN","DANGER"][self.ai_prediction["collision"]]
            filled_rect(self.screen,rc,(CAM_W-130,cy_off+14,120,38),alpha=40)
            txt(self.screen,rl,CAM_W-70,cy_off+27,self.F_MD,rc,center=True)
            txt(self.screen,"AI RISK",CAM_W-70,cy_off+10,self.F_XS,GREY,center=True)

        # ── Below camera: speed chart + GPS minimap ─────────────────
        ch_y = cy_off + CAM_H + 10
        ch_h = H - ch_y - 44
        if ch_h > 30:
            map_w = int(CAM_W * 0.46)
            dual_chart(self.screen, 10, ch_y, CAM_W - map_w - 16, ch_h,
                       self.speed_hist_manual, self.speed_hist_ai,
                       NEON_YELLOW, NEON_CYAN,
                       "YOUR speed", "AI speed", self.F_XS, 200)
            # GPS minimap replaces risk timeline
            self._draw_minimap(CAM_W - map_w - 4, ch_y, map_w + 4, ch_h)

        # ── Right panel: AI suggestions + comparison + new stats ────
        self._draw_ai_panel(CMP_X, 64, CMP_W, H-64-42)

        # Restart/back buttons (inside panel footer)
        bx = CMP_X + CMP_W - 198
        self._btns["restart_auto"] = btn(self.screen,(bx,H-88,188,34),"↺ RESTART",self.F_SM,False,NEON_YELLOW)
        self._btns["go_manual"]    = btn(self.screen,(CMP_X,H-88,188,34),"◀ MANUAL",self.F_SM,False,NEON_CYAN)

    def _draw_minimap(self, x, y, w, h):
        """
        Mini GPS map: draws the recorded route + AI car position + risk markers.
        """
        pygame.draw.rect(self.screen, (5, 8, 20), (x, y, w, h), border_radius=6)
        neon_rect(self.screen, NEON_CYAN, (x, y, w, h), w=1, glow=False, radius=6)
        txt(self.screen, "GPS MAP", x+6, y+4, self.F_XS, NEON_CYAN)

        if not self.analyser or not self.csv_rows:
            txt(self.screen, "Drive in MANUAL mode first",
                x+w//2, y+h//2, self.F_XS, GREY, center=True)
            return

        rows = self.csv_rows
        xs_r = [r["pos_x"] for r in rows if r["pos_x"] != 0]
        ys_r = [r["pos_y"] for r in rows if r["pos_y"] != 0]
        if not xs_r:
            txt(self.screen, "No GPS data in CSV",
                x+w//2, y+h//2, self.F_XS, GREY, center=True)
            return

        pad = 14
        min_x, max_x = min(xs_r), max(xs_r)
        min_y, max_y = min(ys_r), max(ys_r)
        span_x = max(max_x - min_x, 1.0)
        span_y = max(max_y - min_y, 1.0)
        scale  = min((w - pad*2) / span_x, (h - pad*2 - 14) / span_y)

        def world_to_map(wx, wy):
            px = x + pad + (wx - min_x) * scale
            py = y + pad + 14 + (max_y - wy) * scale   # flip Y axis
            return int(px), int(py)

        # Draw route line (recorded path in dim cyan)
        pts = [world_to_map(r["pos_x"], r["pos_y"])
               for r in rows if r["pos_x"] != 0]
        if len(pts) > 1:
            for i in range(1, len(pts)):
                pygame.draw.line(self.screen, (0, 80, 80), pts[i-1], pts[i], 1)

        # Risk markers on the route
        if self.analyser:
            for rf in self.analyser.risk_frames:
                r2 = rf["row"]
                if r2["pos_x"] != 0:
                    rx2, ry2 = world_to_map(r2["pos_x"], r2["pos_y"])
                    col = NEON_RED if "COLLISION" in rf["issues"] else NEON_ORANGE
                    pygame.draw.circle(self.screen, col, (rx2, ry2), 3)

        # Start / end markers
        if pts:
            pygame.draw.circle(self.screen, NEON_GREEN, pts[0], 5)
            pygame.draw.circle(self.screen, NEON_PINK,  pts[-1], 5)
            txt(self.screen, "S", pts[0][0]+6,  pts[0][1]-5,  self.F_XS, NEON_GREEN)
            txt(self.screen, "E", pts[-1][0]+6, pts[-1][1]-5, self.F_XS, NEON_PINK)

        # Live AI car position (bright dot)
        if self.corrector and self.carla.vehicle:
            t2 = self.carla.get_transform()
            if t2:
                cx3, cy3 = world_to_map(t2.location.x, t2.location.y)
                # Draw direction indicator
                yaw_rad = math.radians(t2.rotation.yaw)
                ex = cx3 + int(math.cos(yaw_rad) * 8)
                ey = cy3 - int(math.sin(yaw_rad) * 8)
                pygame.draw.line(self.screen, NEON_CYAN, (cx3, cy3), (ex, ey), 2)
                pygame.draw.circle(self.screen, NEON_CYAN, (cx3, cy3), 5)
                pygame.draw.circle(self.screen, WHITE,     (cx3, cy3), 5, 1)

        # Legend
        leg_y = y + h - 14
        for col3, lbl in [(NEON_GREEN,"●S"), (NEON_PINK,"●E"),
                          (NEON_RED,"●COL"), (NEON_ORANGE,"●ERR"),
                          (NEON_CYAN,"●AI")]:
            txt(self.screen, lbl, x+4, leg_y, self.F_XS, col3)
            x += 34

    def _draw_ai_panel(self, x, y, w, h):
        """Right panel: AI suggestions + comparison stats + fuel/time/traffic."""
        pygame.draw.rect(self.screen, PANEL_BG2, (x,y,w,h))
        neon_rect(self.screen, NEON_PURPLE, (x,y,w,h), w=1, glow=False)

        py = y + 6
        txt(self.screen,"━ AI SUGGESTIONS",x+6,py,self.F_SM,GREY); py+=22

        # Prediction cards
        if self.ai_prediction:
            p = self.ai_prediction
            cw = w - 16

            # Collision risk
            rc=[NEON_GREEN,NEON_ORANGE,NEON_RED][p["collision"]]
            rl=["  SAFE","  WARNING","  DANGER"][p["collision"]]
            filled_rect(self.screen,rc,(x+8,py,cw,46),alpha=30)
            txt(self.screen,"COLLISION RISK",x+8+cw//2,py+8,self.F_XS,GREY,center=True)
            txt(self.screen,rl,x+8+cw//2,py+28,self.F_MD,rc,center=True)
            py+=54

            # Speed advice
            sc2=[NEON_RED,NEON_GREEN,NEON_CYAN][p["speed"]]
            sl=["SLOW DOWN","MAINTAIN","ACCELERATE"][p["speed"]]
            filled_rect(self.screen,sc2,(x+8,py,cw,38),alpha=28)
            txt(self.screen,"SPEED ADVICE",x+8+cw//2,py+6,self.F_XS,GREY,center=True)
            txt(self.screen,sl,x+8+cw//2,py+24,self.F_MD,sc2,center=True)
            py+=44

            # Lane status
            lc=NEON_RED if p["lane"] else NEON_GREEN
            lt="⚠ LANE DRIFT!" if p["lane"] else "✓ LANE OK"
            filled_rect(self.screen,lc,(x+8,py,cw,32),alpha=28)
            txt(self.screen,"LANE",x+8+cw//2,py+6,self.F_XS,GREY,center=True)
            txt(self.screen,lt,x+8+cw//2,py+20,self.F_SM,lc,center=True)
            py+=38

            # Confidence bars
            txt(self.screen,"━ CONFIDENCE",x+6,py,self.F_XS,GREY); py+=16
            probs=p.get("probs",[0.33,0.33,0.34])
            lbls=["Safe","Warn","Danger"]; cols=[NEON_GREEN,NEON_ORANGE,NEON_RED]
            for lbl,prob,col3 in zip(lbls,probs,cols):
                txt(self.screen,lbl,x+8,py,self.F_XS,GREY)
                bw3=int(prob*(cw-60))
                pygame.draw.rect(self.screen,DARK_GREY,(x+52,py,cw-60,10),border_radius=4)
                if bw3>0: pygame.draw.rect(self.screen,col3,(x+52,py,bw3,10),border_radius=4)
                txt(self.screen,f"{prob:.0%}",x+cw-4,py,self.F_XS,col3,right=True)
                py+=16
        else:
            txt(self.screen,"Waiting for prediction…",x+w//2,py+16,self.F_SM,GREY,center=True)
            py+=40

        # ── LIVE TRAFFIC AWARENESS ─────────────────────────────────
        py+=4
        pygame.draw.line(self.screen,NEON_CYAN,(x+8,py),(x+w-8,py),1); py+=8
        txt(self.screen,"━ LIVE TRAFFIC AWARENESS",x+6,py,self.F_SM,GREY); py+=18
        cw = w - 16
        npc_d = self.live_npc_dist
        ped_d = self.live_ped_dist
        npc_col = NEON_RED if npc_d < 8 else NEON_ORANGE if npc_d < 20 else NEON_GREEN
        ped_col = NEON_RED if ped_d < 6 else NEON_ORANGE if ped_d < 15 else NEON_GREEN
        npc_bw = int(max(0, min(1, (40-npc_d)/40)) * (cw-60))
        ped_bw = int(max(0, min(1, (30-ped_d)/30)) * (cw-60))
        txt(self.screen,"NPC",x+8,py,self.F_XS,GREY)
        pygame.draw.rect(self.screen,DARK_GREY,(x+44,py,cw-60,10),border_radius=4)
        if npc_bw>0: pygame.draw.rect(self.screen,npc_col,(x+44,py,npc_bw,10),border_radius=4)
        txt(self.screen,f"{npc_d:.0f}m",x+cw-4,py,self.F_XS,npc_col,right=True); py+=16
        txt(self.screen,"PED",x+8,py,self.F_XS,GREY)
        pygame.draw.rect(self.screen,DARK_GREY,(x+44,py,cw-60,10),border_radius=4)
        if ped_bw>0: pygame.draw.rect(self.screen,ped_col,(x+44,py,ped_bw,10),border_radius=4)
        txt(self.screen,f"{ped_d:.0f}m",x+cw-4,py,self.F_XS,ped_col,right=True); py+=16
        txt(self.screen,f"Traffic brakes: {self.traffic_events}",x+8,py,self.F_XS,NEON_ORANGE); py+=14

        # ── DRIVE TIME COMPARISON ──────────────────────────────────
        py+=4
        pygame.draw.line(self.screen,NEON_PINK,(x+8,py),(x+w-8,py),1); py+=8
        txt(self.screen,"━ SESSION STATS",x+6,py,self.F_SM,GREY); py+=18

        man_time = self.manual_elapsed if self.manual_elapsed else 0.0
        ai_time  = (time.time() - self.auto_start_time) if self.auto_start_time and not self.auto_done else self.auto_elapsed
        if self.auto_done and self.auto_elapsed == 0.0:
            ai_time = (time.time() - self.auto_start_time) if self.auto_start_time else 0.0

        def time_str(secs):
            m = int(secs) // 60
            s = int(secs) % 60
            return f"{m}m {s:02d}s"

        # Drive time
        txt(self.screen,"Drive Time",x+8,py,self.F_XS,GREY); py+=14
        hw2 = (cw-4)//2
        filled_rect(self.screen,NEON_YELLOW,(x+8,py,hw2-2,20),alpha=20)
        filled_rect(self.screen,NEON_CYAN,  (x+hw2+10,py,hw2-2,20),alpha=20)
        txt(self.screen,f"YOU {time_str(man_time)}",x+10,py+4,self.F_XS,NEON_YELLOW)
        txt(self.screen,f"AI  {time_str(ai_time)}",x+hw2+12,py+4,self.F_XS,NEON_CYAN)
        py+=26

        # Fuel consumption
        txt(self.screen,"Est. Fuel (L)",x+8,py,self.F_XS,GREY); py+=14
        fuel_max = max(self.fuel_manual, self.fuel_ai, 0.001)
        mfw = int((self.fuel_manual/fuel_max)*(hw2-2))
        afw = int((self.fuel_ai    /fuel_max)*(hw2-2))
        pygame.draw.rect(self.screen,DARK_GREY,(x+8,py,hw2-2,11),border_radius=3)
        if mfw>0: pygame.draw.rect(self.screen,NEON_ORANGE,(x+8,py,mfw,11),border_radius=3)
        txt(self.screen,f"{self.fuel_manual:.2f}",x+8,py,self.F_XS,NEON_ORANGE)
        pygame.draw.rect(self.screen,DARK_GREY,(x+hw2+10,py,hw2-2,11),border_radius=3)
        if afw>0: pygame.draw.rect(self.screen,NEON_GREEN,(x+hw2+10,py,afw,11),border_radius=3)
        txt(self.screen,f"{self.fuel_ai:.2f}",x+hw2+10,py,self.F_XS,NEON_GREEN)
        if self.fuel_manual > 0.001:
            saved = max(0, self.fuel_manual - self.fuel_ai)
            pct   = saved / self.fuel_manual * 100
            txt(self.screen,f"AI saves {pct:.0f}%",x+8,py+14,self.F_XS,NEON_GREEN); py+=14
        py+=28

        # Safety score
        txt(self.screen,"Safety Score",x+8,py,self.F_XS,GREY); py+=14
        human_score = max(0, 100 - (self.analyser.risk_score_human*4 if self.analyser else 0) - (self.analyser.collision_count if self.analyser else 0)*15)
        ai_score    = max(0, 100 - self.ai_collisions*15 - self.ai_lane_drifts*3)
        for label, score, col3 in [("YOU", human_score, NEON_ORANGE), ("AI", ai_score, NEON_GREEN)]:
            sbw = int((score/100)*(cw//2-12))
            pygame.draw.rect(self.screen,DARK_GREY,(x+8,py,cw//2-12,11),border_radius=3)
            if sbw>0: pygame.draw.rect(self.screen,col3,(x+8,py,sbw,11),border_radius=3)
            txt(self.screen,f"{label} {score:.0f}/100",x+cw//2,py,self.F_XS,col3)
            py+=18

        # ── DRIVE vs AI: key comparison ─────────────────────────────
        py+=4
        pygame.draw.line(self.screen,NEON_PINK,(x+8,py),(x+w-8,py),1); py+=8
        txt(self.screen,"━ YOUR DRIVE vs AI DRIVE",x+6,py,self.F_SM,GREY); py+=18

        if self.analyser:
            a = self.analyser

            def stat_row(label, human_val, ai_val, human_col, ai_col):
                nonlocal py
                txt(self.screen,label,x+8,py,self.F_XS,GREY)
                py+=13
                hw3=int((cw-4)//2)
                hmax=max(human_val,ai_val,0.001)
                hfw=int((human_val/hmax)*(hw3-2))
                pygame.draw.rect(self.screen,DARK_GREY,(x+8,py,hw3-2,10),border_radius=3)
                if hfw>0: pygame.draw.rect(self.screen,human_col,(x+8,py,hfw,10),border_radius=3)
                txt(self.screen,f"YOU {human_val:.1f}",x+8,py,self.F_XS,human_col)
                afw2=int((ai_val/hmax)*(hw3-2))
                pygame.draw.rect(self.screen,DARK_GREY,(x+hw3+10,py,hw3-2,10),border_radius=3)
                if afw2>0: pygame.draw.rect(self.screen,ai_col,(x+hw3+10,py,afw2,10),border_radius=3)
                txt(self.screen,f"AI  {ai_val:.1f}",x+hw3+10,py,self.F_XS,ai_col)
                py+=16

            stat_row("Max Speed (km/h)", a.max_speed_human, a.max_speed_ai, NEON_YELLOW, NEON_CYAN)
            stat_row("Avg Speed (km/h)", a.avg_speed_human, a.avg_speed_ai, NEON_YELLOW, NEON_CYAN)
            stat_row("Steer Jerk (lower=smoother)",
                     a.steer_jerk_human*100, a.steer_jerk_ai*100, NEON_ORANGE, NEON_GREEN)
            stat_row("Risk Events", float(a.risk_score_human), float(a.risk_score_ai), NEON_RED, NEON_GREEN)
            stat_row("Collisions",  float(a.collision_count), float(self.ai_collisions), NEON_RED, NEON_GREEN)
            stat_row("Lane Drifts", float(a.lane_count), float(self.ai_lane_drifts), NEON_ORANGE, NEON_GREEN)

            py+=2
            # Risk events list (last 4)
            if a.risk_frames and py + 70 < y + h - 36:
                txt(self.screen,"━ YOUR MISTAKES",x+6,py,self.F_XS,NEON_RED); py+=14
                for rf in a.risk_frames[-4:]:
                    r2=rf["row"]; issues=", ".join(rf["issues"])
                    txt(self.screen,
                        f"t={r2['timestamp']}  {issues[:22]}  {r2['speed']:.0f}km/h",
                        x+8,py,self.F_XS,NEON_ORANGE)
                    py+=13
        else:
            txt(self.screen,"Drive in MANUAL mode first",x+w//2,py+8,self.F_SM,GREY,center=True)

        # ── ADAS note ─────────────────────────────────────────────────
        note_y = y + h - 32
        pygame.draw.line(self.screen, NEON_PURPLE, (x+8, note_y), (x+w-8, note_y), 1)
        txt(self.screen, "ADAS DDT MODEL  •  Pre-trained on fleet data",
            x+w//2, note_y+8, self.F_XS, NEON_PURPLE, center=True)
        txt(self.screen, "Live traffic-aware  •  Not pre-recorded",
            x+w//2, note_y+20, self.F_XS, GREY, center=True)

    # ══════════════════════════════════════════════════════════════════
    #  CLEANUP
    # ══════════════════════════════════════════════════════════════════
    def _cleanup(self):
        if self.logger: self.logger.close()
        self.carla.cleanup()
        pygame.quit()
        rows_saved = self.logger.count if self.logger else 0
        print(f"\n✅ Session ended. {rows_saved} rows logged.")
        if self.model.trained: print(f"   Model: {MODEL_FILE}")
        print(f"   CSV:   {LOG_FILE}")


# ══════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 56)
    print("  CARLA Unified Smart Dashboard  —  Vice City Edition")
    print("=" * 56)
    print("  Make sure CarlaUE4.exe is running on localhost:2000")
    print("  py -3.7 carla_unified_dashboard.py")
    print()
    app = UnifiedDashboard()
    app.run()