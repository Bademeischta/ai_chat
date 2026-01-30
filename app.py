import eventlet
eventlet.monkey_patch()

import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from threading import Lock

# Constants
ARENA_SIZE = 2000
TICK_RATE = 30
DT = 1.0 / TICK_RATE
FOOD_COUNT = 100
MAX_BOT_SPEED = 200
BASE_RADIUS = 20
MASS_LOSS_RATE = 0.02
RAY_COUNT = 16
RAY_RANGE = 400

class SpatialHash:
    def __init__(self, size=2000, cell_size=100):
        self.cell_size = cell_size
        self.grid = {}

    def _hash(self, x, y):
        return (int(x // self.cell_size), int(y // self.cell_size))

    def clear(self):
        self.grid = {}

    def insert(self, obj):
        h = self._hash(obj.x, obj.y)
        if h not in self.grid:
            self.grid[h] = []
        self.grid[h].append(obj)

    def query(self, x, y, radius):
        cells = []
        for i in range(int((x - radius) // self.cell_size), int((x + radius) // self.cell_size) + 1):
            for j in range(int((y - radius) // self.cell_size), int((y + radius) // self.cell_size) + 1):
                cells.append((i, j))
        
        found = []
        for h in cells:
            if h in self.grid:
                found.extend(self.grid[h])
        return found

class Food:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.radius = 5

class Bot:
    def __init__(self, bot_id, algo_name, color, brain):
        self.id = bot_id
        self.algo_name = algo_name
        self.color = color
        self.brain = brain
        self.respawn()
        self.kills = 0
        self.deaths = 0
        self.foods_eaten = 0
        self.total_reward = 0
        self.age = 0

    def respawn(self):
        self.x = random.uniform(50, ARENA_SIZE - 50)
        self.y = random.uniform(50, ARENA_SIZE - 50)
        self.vx = 0; self.vy = 0
        self.mass = 50
        self.radius = BASE_RADIUS
        self.boost = False

    def update_radius(self):
        self.radius = BASE_RADIUS * (self.mass / 50)**0.5

    def get_max_speed(self):
        speed = MAX_BOT_SPEED * (50 / self.mass)**0.3
        if self.boost: speed *= 1.5
        return speed

    def step(self, action_idx):
        angle_idx = action_idx % 8
        self.boost = action_idx >= 8
        angle = angle_idx * (2 * np.pi / 8)
        target_vx = np.cos(angle) * self.get_max_speed()
        target_vy = np.sin(angle) * self.get_max_speed()
        accel = 10
        self.vx += (target_vx - self.vx) * accel * DT
        self.vy += (target_vy - self.vy) * accel * DT
        self.x += self.vx * DT
        self.y += self.vy * DT
        penalty = 0
        if self.x < self.radius: self.x = self.radius; self.vx *= -0.5; penalty = -2
        if self.x > ARENA_SIZE - self.radius: self.x = ARENA_SIZE - self.radius; self.vx *= -0.5; penalty = -2
        if self.y < self.radius: self.y = self.radius; self.vy *= -0.5; penalty = -2
        if self.y > ARENA_SIZE - self.radius: self.y = ARENA_SIZE - self.radius; self.vy *= -0.5; penalty = -2
        loss = MASS_LOSS_RATE * self.mass * DT
        if self.boost: loss *= 5
        self.mass = max(20, self.mass - loss)
        self.update_radius()
        self.age += 1
        return 1.0/TICK_RATE + penalty

class GameWorld:
    def __init__(self):
        self.bots = []
        self.food = [Food(random.uniform(0, ARENA_SIZE), random.uniform(0, ARENA_SIZE)) for _ in range(FOOD_COUNT)]
        self.spatial_hash = SpatialHash()
        self.tick_count = 0
        self.best_fitness = 0

    def get_obs(self, bot):
        obs = []
        for i in range(RAY_COUNT):
            angle = i * (2 * np.pi / RAY_COUNT)
            dx = np.cos(angle); dy = np.sin(angle)
            ray_res = [1.0, 0, 0, 0, 0]
            d_wall = RAY_RANGE
            if dx > 0: d_wall = min(d_wall, (ARENA_SIZE - bot.x) / dx)
            elif dx < 0: d_wall = min(d_wall, (0 - bot.x) / dx)
            if dy > 0: d_wall = min(d_wall, (ARENA_SIZE - bot.y) / dy)
            elif dy < 0: d_wall = min(d_wall, (0 - bot.y) / dy)
            if d_wall < RAY_RANGE: ray_res = [d_wall / RAY_RANGE, 0, 0, 0, 1]
            nearby = self.spatial_hash.query(bot.x, bot.y, RAY_RANGE)
            for other in nearby:
                if other == bot: continue
                ox, oy = other.x - bot.x, other.y - bot.y
                dist = (ox**2 + oy**2)**0.5
                if dist > RAY_RANGE: continue
                proj = ox * dx + oy * dy
                if proj < 0: continue
                perp_dist_sq = (ox**2 + oy**2) - proj**2
                if perp_dist_sq < other.radius**2:
                    hit_dist = proj - (other.radius**2 - perp_dist_sq)**0.5
                    if hit_dist < ray_res[0] * RAY_RANGE:
                        ray_res = [hit_dist / RAY_RANGE, 0, 0, 0, 0]
                        if isinstance(other, Food): ray_res[1] = 1
                        else:
                            if other.mass * 1.2 < bot.mass: ray_res[2] = 1
                            elif bot.mass * 1.2 < other.mass: ray_res[3] = 1
                            else: ray_res[4] = 1
            obs.extend(ray_res)
        obs.append(bot.mass / 500.0)
        obs.append((bot.vx**2 + bot.vy**2)**0.5 / MAX_BOT_SPEED)
        min_wall = min(bot.x, ARENA_SIZE - bot.x, bot.y, ARENA_SIZE - bot.y)
        obs.append(min_wall / ARENA_SIZE)
        return np.array(obs, dtype=np.float32)

    def update(self):
        self.tick_count += 1
        self.spatial_hash.clear()
        for f in self.food: self.spatial_hash.insert(f)
        for b in self.bots: self.spatial_hash.insert(b)
        for b in self.bots:
            obs = self.get_obs(b)
            action = b.brain.decide(obs)
            reward = b.step(action)
            nearby = self.spatial_hash.query(b.x, b.y, b.radius + 50)
            for other in nearby:
                if other == b: continue
                dx, dy = other.x - b.x, other.y - b.y
                dist = (dx**2 + dy**2)**0.5
                if dist < b.radius + other.radius:
                    if isinstance(other, Food):
                        if other in self.food:
                            b.mass += 5; b.foods_eaten += 1; reward += 10
                            self.food.remove(other)
                            self.food.append(Food(random.uniform(0, ARENA_SIZE), random.uniform(0, ARENA_SIZE)))
                    elif isinstance(other, Bot):
                        if b.mass > 1.2 * other.mass:
                            b.mass += other.mass * 0.5; b.kills += 1; reward += 100
                            other.deaths += 1; other.total_reward -= 100
                            other.brain.on_step(self.get_obs(other), 0, -100, self.get_obs(other), True)
                            other.respawn()
            b.total_reward += reward
            self.best_fitness = max(self.best_fitness, b.total_reward)
            b.brain.on_step(obs, action, reward, self.get_obs(b), False)

class BaseBrain:
    def decide(self, obs): return 0
    def on_step(self, obs, action, reward, next_obs, done): pass
    def metric_string(self): return ""
    def save(self, folder, name): pass
    def load(self, folder, name): pass

class RandomBrain(BaseBrain):
    def __init__(self): self.current_action = 0
    def decide(self, obs):
        if random.random() < 0.1: self.current_action = random.randint(0, 15)
        return self.current_action
    def metric_string(self): return f"act:{self.current_action}"

class DecisionTreeBrain(BaseBrain):
    def decide(self, obs):
        bfd = 1.0; bfa = -1; btd = 1.0; bta = -1; bpd = 1.0; bpa = -1
        for i in range(16):
            base = i * 5; d = obs[base]
            if obs[base+1] > 0.5 and d < bfd: bfd = d; bfa = i
            if obs[base+2] > 0.5 and d < bpd: bpd = d; bpa = i
            if obs[base+3] > 0.5 and d < btd: btd = d; bta = i
        self.mode = "WANDER"; action = random.randint(0, 7)
        if btd < 0.5: self.mode = "FLEE"; action = (bta + 8) % 8 + 8
        elif bpd < 0.8: self.mode = "CHASE"; action = bpa + 8
        elif bfd < 1.0: self.mode = "EAT"; action = bfa
        return action
    def metric_string(self): return getattr(self, 'mode', 'IDLE')

class PotentialFieldBrain(BaseBrain):
    def decide(self, obs):
        fx, fy = 0, 0
        for i in range(16):
            angle = i * (2 * np.pi / 16); dx, dy = np.cos(angle), np.sin(angle)
            base = i * 5; d = obs[base]
            if obs[base+1] > 0.5: fx += dx / (d + 0.1); fy += dy / (d + 0.1)
            if obs[base+2] > 0.5: fx += 2 * dx / (d + 0.1); fy += 2 * dy / (d + 0.1)
            if obs[base+3] > 0.5: fx -= 5 * dx / (d + 0.1); fy -= 5 * dy / (d + 0.1)
            if obs[base+4] > 0.5: fx -= dx / (d + 0.1); fy -= dy / (d + 0.1)
        self.force = (fx**2 + fy**2)**0.5
        angle = np.arctan2(fy, fx)
        return int(((angle / (2*np.pi) * 8) + 8) % 8) + (8 if self.force > 5 else 0)
    def metric_string(self): return f"f:{self.force:.1f}"

class PIDControllerBrain(BaseBrain):
    def __init__(self): self.error_sum = 0; self.prev_error = 0; self.target_angle = 0
    def decide(self, obs):
        best_d = 1.1; best_a = 0
        for i in range(16):
            if obs[i*5+1] > 0.5 and obs[i*5] < best_d: best_d = obs[i*5]; best_a = i * (2*np.pi/16)
        error = (best_a - self.target_angle + np.pi) % (2*np.pi) - np.pi
        self.error_sum += error; diff = error - self.prev_error
        self.target_angle = (self.target_angle + 1.0 * error + 0.01 * self.error_sum + 0.5 * diff) % (2*np.pi)
        self.prev_error = error; self.last_error = error
        return int(((self.target_angle / (2*np.pi) * 8) + 8) % 8)
    def metric_string(self): return f"err:{self.last_error:.2f}"

class HeuristicSearchBrain(BaseBrain):
    def decide(self, obs):
        best_s = -1e9; best_act = 0
        for a in range(8):
            angle = a * (2*np.pi/8); dx, dy = np.cos(angle), np.sin(angle); score = 0
            for i in range(16):
                ra = i * (2*np.pi/16); rdx, rdy = np.cos(ra), np.sin(ra); dot = dx*rdx + dy*rdy; base = i*5; d = obs[base]
                if obs[base+1] > 0.5: score += dot * (1.0 - d)
                if obs[base+2] > 0.5: score += 2 * dot * (1.0 - d)
                if obs[base+3] > 0.5: score -= 5 * dot * (1.0 - d)
            if score > best_s: best_s = score; best_act = a
        self.best_s = best_s
        return best_act
    def metric_string(self): return f"s:{self.best_s:.1f}"

class TabularQBrain(BaseBrain):
    def __init__(self): self.q_table = {}; self.epsilon = 0.5
    def _get_state(self, obs):
        bd = 1.0; bt = 0; ba = 0
        for i in range(16):
            if obs[i*5] < bd: bd = obs[i*5]; bt = np.argmax(obs[i*5+1:i*5+5]) + 1; ba = i // 2
        return (bt, ba, int(bd * 5))
    def decide(self, obs):
        s = self._get_state(obs)
        if random.random() < self.epsilon: return random.randint(0, 7)
        return np.argmax(self.q_table.get(s, np.zeros(8)))
    def on_step(self, obs, action, reward, next_obs, done):
        s = self._get_state(obs); ns = self._get_state(next_obs); q = self.q_table.get(s, np.zeros(8))
        q[action % 8] += 0.1 * (reward + 0.9 * np.max(self.q_table.get(ns, np.zeros(8))) - q[action % 8])
        self.q_table[s] = q; self.epsilon = max(0.05, self.epsilon * 0.9999)
    def metric_string(self): return f"e:{self.epsilon:.2f}"
    def save(self, folder, name): np.save(os.path.join(folder, f"{name}_q.npy"), self.q_table)
    def load(self, folder, name):
        p = os.path.join(folder, f"{name}_q.npy")
        if os.path.exists(p): self.q_table = np.load(p, allow_pickle=True).item()

class SARSABrain(TabularQBrain):
    def on_step(self, obs, action, reward, next_obs, done):
        s = self._get_state(obs); ns = self._get_state(next_obs); na = self.decide(next_obs); q = self.q_table.get(s, np.zeros(8))
        q[action % 8] += 0.1 * (reward + 0.9 * self.q_table.get(ns, np.zeros(8))[na % 8] - q[action % 8])
        self.q_table[s] = q; self.epsilon = max(0.05, self.epsilon * 0.9999)

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, out_dim))
    def forward(self, x): return self.net(x)

class DQNBrain(BaseBrain):
    def __init__(self, in_dim=83, out_dim=16):
        self.model = MLP(in_dim, out_dim); self.target = MLP(in_dim, out_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.memory = []; self.epsilon = 0.5; self.step_cnt = 0
    def decide(self, obs):
        if random.random() < self.epsilon: return random.randint(0, 15)
        with torch.no_grad(): return torch.argmax(self.model(torch.from_numpy(obs))).item()
    def on_step(self, obs, action, reward, next_obs, done):
        self.memory.append((obs, action, reward, next_obs, done))
        if len(self.memory) > 5000: self.memory.pop(0)
        self.step_cnt += 1
        if self.step_cnt % 60 == 0 and len(self.memory) > 64: self.train()
        if self.step_cnt % 500 == 0: self.target.load_state_dict(self.model.state_dict())
        self.epsilon = max(0.1, self.epsilon * 0.9999)
    def train(self):
        batch = random.sample(self.memory, 64)
        states = torch.stack([torch.from_numpy(s) for s, a, r, ns, d in batch])
        actions = torch.tensor([a for s, a, r, ns, d in batch])
        rewards = torch.tensor([r for s, a, r, ns, d in batch], dtype=torch.float32)
        next_states = torch.stack([torch.from_numpy(ns) for s, a, r, ns, d in batch])
        dones = torch.tensor([d for s, a, r, ns, d in batch], dtype=torch.float32)
        qs = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()
        next_qs = self.target(next_states).max(1)[0]
        targets = rewards + 0.99 * next_qs * (1 - dones)
        loss = nn.MSELoss()(qs, targets.detach())
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
    def metric_string(self): return f"e:{self.epsilon:.2f}"
    def save(self, folder, name): torch.save(self.model.state_dict(), os.path.join(folder, f"{name}.pth"))
    def load(self, folder, name):
        p = os.path.join(folder, f"{name}.pth")
        if os.path.exists(p):
            d = torch.load(p)
            if isinstance(d, dict) and 'w' in d: self.model.load_state_dict(d['w'])
            else: self.model.load_state_dict(d)

class ActorCriticBrain(BaseBrain):
    def __init__(self, in_dim=83, out_dim=16):
        self.actor = MLP(in_dim, out_dim); self.critic = MLP(in_dim, 1)
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=1e-3)
        self.memory = []; self.step_cnt = 0
    def decide(self, obs):
        with torch.no_grad():
            probs = torch.softmax(self.actor(torch.from_numpy(obs)), dim=0)
            return torch.multinomial(probs, 1).item()
    def on_step(self, obs, action, reward, next_obs, done):
        self.memory.append((obs, action, reward, next_obs, done))
        self.step_cnt += 1
        if len(self.memory) > 128: self.train(); self.memory = []
    def train(self):
        batch = self.memory
        states = torch.stack([torch.from_numpy(s) for s, a, r, ns, d in batch])
        actions = torch.tensor([a for s, a, r, ns, d in batch])
        rewards = torch.tensor([r for s, a, r, ns, d in batch], dtype=torch.float32)
        next_states = torch.stack([torch.from_numpy(ns) for s, a, r, ns, d in batch])
        dones = torch.tensor([d for s, a, r, ns, d in batch], dtype=torch.float32)
        v = self.critic(states).squeeze(); nv = self.critic(next_states).squeeze()
        targets = rewards + 0.99 * nv * (1 - dones); adv = targets - v
        probs = torch.softmax(self.actor(states), dim=1)
        log_p = torch.log(probs.gather(1, actions.unsqueeze(1)).squeeze() + 1e-10)
        loss = -(log_p * adv.detach()).mean() + 0.5 * nn.MSELoss()(v, targets.detach())
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
    def metric_string(self): return "A2C"
    def save(self, folder, name): torch.save({'a': self.actor.state_dict(), 'c': self.critic.state_dict()}, os.path.join(folder, f"{name}.pth"))
    def load(self, folder, name):
        p = os.path.join(folder, f"{name}.pth")
        if os.path.exists(p): d = torch.load(p); self.actor.load_state_dict(d['a']); self.critic.load_state_dict(d['c'])

class GeneticBrain(BaseBrain):
    def __init__(self, in_dim=83, out_dim=16):
        self.model = MLP(in_dim, out_dim); self.gen = 0; self.best_fitness = -1e9; self.cur_fitness = 0; self.foods = 0
    def decide(self, obs):
        with torch.no_grad(): return torch.argmax(self.model(torch.from_numpy(obs))).item()
    def on_step(self, obs, action, reward, next_obs, done):
        self.cur_fitness += reward
        if reward > 5: self.foods += 1
        if self.foods >= 5 or done:
            if self.cur_fitness > self.best_fitness: self.best_fitness = self.cur_fitness
            else:
                for p in self.model.parameters(): p.data += torch.randn_like(p.data) * 0.1
            self.cur_fitness = 0; self.foods = 0; self.gen += 1
    def metric_string(self): return f"G:{self.gen}"
    def save(self, folder, name):
        torch.save({'w': self.model.state_dict(), 'g': self.gen, 'bf': self.best_fitness}, os.path.join(folder, f"{name}.pth"))
    def load(self, folder, name):
        p = os.path.join(folder, f"{name}.pth")
        if os.path.exists(p):
            d = torch.load(p)
            if isinstance(d, dict) and 'w' in d:
                self.model.load_state_dict(d['w']); self.gen = d.get('g', 0); self.best_fitness = d.get('bf', -1e9)
            else: self.model.load_state_dict(d)

class BNSDBrain(BaseBrain):
    def __init__(self, in_dim=83, out_dim=16):
        self.model = MLP(in_dim, out_dim); self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.novelty = {}; self.hof = []; self.cur_traj = []; self.step_cnt = 0
    def decide(self, obs):
        with torch.no_grad():
            if random.random() < 0.1: return random.randint(0, 15)
            return torch.argmax(self.model(torch.from_numpy(obs))).item()
    def on_step(self, obs, action, reward, next_obs, done):
        sb = tuple((obs[::10] * 5).astype(int))
        self.novelty[(sb, action)] = self.novelty.get((sb, action), 0) + 1
        nr = 1.0 / (self.novelty[(sb, action)]**0.5)
        self.cur_traj.append((obs, action, reward + nr))
        if done or len(self.cur_traj) >= 200:
            ret = sum(t[2] for t in self.cur_traj)
            if len(self.hof) < 10 or ret > min([h[0] for h in self.hof] + [-1e9]):
                self.hof.append((ret, self.cur_traj)); self.hof.sort(key=lambda x: x[0], reverse=True); self.hof = self.hof[:10]
            self.cur_traj = []
        self.step_cnt += 1
        if self.step_cnt % 100 == 0 and len(self.hof) > 0:
            traj = random.choice(self.hof)[1]; batch = random.sample(traj, min(len(traj), 32))
            s = torch.stack([torch.from_numpy(x[0]) for x in batch]); a = torch.tensor([x[1] for x in batch])
            loss = nn.CrossEntropyLoss()(self.model(s), a)
            self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
    def metric_string(self): return f"nov:{len(self.novelty)}"
    def save(self, folder, name): torch.save({'w': self.model.state_dict(), 'n': self.novelty, 'h': self.hof}, os.path.join(folder, f"{name}.pth"))
    def load(self, folder, name):
        p = os.path.join(folder, f"{name}.pth")
        if os.path.exists(p):
            d = torch.load(p)
            if isinstance(d, dict) and 'w' in d:
                self.model.load_state_dict(d['w']); self.novelty = d.get('n', {}); self.hof = d.get('h', [])
            else: self.model.load_state_dict(d)

app = Flask(__name__)
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*")
world = GameWorld()
lock = Lock()
SAVE_FOLDER = "/content/drive/MyDrive/IO_Game_Saves/" if os.path.exists("/content/drive") else "IO_Game_Saves/"
os.makedirs(SAVE_FOLDER, exist_ok=True)
ALGO_TYPES = [
    ("Random", RandomBrain, "#888888"), ("DecisionTree", DecisionTreeBrain, "#ff5555"),
    ("PotentialField", PotentialFieldBrain, "#55ff55"), ("PID", PIDControllerBrain, "#5555ff"),
    ("Heuristic", HeuristicSearchBrain, "#ffff55"), ("TabularQ", TabularQBrain, "#55ffff"),
    ("SARSA", SARSABrain, "#ff55ff"), ("DQN", DQNBrain, "#ff9900"),
    ("ActorCritic", ActorCriticBrain, "#9900ff"), ("Genetic", GeneticBrain, "#00ff99"),
    ("Novel-BNSD", BNSDBrain, "#ffffff")
]
def init_world():
    for name, brain_cls, color in ALGO_TYPES:
        for i in range(2):
            bot_id = f"{name.lower()}_{i}"; brain = brain_cls(); brain.load(SAVE_FOLDER, bot_id)
            world.bots.append(Bot(bot_id, name, color, brain))
@app.route('/')
def index(): return render_template('index.html')
def game_loop():
    while True:
        with lock:
            world.update()
            if world.tick_count % 1000 == 0:
                for b in world.bots: b.brain.save(SAVE_FOLDER, b.id)
            bot_data = [{'id': b.id, 'x': round(b.x, 1), 'y': round(b.y, 1), 'r': round(b.radius, 1), 'c': b.color, 'algo': b.algo_name, 'm': b.brain.metric_string(), 'mass': int(b.mass), 'k': b.kills, 'd': b.deaths} for b in world.bots]
            food_data = [{'x': round(f.x, 1), 'y': round(f.y, 1)} for f in world.food]
            socketio.emit('update', {'bots': bot_data, 'food': food_data, 'tick': world.tick_count, 'best': int(world.best_fitness), 'alive': len(world.bots)})
        eventlet.sleep(DT)
if __name__ == '__main__':
    init_world()
    socketio.start_background_task(game_loop)
    socketio.run(app, host='0.0.0.0', port=5000)
