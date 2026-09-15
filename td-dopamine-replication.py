import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

#define cue times, and key variables 
CS = 10 
DELAY = 8 #delay between cue and reward, in terms of time-step 
US = CS + DELAY 
T = US + 8 
gamma = 0.98 #discount factor (γ) 
alpha = 0.3 #learning rate (α) set to 0.3

#setting up the core equation
def x(t, N):
    #x(t) represents a vector of signals; one per possible delay, each corresponding to a weight 
    vector = np.zeros(N)
    i = t - CS
    if 0 <= i < N: #only significant when timestep is past the CS
        vector[i] = 1.0
    return vector

#define prediction V(t)
def V_hat(t,w,N):
    return np.dot(w, x(t,N))

#define error δ(t) as per equation 3 in paper
def delta(t, w, r, N):
    return r + (gamma * V_hat(t+1, w, N)) - V_hat(t,w, N)

#define change in weights as per equation 5 in paper
def run_trial(w, us_t, N, reward=True, learn=True): #us_t defines the time-step where US is given
    deltas = np.zeros(T) # sets up an array for deltas 
    for t in range (T-1):
        r = 1.0 if (reward and (t+1 == us_t)) else 0 #sets r=1 if we are at the moment that reward is scheduled for AND reward can happen at this trial (it is not ommitted)
        d = delta(t,w,r, N)
        deltas[t+1] = d #records at t+1 point in error array
        if learn: 
            w+= alpha*x(t,N)*d #equation 5
    return deltas   
    
#recreating setup for Figure 3
N_original=8 
w = np.zeros(N_original)
n_train_trials=60
n_omit_trials=3

all_deltas=[] #collects deltas
#60 training trials setup
for trial in range(n_train_trials):
    d = run_trial(w, US, N_original, reward=True, learn=True)
    all_deltas.append(d)
#3 ommission trials setup
for trial in range (n_omit_trials):
    d = run_trial(w.copy(), US, N_original, reward=False, learn=False)
    all_deltas.append(d) #note that ommitted trials will be at the end of the trial list 

#saving data plot heat-map to recreate Figure 3 - ORIGINAL CONDITIONS
heatmap_data = np.array(all_deltas)

#ADDED STOCHASTIC ELEMENT - "jitter" in the timesteps between cue time and reward time 
rng = np.random.default_rng(0) #seeded vs unseeded 
jitter = 2 #jitter is US +/- up to 2 steps
N_jitter = DELAY + jitter + 1 

#train model with jitter timing 
w_jitter = np.zeros(N_jitter)
all_deltas_jitter = []

for trial in range(n_train_trials):
    us_t = US + rng.integers(-jitter, jitter + 1)
    d=run_trial(w_jitter, us_t, N_jitter, reward=True, learn=True)
    all_deltas_jitter.append(d)

for trial in range(n_omit_trials):
    d = run_trial(w_jitter.copy(), US, N_jitter, reward=False, learn=False)
    all_deltas_jitter.append(d)

heatmap_data_jitter = np.array(all_deltas_jitter)

#size-by-size comparison plot of Figure 3, and stochastic element added
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

vmax = max(np.abs(heatmap_data).max(), np.abs(heatmap_data_jitter).max())

for ax, data, title in [
    (ax1, heatmap_data,        "fixed reward timing\n(deterministic)"),
    (ax2, heatmap_data_jitter, f"jittered reward timing\n(stochastic, ±{jitter} steps)"),
]:
    im = ax.pcolormesh(
        np.arange(T), np.arange(data.shape[0]), data,
        cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto"
    )
    ax.axvline(CS, color="black", linestyle="--", linewidth=1)
    ax.axvline(US, color="black", linestyle="--", linewidth=1)
    ax.text(CS, -2, "CS", ha="center", fontsize=9)
    ax.text(US, -2, "R", ha="center", fontsize=9)
    ax.axhline(n_train_trials - 0.5, color="black", linewidth=1)
    ax.set_xlabel("Time step within trial")
    ax.set_title(title, fontsize=11)

ax1.set_ylabel("trial number")
fig.colorbar(im, ax=[ax1, ax2], label=r"TD error δ(t)", fraction=0.025, pad=0.02)
fig.suptitle("effect on randomized reward timing on TD error δ(t)", fontsize=13, fontweight="bold")
plt.rcParams["font.family"] = "sans-serif"
plt.show()