# Replicating Schultz, Dayan & Montague (1997): A TD-Learning Model of Dopamine

Tasked with replicating Schultz, Dayan, and Montague's TD model and figure from their 1997 publication *A Neural Substrate for Prediction and Reward*, I decided to build a TD(0) model that used the paper's "complete serial-compound" stimulus representation.

## What I found

Training this on 60 trials of a cue-reward task, where the time-step of the cue and reward were set constant at t=10 and t=18, the model successfully reproduced the three signature cases. In the earliest trials, δ spikes only at the moment of reward. After about 60 trials, the weights converge to 1 for every delay, and the δ spike moves entirely to the cue: the reward itself produces no surprise. Then, once the reward is omitted, the trained model produces a negative dip in δ at the time-step where the reward was expected, while the positive error at the cue time-step is maintained. A trial vs time heatmap of δ (the left graph of the figure) across training showcases this pattern directly. We can visibly see the error propagate backward one weight per trial, moving from the reward towards the cue.

## Where my version differs from the paper

One area of difference between my version and the original paper's is my creative liberty with the exact simulation hyperparameters (learning rate and discount factor), meaning that my replication wasn't an exact quantitative replication. I decided to use α=0.3, 60 trials, and γ=0.98 (highly far-sighted) as reasonable choices.

Another interesting divergence was an extension I added beyond the paper: making the task stochastic rather than deterministic by jittering the reward's arrival time by ±2 steps. My thought-process behind this was to create a more biologically relevant result: from trial to trial, real reward timing is rarely fixed in animal studies, and introducing temporal variability allowed me to examine how a TD model distributes prediction error when the exact timing of the reward is uncertain. To do so, I had to widen the detector array x(t) to cover the full jitter window, and the trained-jitter model (the right graph of the figure) revealed a smooth spread of residual error around the expected reward time that lowers in magnitude as the trial number increases, reflecting the model still converging to the best possible average prediction across the jitter window (the original cue-time itself).

## What I tried that didn't work

The major setback I had was attempting to match the paper's actual figure cues, which were a cue-to-reward delay of 40 time-steps and 40 trials of training. However, with a plain TD(0) model, the backpropagation of the error can only move one time-step per trial, which makes it impossible for my model to fully replicate the figure's showings. This likely means that the paper's figure, displaying a convergence over 40 time-steps within 30 trials, was not done with a pure one-step model as I implemented it.

## One thing I'd try next

Something I'd try next is to implement eligibility traces (TD(λ)) in the model, which would allow prediction errors from the reward to update multiple preceding time-steps rather than just the most recent one, potentially enabling the model to reproduce Figure 3's faster back-propagation.

Overall, this project provided me with a deeper understanding of how temporal difference learning can model the dynamics of reward prediction, and I'm compelled to see how this principle can be refined and expanded upon via future brain-derived reinforcement learning work.
