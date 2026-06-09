# explore-exploit-offline-rl

A study of **offline reinforcement learning** on continuous control tasks.

We investigate how different **data collection strategies** — uniform state-action space exploration, exploitation of an optimal online-learned policy, and noisy variants in between — affect the quality of an offline-learned policy. 

**Algorithms:** TD3, IQL  
**Environments:** CartPoleContinuous-v1, Pendulum-v1, MountainCarContinuous-v0  
**Frameworks:** [BBRL](https://github.com/osigaud/bbrl), [d3rlpy](https://github.com/takuseno/d3rlpy), PyTorch

