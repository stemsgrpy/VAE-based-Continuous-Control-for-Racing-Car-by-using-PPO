# VAE-based-Continuous-Control-for-Racing-Car-by-using-PPO

## Dataset
- Record images through **playing Racing by yourself**     
```
    python record_data.py
```

## Variational AutoEncoder VAE
- Encoder  
  - Mean, Standard deviation (Gaussian distribution)    
- Decoder   
- KL Divergence   

### Train
```
    python vae.py --train
```
<p align="center">
  <img src="/README/VAE0.png" alt="Description" width="393" height="442" border="0" />
  <img src="/README/VAE1.png" alt="Description" width="393" height="442" border="0" />
  <img src="/README/VAE2.png" alt="Description" width="393" height="442" border="0" />
  <img src="/README/VAE3.png" alt="Description" width="393" height="442" border="0" />
</p>
<p align="center">
  Figure 1: VAE Training
</p>

### Test
```
    python vae.py --test
```
<p align="center">
  <img src="/README/ImageSource.jpg" alt="Description" width="120" height="120" border="0" />
  <img src="/README/ImageSource.jpg" alt="Description" width="120" height="120" border="0" />
</p>
<p align="center">
  Figure 2: VAE Testing
</p>

## Input to Output
- State (Input)  
  - **Use pre-trained model VAE**
    - **Only Encoder part**  
```
    config.state_dim = 20 # Encoder Feature
```

- Action (Output)  
  - **Continuous**   
```
    config.action_dim = env.action_space.shape[0]
```

## Reinforcement Learning PPO
### Train
```
python PPO.py --train --env CarRacing-v0

```

### Test
```
python PPO.py --test --env CarRacing-v0 --model_path out/CarRacing-v0-runx/policy_xxxx.pkl
```

### Retrain
```
python PPO.py --retrain --env CarRacing-v0 --model_path out/CarRacing-v0-runx/checkpoint_policy/checkpoint_fr_xxxxx.tar
```

## Result

<p align="center">
  <img src="/README/CarRacing-v0.gif" alt="Description" width="600" height="480" border="0" />
</p>
<p align="center">
  Figure 3: Reinforcement Learning PPO on Racing Car
</p>

## Reference
[Manually Driving CarRacing-v0](https://cdancette.fr/2018/04/09/self-driving-CNN)  
https://github.com/stemsgrpy/Control-Task-for-Box2D-Simulator-by-using-PPO    