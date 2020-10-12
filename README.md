# VAE-based-Continuous-Control-for-Racing-Car-by-using-PPO

## Dataset
- Record images through **playing Racing Car by yourself**     
  - Tips     
    - Decrease straight (by accelerating)     
    - Increase curve    (by decelerating)     
    - Casually play     (data diversity)     

```
    python record_data.py
```

## Variational AutoEncoder VAE

- **Simple VAE**

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Linear-1              [-1, 72, 400]       3,686,800
            Linear-2               [-1, 72, 20]           8,020
            Linear-3               [-1, 72, 20]           8,020
            Linear-4              [-1, 72, 400]           8,400
            Linear-5             [-1, 72, 9216]       3,695,616
================================================================
```

- Complex VAE  

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Linear-1            [-1, 72, 10000]      92,170,000
            Linear-2             [-1, 72, 4096]      40,964,096
            Linear-3             [-1, 72, 2048]       8,390,656
            Linear-4              [-1, 72, 512]       1,049,088
            Linear-5               [-1, 72, 64]          32,832
            Linear-6               [-1, 72, 64]          32,832
            Linear-7              [-1, 72, 512]          33,280
            Linear-8             [-1, 72, 2048]       1,050,624
            Linear-9             [-1, 72, 4096]       8,392,704
           Linear-10            [-1, 72, 10000]      40,970,000
           Linear-11             [-1, 72, 9216]      92,169,216
================================================================
```

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
  <img src="/README/ImageSource0.jpg" alt="Description" width="100" height="100" border="0" />
  <img src="/README/ImageSource1.jpg" alt="Description" width="100" README="100" border="0" />
  <img src="/README/ImageSource2.jpg" alt="Description" width="100" README="100" border="0" />
  <img src="/README/ImageSource3.jpg" alt="Description" width="100" height="100" border="0" />
  <img src="/README/ImageSource4.jpg" alt="Description" width="100" README="100" border="0" />
  <img src="/README/ImageSource5.jpg" alt="Description" width="100" height="100" border="0" />
  <img src="/README/ImageSource6.jpg" alt="Description" width="100" height="100" border="0" />
  <img src="/README/ImageSource7.jpg" alt="Description" width="100" height="100" border="0" />
</p>
<p align="center">
  Figure 2: Image Source 
</p>

<p align="center">
  <img src="/README/image_test0.png" alt="Description" width="100" height="100" border="0" />
  <img src="/README/image_test1.png" alt="Description" width="100" README="100" border="0" />
  <img src="/README/image_test2.png" alt="Description" width="100" README="100" border="0" />
  <img src="/README/image_test3.png" alt="Description" width="100" height="100" border="0" />
  <img src="/README/image_test4.png" alt="Description" width="100" README="100" border="0" />
  <img src="/README/image_test5.png" alt="Description" width="100" height="100" border="0" />
  <img src="/README/image_test6.png" alt="Description" width="100" README="100" border="0" />
  <img src="/README/image_test7.png" alt="Description" width="100" height="100" border="0" />
</p>
<p align="center">
  Figure 3: Simple VAE Testing 
</p>

<p align="center">
  <img src="/README/image_test0d.png" alt="Description" width="100" height="100" border="0" />
  <img src="/README/image_test1d.png" alt="Description" width="100" README="100" border="0" />
  <img src="/README/image_test2d.png" alt="Description" width="100" README="100" border="0" />
  <img src="/README/image_test3d.png" alt="Description" width="100" height="100" border="0" />
  <img src="/README/image_test4d.png" alt="Description" width="100" README="100" border="0" />
  <img src="/README/image_test5d.png" alt="Description" width="100" height="100" border="0" />
  <img src="/README/image_test6d.png" alt="Description" width="100" README="100" border="0" />
  <img src="/README/image_test7d.png" alt="Description" width="100" height="100" border="0" />
</p>
<p align="center">
  Figure 4: Complex VAE Testing 
</p>

## Input to Output
- State (Input)  
  - **Simple VAE is sufficient for feature extraction**  
    - **Use pre-trained Simple VAE model**  
      - **Only Encoder part**  
```
    config.state_dim = 20 # Encoder Feature # 64
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
  Figure 5: Reinforcement Learning PPO on Racing Car
</p>

## Reference
[Manually Driving CarRacing-v0](https://cdancette.fr/2018/04/09/self-driving-CNN)  
https://github.com/stemsgrpy/Control-Task-for-Box2D-Simulator-by-using-PPO    