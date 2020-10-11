import gym
import numpy as np
import imageio
import os
from pyglet.window import key
import scipy
import scipy.misc
import cv2

BATCH = 1 # 30
FRAME = 7
EXTEND_IMG = 6
IMAGE_NUM = 2088*BATCH
RECORD_START = 18
RECORD_END = IMAGE_NUM*FRAME/EXTEND_IMG + RECORD_START

env = gym.make('CarRacing-v0')
env.reset()

if __name__=='__main__':
    fr = 0
    ffr = 0
    a = np.array([0.0, 0.0, 0.0])
    
    if not os.path.isdir('RacingCarDataset'):
        os.makedirs('RacingCarDataset', exist_ok=True)

    def key_press(k, mod):
        if k==key.LEFT:  a[0] = -1.0
        if k==key.RIGHT: a[0] = +1.0
        if k==key.UP:    a[1] = +1.0
        if k==key.DOWN:  a[2] = +0.8
    def key_release(k, mod):
        if k==key.LEFT  and a[0]==-1.0: a[0] = 0
        if k==key.RIGHT and a[0]==+1.0: a[0] = 0
        if k==key.UP:    a[1] = 0
        if k==key.DOWN:  a[2] = 0

    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release
    state = env.reset()

    while True:
        s, r, done, info = env.step(a)
        env.render()

        if fr > RECORD_END:
            break
        elif fr > RECORD_START and fr % FRAME == 0:
            ffr += 1
            s = cv2.cvtColor(s, cv2.COLOR_BGR2GRAY) # RGB to Gray
            for _ in range(EXTEND_IMG):
                scipy.misc.imsave('./RacingCarDataset/history'+str(ffr)+str(_)+'.jpg', s)
        fr += 1

        if done:
            env.reset()
    env.close()
