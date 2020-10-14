import numpy as np
from torchvision.utils import save_image
from torchvision import transforms
import PIL.Image as Image
import cv2

img_transform = transforms.Compose([
    transforms.ToTensor()
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def to_img(x):
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 96, 96)
    return x

class Tester(object):

    def __init__(self, agent, model, env, model_path, num_episodes=5, max_ep_steps=400, test_ep_steps=2000):
        self.num_episodes = num_episodes
        self.max_ep_steps = max_ep_steps
        self.test_ep_steps = test_ep_steps
        self.agent = agent
        self.env = env

        self.model = model

        # VAE test 
        for _ in range(8):
            img = Image.open('./README/ImageSource'+str(_)+'.jpg')
            img = img_transform(img)
            img = img.view(img.size(0), -1)

            if torch.cuda.is_available():
                img = img.cuda()
            recon_batch, z, mu, logvar = self.model(img)
            save = to_img(recon_batch.cpu().data)
            save_image(save, './VAE_training/image_{}.png'.format('ImageVAE'+str(_)))

        self.agent.is_training = False
        self.agent.load_weights(model_path)
        # self.agent.load_checkpoint(model_path)
        self.policy = lambda x: agent.act(x)

    def test(self, debug=False, visualize=True):
        avg_reward = 0
        for episode in range(self.num_episodes):
            s0 = self.env.reset()

            s0 = cv2.cvtColor(s0, cv2.COLOR_BGR2GRAY)
            s0 = img_transform(s0)
            s0 = s0.view(s0.size(0), -1)
            s0 = s0.cuda()
            recon_batch, z, mu, logvar = self.model(s0)
            s0 = z.squeeze(0).cpu().detach().numpy()
            
            # history = np.concatenate((s0, s0, s0, s0))

            episode_steps = 0
            episode_reward = 0.

            done = False
            while not done:
                if visualize:
                    self.env.render()

                action = self.policy(s0)
                s0, reward, done, info = self.env.step(action)
                reward = float(reward)

                s0 = cv2.cvtColor(s0, cv2.COLOR_BGR2GRAY)
                s0 = img_transform(s0)
                s0 = s0.view(s0.size(0), -1)
                s0 = s0.cuda()
                recon_batch, z, mu, logvar = self.model(s0)
                s0 = z.squeeze(0).cpu().detach().numpy()
                
                # history = np.append(s0, history[:60])

                episode_reward += reward
                episode_steps += 1

                if episode_steps + 1 > self.test_ep_steps:
                    done = True

            if debug:
                print('[Test] episode: %3d, episode_reward: %5f' % (episode, episode_reward))

            avg_reward += episode_reward
        avg_reward /= self.num_episodes
        print("avg reward: %5f" % (avg_reward))




