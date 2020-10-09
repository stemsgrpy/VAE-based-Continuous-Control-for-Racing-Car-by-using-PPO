import math
import numpy as np
from config import Config
from core.logger import TensorBoardLogger
from core.util import get_output_folder

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

class Trainer:
    def __init__(self, agent, model, env, config: Config):
        self.agent = agent
        self.env = env
        self.config = config

        self.model = model
        self.SaveImage = False

        '''
        # VAE test 
        img = Image.open('README/ImageSource.jpg')
        img = img_transform(img)
        img = img.view(img.size(0), -1)

        img = img.cuda()
        recon_batch, z, mu, logvar = self.model(img)
        save = to_img(recon_batch.cpu().data)
        save_image(save, './image_{}.jpg'.format('ImageVAE'))
        '''
        
        self.outputdir = get_output_folder(self.config.output, self.config.env)
        self.agent.save_config(self.outputdir)
        self.board_logger = TensorBoardLogger(self.outputdir)
        print(self.env.action_space.low, self.env.action_space.high)

    def train(self, pre_fr=0):
        all_rewards = []
        tmp_reward = 0
        episode_reward = 0
        ep_num = 0
        is_win = False

        state = self.env.reset()

        state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        state = img_transform(state)
        state = state.view(state.size(0), -1)
        state = state.cuda()
        recon_batch, z, mu, logvar = self.model(state)
        state = z.squeeze(0).cpu().detach().numpy()

        if self.SaveImage:
            save = to_img(recon_batch.cpu().data)
            save_image(save, './history/image_{}.png'.format('_'))

        # history = np.concatenate((state, state, state, state))

        for fr in range(pre_fr + 1, self.config.frames + 1):
            self.env.render()

            #JJ
            action = self.agent.act(state)
            action = action.clip(self.env.action_space.low, self.env.action_space.high)

            next_state, reward, done, _ = self.env.step(action)
            reward = float(reward)

            next_state = cv2.cvtColor(next_state, cv2.COLOR_BGR2GRAY)
            next_state = img_transform(next_state)
            next_state = next_state.view(next_state.size(0), -1)
            next_state = next_state.cuda()
            recon_batch, z, mu, logvar = self.model(next_state)
            next_state = z.squeeze(0).cpu().detach().numpy()

            if self.SaveImage:
                save = to_img(recon_batch.cpu().data)
                save_image(save, './history/image_{}.png'.format(fr))   
                            
            # next_history = np.append(next_state, history[:60])

            self.agent.buffer.rewards.append(reward)
            self.agent.buffer.is_terminals.append(done)

            state = next_state
            # history = next_history
            episode_reward += reward

            if fr % self.config.update_tar_interval == 0:
                self.agent.learning(fr)
                self.agent.buffer.clear_memory()

            if fr % self.config.print_interval == 0:
                print("frames: %5d, reward: %5f, episode: %4d" % (fr, np.mean(all_rewards[-10:]), ep_num))

            if fr % self.config.log_interval == 0:
                self.board_logger.scalar_summary('Reward per episode', ep_num, all_rewards[-1])

            if self.config.checkpoint and fr % self.config.checkpoint_interval == 0:
                self.agent.save_checkpoint(fr, self.outputdir)

            if done:
                state = self.env.reset()

                state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
                state = img_transform(state)
                state = state.view(state.size(0), -1)
                state = state.cuda()
                recon_batch, z, mu, logvar = self.model(state)
                state = z.squeeze(0).cpu().detach().numpy()

                # history = np.concatenate((state, state, state, state))

                all_rewards.append(episode_reward)
                episode_reward = 0
                ep_num += 1
                avg_reward = float(np.mean(all_rewards[-100:]))
                self.board_logger.scalar_summary('Best 100-episodes average reward', ep_num, avg_reward)

                if len(all_rewards) >= 100 and avg_reward >= self.config.win_reward and all_rewards[-1] > self.config.win_reward:
                    is_win = True
                    self.agent.save_model(self.outputdir, 'best')
                    print('Ran %d episodes best 100-episodes average reward is %3f. Solved after %d trials ✔' % (ep_num, avg_reward, ep_num - 100))
                    if self.config.win_break:
                        break
                elif len(all_rewards) >= 100 and avg_reward > tmp_reward:
                    tmp_reward = avg_reward
                    self.agent.save_model(self.outputdir, 'tmp')
                    print('Ran %d episodes tmp 100-episodes average reward is %3f. tmp Solved after %d trials' % (ep_num, avg_reward, ep_num - 100))

        if not is_win:
            print('Did not solve after %d episodes' % ep_num)
            self.agent.save_model(self.outputdir, 'last')
