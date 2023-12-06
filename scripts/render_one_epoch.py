"""
Temporary demo, will delete later
"""
import pdb
from diffuser.datasets.d4rl import suppress_output
with suppress_output():
    ## d4rl prints out a variety of warnings
    import d4rl
import gym
from d4rl.kitchen.adept_envs.franka.kitchen_multitask_v0 import KitchenTaskRelaxV1
import glfw


from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np


def save_imgs_as_gif(frames, path='./', filename='kitchen.gif'):
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    def animate(i):
        patch.set_data(frames[i])
    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)


def run_one_epoch(env, dataset):
    # env = KitchenTaskRelaxV1()
    renderer = env.sim_robot.renderer
    env.reset()
    for idx, sample in enumerate(dataset):
        trajectories, conditions, returns, text_features = sample
        this_done = False
        print(trajectories.shape)
        sample_steps = trajectories.shape[0]
        action_dim = 9
        action, observations = trajectories[:, :action_dim], trajectories[:, action_dim:]
        step = 0
        renders = []
        for step in range(sample_steps):
            this_obs, this_reward, this_done, _ = env.step(action[step])
            # ren = env.sim.render(480, 360)
            pdb.set_trace()
            ren = renderer.render_offscreen(512, 384, camera_id=-1)
            renders.append(ren)

            if this_done:
                
                break
            step += 1
            # print(this_obs - )
        save_imgs_as_gif(renders)
        pdb.set_trace()
        break
        

if __name__ == '__main__':
    pass
    # main()