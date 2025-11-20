import gymnasium as gym
import ale_py

import numpy as np
import matplotlib.pyplot as plt

import cv2
import imageio

class CustomEnv(gym.ObservationWrapper):
    """
    Recorta las áreas inútiles del frame de Atari (pixeles 
    en negro donde no llega ni la nave ni los enemigos)
    """
    def __init__(self, env, top=10, bottom=15, left=3, right=15, target_size=(84, 84)):
        super().__init__(env)
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right
        self.target_size = target_size  # (height, width)

        # Nuevo shape de la observación
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.target_size[0], self.target_size[1], 1),
            dtype=np.uint8
        )

    def observation(self, obs):
        cropped = obs[
            self.top:obs.shape[0]-self.bottom,       # recorte vertical
            self.left:obs.shape[1]-self.right,      # recorte horizontal
            :
        ]
        resized = cv2.resize(
            cropped,
            (self.target_size[1], self.target_size[0]),  # width, height
            interpolation=cv2.INTER_AREA
        )

        # Convertir a escala de grises
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        # Añadir dimensión de canal
        gray = np.expand_dims(gray, axis=-1)
        
        return gray

    """
    Hace un video de un episodio con acciones aleatorias
    """
    def record_video(self, filename="space_invaders_cropped.mp4"):
        obs, info = self.reset()
        done = False
        frames = []
    
        while not done:
            # Se guarda solo la imagen 2D (H, W) para video
            frames.append(obs[:, :, 0])
            action = self.action_space.sample()
            obs, reward, terminated, truncated, info = self.step(action)
            done = terminated or truncated

    
        # Guardar video en escala de grises directamente
        imageio.mimsave(filename, frames, fps=30, format="FFMPEG", macro_block_size=None)
        print(f"Video guardado en {filename}")

    """
    Divide la observación en tiras horizontales y/o verticales.
    Devuelve una lista con todas las tiras.
    """
    def crop_strips(self, obs, num_horizontal=0, num_vertical=16):
        
        h, w, c = obs.shape
        strips = []

        # Cortes horizontales
        if num_horizontal > 1:
            h_strip = h // num_horizontal
            for i in range(num_horizontal):
                strips.append(obs[i*h_strip : (i+1)*h_strip, :, :])
        else:
            strips.append(obs)  # si no hay cortes horizontales, agregamos la imagen completa

        # Cortes verticales sobre cada tira horizontal
        if num_vertical > 1:
            vertical_strips = []
            w_strip = w // num_vertical
            for strip in strips:
                for j in range(num_vertical):
                    vertical_strips.append(strip[:, j*w_strip:(j+1)*w_strip, :])
            strips = vertical_strips

        return strips
    
class RewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        lives = self.env.unwrapped.ale.lives()
        if reward > 0:
            match lives:
                case 3:
                    return reward
                case 2:
                    return reward * 0.9
                case 1:
                    return reward * 0.8
        return reward;

class LifePenaltyWrapper(gym.Wrapper):
    def __init__(self, env, penalty=-20):
        super().__init__(env)
        self.penalty = penalty
        self.last_lives = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_lives = self.env.unwrapped.ale.lives()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        current_lives = self.env.unwrapped.ale.lives()

        # Penalizar si pierde una vida
        if current_lives < self.last_lives:
            reward += self.penalty

        self.last_lives = current_lives

        return obs, reward, terminated, truncated, info


def create_space_invaders_env(render_mode="rgb_array"):
    """
    Crea y configura el ambiente de Space Invaders con los wrappers personalizados
    """
    
    # Registrar ambientes de ALE
    gym.register_envs(ale_py)
    
    # Crear ambiente base
    env = gym.make(
        "ALE/SpaceInvaders-v5",
        frameskip=3,
        render_mode=render_mode
    )

    # Aplicar wrapper de penalización
    env = LifePenaltyWrapper(env, penalty=-20)

    # Aplicar wrapper de recompensas
    env = RewardWrapper(env)
    
    # Aplicar wrapper de recorte y redimensionamiento
    env = CustomEnv(
        env,
        top=10,
        bottom=15,
        left=3,
        right=15,
        target_size=(84, 84)
    )
    
    return env