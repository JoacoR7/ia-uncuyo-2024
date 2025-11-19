import gymnasium as gym

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
            shape=(self.target_size[0], self.target_size[1], 3),
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
    
