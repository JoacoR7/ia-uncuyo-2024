import csv
import time
from stable_baselines3.common.callbacks import BaseCallback

class EpisodeCSVLogger(BaseCallback):
    def __init__(self, filename="episode_data.csv", verbose=0):
        super().__init__(verbose)
        self.filename = filename
        self.header_written = False
        self.episode_start_time = None
        self.episode_count = 0
        self.last_metrics = {}

    def _on_training_start(self):
        self.episode_start_time = time.time()

    def _on_rollout_end(self):
        """Capturar métricas al final del rollout (cuando se calculan las pérdidas)"""
        if hasattr(self.model, 'logger') and self.model.logger is not None:
            self.last_metrics = dict(self.model.logger.name_to_value)

    def _on_step(self):
        infos = self.locals["infos"]

        for info in infos:
            if "episode" in info:
                ep_reward = info["episode"]["r"]
                ep_length = info["episode"]["l"]
                ep_time = time.time() - self.episode_start_time
                self.episode_count += 1

                # Obtener pérdidas guardadas
                policy_loss = self.last_metrics.get("train/policy_loss", None)
                value_loss = self.last_metrics.get("train/value_loss", None)
                loss = self.last_metrics.get("train/loss", None)

                if not self.header_written:
                    with open(self.filename, "w", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            "episode_number",
                            "episode_reward",
                            "episode_length",
                            "episode_time_seconds",
                            "policy_loss",
                            "value_loss",
                            "total_loss"
                        ])
                    self.header_written = True

                with open(self.filename, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        self.episode_count,
                        ep_reward,
                        ep_length,
                        ep_time,
                        policy_loss if policy_loss is not None else "",
                        value_loss if value_loss is not None else "",
                        loss if loss is not None else ""
                    ])

                self.episode_start_time = time.time()

        return True