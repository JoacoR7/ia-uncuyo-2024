import random

class ReflexiveAgent:
    def __init__(self, env):
        self.env = env  
        self.actions = ["up", "down", "left", "right", "clean", "idle"]

    def up(self):
        self.env.accept_action("Up")

    def down(self):
        self.env.accept_action("Down")

    def left(self):
        self.env.accept_action("Left")

    def right(self):
        self.env.accept_action("Right")

    def clean(self):
        self.env.accept_action("Clean")

    def idle(self):
        self.env.accept_action("Idle")

    def perspective(self):
        return self.env.is_dirty()

    def think(self):
        if self.perspective() == 1:  #Si la celda está sucia, limpia
            self.clean()
        else:  #Si está limpia, elige una acción aleatoria para moverse
            action = random.choice(self.actions[:-2])  # Excluye "clean" e "idle"
            if action == "up":
                self.up()
            elif action == "down":
                self.down()
            elif action == "left":
                self.left()
            elif action == "right":
                self.right()