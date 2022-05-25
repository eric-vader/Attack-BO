# Suppress pygame output
# https://stackoverflow.com/questions/51464455/why-when-import-pygame-it-prints-the-version-and-welcome-message-how-delete-it
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

from .generate_simudata3 import robot_push_3d
from .generate_simudata4 import robot_push_4d