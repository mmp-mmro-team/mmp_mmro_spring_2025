from torchvision import transforms
from torchvision.transforms import functional as torchvision_functional
from typing import Callable, Optional
import vizdoom
import gymnasium as gym
from IPython.display import display, Image as IPythonImage
import torch
import multiprocessing
import numpy as np
import time
from io import BytesIO
from PIL import Image

def create_and_show_gif(image_list):
    pil_images = []
    for img_array in image_list:
        if img_array.dtype != np.uint8:
            img_array = (img_array * 255).astype(np.uint8)  # Scale float images to uint8 if needed
        pil_images.append(Image.fromarray(img_array))

    gif_buffer = BytesIO()
    pil_images[0].save(
        gif_buffer,
        format="GIF",
        save_all=True,
        append_images=pil_images[1:],
        duration=0,
        loop=0
    )

    gif_buffer.seek(0)
    display(IPythonImage(data=gif_buffer.read(), format='gif'))


class DoomGame:
    def __init__(self, config_path: str = "/usr/local/lib/python3.11/dist-packages/vizdoom/scenarios/health_gathering.cfg", window_visible: bool = False, timeout_seconds=1.0):
        self.timeout_seconds = timeout_seconds
        self.window_visible = window_visible
        self.parent_conn, self.child_conn = multiprocessing.Pipe()
        self.stop_event = multiprocessing.Event()

        # Start doom process
        self.doom_process = multiprocessing.Process(
            target=self._run_doom,
            args=(self.child_conn, self.stop_event, config_path)
        )
        self.doom_process.start()

        # Get initial state
        self.parent_conn.send(("reset", None))
        self.state = self.parent_conn.recv()

    def _run_doom(self, conn, stop_event, config_file):
        game = vizdoom.DoomGame()
        game.load_config(config_file)
        game.set_window_visible(self.window_visible)
        game.init()

        try:
            while not stop_event.is_set():
                # Wait for command from main process
                if not conn.poll(timeout=0.1):  # Check pipe every 0.1 seconds
                    continue

                cmd, action = conn.recv()

                if cmd == "step":
                    reward = game.make_action(action)
                    done = game.is_episode_finished()

                    if not done:
                        state = game.get_state()
                    else:
                        state = None
                        game.new_episode()

                    conn.send((state, reward, done))

                elif cmd == "reset":
                    game.new_episode()
                    state = game.get_state()
                    conn.send(state)

                elif cmd == "close":
                    break

        finally:
            game.close()
            conn.close()

    def get_state(self) -> Optional[vizdoom.GameState]:
        """Return the current state."""
        return self.state

    def make_action(self, action):
        """Execute an action and return (new_state, reward, done)."""
        self.parent_conn.send(("step", action))

        if not self.parent_conn.poll(timeout=self.timeout_seconds):
            raise TimeoutError("Doom process not responding")

        self.state, reward, done = self.parent_conn.recv()
        return self.state, reward, done

    def reset(self):
        """Reset the environment and return initial state."""
        self.parent_conn.send(("reset", None))

        if not self.parent_conn.poll(timeout=self.timeout_seconds):
            raise TimeoutError("Doom process not responding")

        self.state = self.parent_conn.recv()
        return self.state

    def close(self):
        """Clean up resources."""
        self.parent_conn.send(("close", None))
        self.stop_event.set()

        # Give doom process time to clean up
        self.doom_process.join(timeout=5)

        # Force terminate if still running
        if self.doom_process.is_alive():
            self.doom_process.terminate()
            self.doom_process.join()

        self.parent_conn.close()



class Doom(gym.Env):
    def __init__(
            self,
            config_path: str = "/usr/local/lib/python3.11/dist-packages/vizdoom/scenarios/health_gathering.cfg",
            delay_milliseconds: int = 0,
            window_visible: bool = False,
            show_replay: bool = False,
            transforms = None
    ):
        self.game = DoomGame(config_path=config_path, window_visible=window_visible)
        self.delay_milliseconds = delay_milliseconds
        self.show_replay = show_replay

        self.transforms = transforms
        self.images = []

    def _get_state(self) -> torch.FloatTensor:
        state: Optional[vizdoom.GameState] = self.game.get_state()
        if state is None:
            raise ValueError("Game is not initialized")

        result = torch.FloatTensor(state.screen_buffer).unsqueeze(0) / 255
        if self.transforms is not None:
            result = self.transforms(result)
        return result

    def _save_image(self):
        state: Optional[vizdoom.GameState] = self.game.get_state()
        if state is None:
            raise ValueError("Game is not initialized")

        self.images.append(np.transpose(state.screen_buffer, (1, 2, 0)))

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        if self.show_replay and len(self.images) > 0:
            create_and_show_gif(self.images)

        self.game.reset()

        self.images = []
        self._save_image()

        state = self._get_state()
        return state, None

    def step(self, action: int):
        action = [
            [1, 0, 0],
            [0, 1, 0,],
            [0, 0, 1]
        ][action]

        state, reward, terminated = self.game.make_action(action)
        if not terminated:
            state = self._get_state()
            self._save_image()
        else:
            self.game.reset()

        if self.delay_milliseconds > 0:
            time.sleep(self.delay_milliseconds / 1000)

        return state, reward, terminated, None, None

def init_game(window_visible: bool = False, preprocess_transforms=None, config_path="/usr/local/lib/python3.11/dist-packages/vizdoom/scenarios/health_gathering.cfg") -> Doom:
    return Doom(
        window_visible=window_visible,
        transforms=preprocess_transforms,
        config_path = config_path,
    )
