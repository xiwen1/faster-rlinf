# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import multiprocessing
import warnings
from multiprocessing import connection
from typing import Any, Callable, Optional, Union

import gym
import numpy as np

from rlinf.envs.sim.libero.utils import (
    get_libero_image,
    get_libero_type,
    get_libero_wrist_image,
    quat2axisangle,
)
from rlinf.envs.venv import (
    BaseVectorEnv,
    CloudpickleWrapper,
    EnvWorker,
    ShArray,
    SubprocEnvWorker,
    SubprocVectorEnv,
    _setup_buf,
)

# ---------------------------------------------------------------------------
# Dynamic Module Import Logic for Libero Pro / Plus
# ---------------------------------------------------------------------------
libero_type = get_libero_type()

if libero_type == "pro":
    try:
        from liberopro.liberopro.envs import OffScreenRenderEnv
    except ImportError as e:
        print(
            f"[Venv] Warning: LIBERO_TYPE=pro but import failed ({e}). Falling back to standard libero..."
        )
        from libero.libero.envs import OffScreenRenderEnv

elif libero_type == "plus":
    try:
        from liberoplus.liberoplus.envs import OffScreenRenderEnv
    except ImportError as e:
        print(
            f"[Venv] Warning: LIBERO_TYPE=plus but import failed ({e}). Falling back to standard libero..."
        )
        from libero.libero.envs import OffScreenRenderEnv

else:
    try:
        from libero.libero.envs import OffScreenRenderEnv
    except ImportError:
        try:
            from liberopro.liberopro.envs import OffScreenRenderEnv
        except ImportError:
            try:
                from liberoplus.liberoplus.envs import OffScreenRenderEnv
            except ImportError:
                raise ImportError(
                    "Could not import OffScreenRenderEnv from libero, liberopro, or liberoplus."
                )


gym_old_venv_step_type = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
gym_new_venv_step_type = tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]
warnings.simplefilter("once", DeprecationWarning)

LIBERO_CAMERA_OBS_NAMES = ("agentview_image", "robot0_eye_in_hand_image")


def _set_camera_rendering(env, enabled: bool) -> None:
    """Enable or disable LIBERO camera observables without resetting the env."""
    rob = getattr(env, "env", env)
    while hasattr(rob, "env"):
        rob = rob.env
    observables = getattr(rob, "_observables", None)
    if observables is None:
        return
    for name in LIBERO_CAMERA_OBS_NAMES:
        if name in observables:
            observables[name]._enabled = enabled


def extract_libero_obs(obs: dict) -> dict:
    """The per-env part of ``LiberoEnv._extract_image_and_state``, run in the env worker.

    faster-rlinf: the fast chunk/reset paths extract the three arrays the policy needs inside
    the worker, so only ~400 KB instead of the full robosuite observable dict crosses the pipe.
    Same functions on the same arrays as the parent-side extraction, so the values are identical;
    ``LiberoEnv._extract_image_and_state`` passes an already-extracted dict through unchanged.
    """
    return {
        "full_image": np.ascontiguousarray(get_libero_image(obs)),
        "wrist_image": np.ascontiguousarray(get_libero_wrist_image(obs)),
        "state": np.concatenate(
            [
                obs["robot0_eef_pos"],
                quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            ]
        ),
    }


def _run_steps(env, actions, render_mask, extract):
    """Step ``env`` through ``actions`` with camera rendering only where ``render_mask`` is True.

    Returns one ``(obs, reward, done, info)`` per step; ``obs`` is None for steps that were not
    rendered. Rendering is switched exactly like ``LiberoEnv.chunk_step`` does it and is always
    switched back on at the end.
    """
    out = []
    rendering = True
    try:
        for action, render in zip(actions, render_mask):
            render = bool(render)
            if render != rendering:
                _set_camera_rendering(env, render)
                rendering = render
            obs, reward, done, info = env.step(action)
            if not render:
                obs = None
            elif extract:
                obs = extract_libero_obs(obs)
            out.append((obs, reward, done, info))
    finally:
        if not rendering:
            _set_camera_rendering(env, True)
    return out


def _worker(
    parent: connection.Connection,
    p: connection.Connection,
    env_fn_wrapper: CloudpickleWrapper,
    obs_bufs: Optional[Union[dict, tuple, ShArray]] = None,
) -> None:
    def _encode_obs(
        obs: Union[dict, tuple, np.ndarray], buffer: Union[dict, tuple, ShArray]
    ) -> None:
        if isinstance(obs, np.ndarray) and isinstance(buffer, ShArray):
            buffer.save(obs)
        elif isinstance(obs, tuple) and isinstance(buffer, tuple):
            for o, b in zip(obs, buffer):
                _encode_obs(o, b)
        elif isinstance(obs, dict) and isinstance(buffer, dict):
            for k in obs.keys():
                _encode_obs(obs[k], buffer[k])
        return None

    parent.close()
    env = env_fn_wrapper.data()
    try:
        while True:
            try:
                cmd, data = p.recv()
            except EOFError:  # the pipe has been closed
                p.close()
                break
            if cmd == "step":
                env_return = env.step(data)
                if obs_bufs is not None:
                    _encode_obs(env_return[0], obs_bufs)
                    env_return = (None, *env_return[1:])
                p.send(env_return)
            elif cmd == "reset":
                retval = env.reset(**data)
                reset_returns_info = (
                    isinstance(retval, (tuple, list))
                    and len(retval) == 2
                    and isinstance(retval[1], dict)
                )
                if reset_returns_info:
                    obs, info = retval
                else:
                    obs = retval
                if obs_bufs is not None:
                    _encode_obs(obs, obs_bufs)
                    obs = None
                if reset_returns_info:
                    p.send((obs, info))
                else:
                    p.send(obs)
            elif cmd == "close":
                p.send(env.close())
                p.close()
                break
            elif cmd == "render":
                p.send(env.render(**data) if hasattr(env, "render") else None)
            elif cmd == "seed":
                if hasattr(env, "seed"):
                    p.send(env.seed(data))
                else:
                    env.reset(seed=data)
                    p.send(None)
            elif cmd == "getattr":
                p.send(getattr(env, data) if hasattr(env, data) else None)
            elif cmd == "setattr":
                setattr(env.unwrapped, data["key"], data["value"])
            elif cmd == "check_success":
                p.send(env.check_success())
            elif cmd == "get_segmentation_of_interest":
                p.send(env.get_segmentation_of_interest(data))
            elif cmd == "get_sim_state":
                p.send(env.get_sim_state())
            elif cmd == "set_init_state":
                obs = env.set_init_state(data)
                p.send(obs)
            elif cmd == "reconfigure":
                env.close()
                seed = data.pop("seed")
                env = OffScreenRenderEnv(**data)
                env.seed(seed)
                p.send(None)
            elif cmd == "get_camera_meta":
                # Compute camera intrinsics/extrinsics and depth near/far
                # from the robosuite sim, which is only reachable inside the
                # worker subprocess.  Returns picklable lists/floats so the
                # driver can back-project pixels to world without GT poses.
                from robosuite.utils import camera_utils

                rob = getattr(env, "env", env)
                while hasattr(rob, "env"):
                    rob = rob.env
                sim = rob.sim
                cam = data.get("camera_name", "agentview")
                h = int(data.get("height", 256))
                w = int(data.get("width", 256))
                K = camera_utils.get_camera_intrinsic_matrix(sim, cam, h, w)
                E = camera_utils.get_camera_extrinsic_matrix(sim, cam)
                extent = float(sim.model.stat.extent)
                near = float(sim.model.vis.map.znear) * extent
                far = float(sim.model.vis.map.zfar) * extent
                p.send(
                    {
                        "camera_name": cam,
                        "height": h,
                        "width": w,
                        "intrinsic_K": K.tolist(),
                        "extrinsic_cam2world": E.tolist(),
                        "depth_near": near,
                        "depth_far": far,
                    }
                )
            elif cmd == "render_camera":
                rob = getattr(env, "env", env)
                while hasattr(rob, "env"):
                    rob = rob.env
                sim = rob.sim
                cam = data.get("camera_name", "agentview")
                h = int(data.get("height", 1024))
                w = int(data.get("width", 1024))
                depth = bool(data.get("depth", False))
                p.send(
                    sim.render(
                        width=w,
                        height=h,
                        camera_name=cam,
                        depth=depth,
                    )
                )
            elif cmd == "set_camera_rendering":
                _set_camera_rendering(env, data)
                p.send(None)
            elif cmd == "chunk_step_fast":
                # faster-rlinf: a whole action chunk in one round trip. The env sees exactly
                # the calls of LiberoEnv.chunk_step -> step (same actions, same order, same
                # rendering switches), so its state evolves bit for bit as upstream.
                p.send(
                    _run_steps(env, data["actions"], data["render"], data["extract"])
                )
            elif cmd == "reset_episode_fast":
                # faster-rlinf: reconfigure + seed + reset + set_init_state + settle steps in
                # one round trip, all workers in parallel. Same env calls in the same order as
                # LiberoEnv._reconfigure followed by the num_steps_wait loop; the settle steps
                # render only the last frame (the only one that is returned).
                if data["reconfigure"] is not None:
                    params = dict(data["reconfigure"])
                    env.close()
                    seed = params.pop("seed")
                    env = OffScreenRenderEnv(**params)
                    env.seed(seed)
                env.seed(data["seed"])
                env.reset()
                if data["init_state"] is not None:
                    env.set_init_state(data["init_state"])
                n = len(data["wait_actions"])
                mask = [data["render_all"] or k == n - 1 for k in range(n)]
                p.send(_run_steps(env, data["wait_actions"], mask, data["extract"])[-1])
            else:
                p.close()
                raise NotImplementedError
    except KeyboardInterrupt:
        p.close()


class ReconfigureSubprocEnvWorker(SubprocEnvWorker):
    def __init__(self, env_fn: Callable[[], gym.Env], share_memory: bool = False):
        ctx = multiprocessing.get_context("spawn")
        self.parent_remote, self.child_remote = ctx.Pipe()
        self.share_memory = share_memory
        self.buffer: Optional[Union[dict, tuple, ShArray]] = None
        if self.share_memory:
            dummy = env_fn()
            obs_space = dummy.observation_space
            dummy.close()
            del dummy
            self.buffer = _setup_buf(obs_space)
        args = (
            self.parent_remote,
            self.child_remote,
            CloudpickleWrapper(env_fn),
            self.buffer,
        )
        self.process = ctx.Process(target=_worker, args=args, daemon=True)
        self.process.start()
        self.child_remote.close()
        EnvWorker.__init__(self, env_fn)

    def reconfigure_env_fn(self, env_fn_param):
        self.parent_remote.send(["reconfigure", env_fn_param])
        return self.parent_remote.recv()


class ReconfigureSubprocEnv(SubprocVectorEnv):
    def __init__(self, env_fns: list[Callable[[], gym.Env]], **kwargs: Any) -> None:
        def worker_fn(fn: Callable[[], gym.Env]) -> ReconfigureSubprocEnvWorker:
            return ReconfigureSubprocEnvWorker(fn, share_memory=False)

        BaseVectorEnv.__init__(self, env_fns, worker_fn, **kwargs)

    def reconfigure_env_fns(self, env_fns, id=None):
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        if self.is_async:
            self._assert_id(id)

        for j, i in enumerate(id):
            self.workers[i].reconfigure_env_fn(env_fns[j])

    def chunk_step_fast(self, actions, render_mask, extract=True):
        """faster-rlinf: step every env through its whole action chunk in one round trip.

        ``actions``: [num_envs, chunk, action_dim]. Returns, per control step, a tuple
        ``(obs_stack, reward_stack, done_stack, info_stack)`` built exactly like
        ``BaseVectorEnv.step`` builds it (``env_id`` added to each info, ``np.stack``), with
        ``obs_stack`` None for steps that were not rendered.
        """
        self._assert_is_not_closed()
        assert not self.is_async
        assert len(actions) == self.env_num
        for j, w in enumerate(self.workers):
            w.parent_remote.send(
                [
                    "chunk_step_fast",
                    {"actions": actions[j], "render": render_mask, "extract": extract},
                ]
            )
        per_env = [w.parent_remote.recv() for w in self.workers]
        steps = []
        for i in range(len(render_mask)):
            result = []
            for j in range(self.env_num):
                obs, reward, done, info = per_env[j][i]
                info["env_id"] = j
                result.append((obs, reward, done, info))
            obs_list, *others = tuple(zip(*result))
            if render_mask[i]:
                try:
                    obs_stack = np.stack(obs_list)
                except ValueError:
                    obs_stack = np.array(obs_list, dtype=object)
            else:
                obs_stack = None
            steps.append((obs_stack, *map(np.stack, others)))
        return steps

    def reset_episodes_fast(
        self, env_idx, reconfigure_params, seeds, init_states, wait_actions, render_all,
        extract=True,
    ):
        """faster-rlinf: reconfigure/seed/reset/set_init_state/settle ``env_idx`` in parallel.

        ``reconfigure_params``: {env_id: env_fn params incl. "seed"} for the envs to rebuild;
        ``seeds``: one seed per worker (all workers are re-seeded, as ``BaseVectorEnv.seed``
        does); ``init_states``: one per ``env_idx`` entry or None; ``wait_actions``:
        [len(env_idx), num_steps_wait, action_dim]. Returns the last settle-step obs per env.
        """
        self.reset_episodes_fast_send(
            env_idx, reconfigure_params, seeds, init_states, wait_actions, render_all,
            extract,
        )
        return self.reset_episodes_fast_recv(env_idx)

    def reset_episodes_fast_recv(self, env_idx):
        return [self.workers[int(i)].parent_remote.recv() for i in env_idx]

    def reset_episodes_fast_send(
        self, env_idx, reconfigure_params, seeds, init_states, wait_actions, render_all,
        extract=True,
    ):
        """Send half of ``reset_episodes_fast`` (results collected by ``*_recv``)."""
        self._assert_is_not_closed()
        assert not self.is_async
        idx = [int(i) for i in env_idx]
        in_idx = set(idx)
        for i, w in enumerate(self.workers):
            if i not in in_idx:
                w.seed(seeds[i])
        for j, i in enumerate(idx):
            w = self.workers[i]
            EnvWorker.seed(w, seeds[i])
            w.parent_remote.send(
                [
                    "reset_episode_fast",
                    {
                        "reconfigure": reconfigure_params.get(i),
                        "seed": seeds[i],
                        "init_state": None if init_states is None else init_states[j],
                        "wait_actions": wait_actions[j],
                        "render_all": render_all,
                        "extract": extract,
                    },
                ]
            )

    def set_camera_rendering(self, enabled: bool, id=None):
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        if self.is_async:
            self._assert_id(id)
        for i in id:
            self.workers[i].parent_remote.send(["set_camera_rendering", enabled])
        for i in id:
            self.workers[i].parent_remote.recv()
