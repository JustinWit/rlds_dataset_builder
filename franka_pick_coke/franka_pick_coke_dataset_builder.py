from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import pickle as pkl
import math

PI = np.pi
EPS = np.finfo(float).eps * 4.0

# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    "sxyz": (0, 0, 0, 0),
    "sxyx": (0, 0, 1, 0),
    "sxzy": (0, 1, 0, 0),
    "sxzx": (0, 1, 1, 0),
    "syzx": (1, 0, 0, 0),
    "syzy": (1, 0, 1, 0),
    "syxz": (1, 1, 0, 0),
    "syxy": (1, 1, 1, 0),
    "szxy": (2, 0, 0, 0),
    "szxz": (2, 0, 1, 0),
    "szyx": (2, 1, 0, 0),
    "szyz": (2, 1, 1, 0),
    "rzyx": (0, 0, 0, 1),
    "rxyx": (0, 0, 1, 1),
    "ryzx": (0, 1, 0, 1),
    "rxzx": (0, 1, 1, 1),
    "rxzy": (1, 0, 0, 1),
    "ryzy": (1, 0, 1, 1),
    "rzxy": (1, 1, 0, 1),
    "ryxy": (1, 1, 1, 1),
    "ryxz": (2, 0, 0, 1),
    "rzxz": (2, 0, 1, 1),
    "rxyz": (2, 1, 0, 1),
    "rzyz": (2, 1, 1, 1),
}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())


def mat2euler(rmat, axes="sxyz"):  # from /home/ripl/deoxys_control/deoxys/deoxys/utils/transform_utils.py
    """
    Converts given rotation matrix to euler angles in radian.

    Args:
        rmat (np.array): 3x3 rotation matrix
        axes (str): One of 24 axis sequences as string or encoded tuple (see top of this module)

    Returns:
        np.array: (r,p,y) converted euler angles in radian vec3 float
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]

    M = np.array(rmat, dtype=np.float32, copy=False)[:3, :3]
    if repetition:
        sy = math.sqrt(M[i, j] * M[i, j] + M[i, k] * M[i, k])
        if sy > EPS:
            ax = math.atan2(M[i, j], M[i, k])
            ay = math.atan2(sy, M[i, i])
            az = math.atan2(M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(sy, M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i] * M[i, i] + M[j, i] * M[j, i])
        if cy > EPS:
            ax = math.atan2(M[k, j], M[k, k])
            ay = math.atan2(-M[k, i], cy)
            az = math.atan2(M[j, i], M[i, i])
        else:
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(-M[k, i], cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return np.array((ax, ay, az), dtype=np.float32)


class FrankaPickCoke(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Tensor(
                            shape=(360, 640, 3),
                            dtype=np.uint8,
                            doc='Front camera RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                        shape=(6,),
                        dtype=np.float32,
                        doc='Robot action, consists of [eef_pos (x, y, z), '
                            'eef_euler (roll, pitch, yaw)].',
                        ),
                        'gripper_state': tfds.features.Tensor(
                        shape=(1,),
                        dtype=np.int8,
                        doc='gripper_cmd (x).',
                        ),
                        # 'image2': tfds.features.Tensor(
                        #     shape=(360, 640, 3),
                        #     dtype=np.uint8,
                        #     doc='Side camera RGB observation. pkl imdex 1',
                        # ),
                        # 'image3': tfds.features.Tensor(
                        #     shape=(360, 640, 3),
                        #     dtype=np.uint8,
                        #     doc='Top camera RGB observation. pkl index 0',
                        # ),
                        # 'depth_image': tfds.features.Tensor(
                        #     shape=(360, 640),
                        #     dtype=np.uint16,
                        #     doc='Front camera depth observation.',
                        # ),
                        # 'image2': tfds.features.Tensor(
                        #     shape=(360, 640),
                        #     dtype=np.uint16,
                        #     doc='Side camera depth observation.',
                        # ),
                        # 'image3': tfds.features.Tensor(
                        #     shape=(360, 640),
                        #     dtype=np.uint16,
                        #     doc='Top camera depth observation.',
                        # ),
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='Robot action: EEF Delta XYZ (3) + Roll-Pitch-Yaw (3) + Gripper Open/Close (1)',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(1,),
                        dtype=np.float16,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(path='data/demo_coke*.pkl'),
            # 'val': self._generate_examples(path='data/val/episode_*.npy'),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path):
            # load raw data --> this should change for your dataset
            with open(episode_path, 'rb') as dbfile:
                db = pkl.load(dbfile)

            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []
            for i in range(len(db['timestamps'])):
                # compute Kona language embedding
                # language_embedding = self._embed("Pick up the coke can")[0].numpy()

                # Convert demonstration data to 6DOF actions
                pos = db['eef_pose'][i, :3, 3]  # (x, y, z)
                rpy = np.array(mat2euler(db['eef_pose'][i, :3, :3]))  # (roll, pitch, yaw)

                # calculate delta action to N step
                N = 10
                # use eef_pose[t+1] - eef_pose[t]
                if i < len(db['timestamps']) - N:
                    next_pos = db['eef_pose'][i + N, :3, 3]
                    next_rpy = np.array(mat2euler(db['eef_pose'][i + N, :3, :3]))
                    delta_pos = next_pos - pos
                    delta_rpy = next_rpy - rpy
                    delta_gripper = [1 if db['gripper_cmd'][i+N] <= 0 else -1]
                else:
                    delta_pos = np.zeros(3)
                    delta_rpy = np.zeros(3)
                    delta_gripper = [1 if db['gripper_cmd'][i] <= 0 else -1]

                episode.append({
                    'observation': {
                        'image': db['rgb_imgs'][2][i],  # 2 is front, 1 is side, 0 is top
                        'state': np.concatenate((pos, rpy)),
                        'gripper_state': np.array([1 if db['gripper_cmd'][i] <= 0 else -1], dtype=np.int8),
                    },
                    'action': np.concatenate((delta_pos, delta_rpy, delta_gripper), dtype=np.float32),
                    'discount': 1.0,
                    'reward': float(i == (len(db['timestamps']) - 1)),
                    'is_first': i == 0,
                    'is_last': i == (len(db['timestamps']) - 1),
                    'is_terminal': i == (len(db['timestamps']) - 1),
                    'language_instruction': "Pick up the coke can",
                    'language_embedding': np.zeros((1,), dtype=np.float16),
                })

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path
                }
            }

            # if you want to skip an example for whatever reason, simply return None
            return episode_path, sample

        # create list of all examples
        episode_paths = glob.glob(path)

        # for smallish datasets, use single-thread parsing
        for sample in episode_paths:
            yield _parse_example(sample)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )

