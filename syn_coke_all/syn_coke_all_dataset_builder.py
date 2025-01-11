from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import pickle as pkl
import math
import cv2

from transform_utils import mat2quat, quat2axisangle, mat2euler, quat2mat, axisangle2quat



class SynCokeAll(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Tensor(
                            shape=(224, 224, 3),
                            dtype=np.uint8,
                            doc='Front camera RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                        shape=(6,),
                        dtype=np.float32,
                        doc='Robot state, consists of [eef_pos (x, y, z), '
                            'eef_euler (roll, pitch, yaw)].',
                        ),
                        'gripper_state': tfds.features.Tensor(
                        shape=(1,),  # scalar of gripper width
                        dtype=np.float32,
                        doc='gripper_state (x).',
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
                        doc='CMD Robot action: EEF Delta XYZ (3) + Roll-Pitch-Yaw (3) + Gripper Open/Close (1)',
                        # xyz can leave as is
                        # rpy can convert to euler
                        # gripper needs to be mapped to 0 -> close, 1 -> open
                    # ),
                    # 'discount': tfds.features.Scalar(
                    #     dtype=np.float32,
                    #     doc='Discount if provided, default to 1.'
                    # ),
                    # 'reward': tfds.features.Scalar(
                    #     dtype=np.float32,
                    #     doc='Reward if provided, 1 on final step for demos.'
                    # ),
                    # 'is_first': tfds.features.Scalar(
                    #     dtype=np.bool_,
                    #     doc='True on first step of the episode.'
                    # ),
                    # 'is_last': tfds.features.Scalar(
                    #     dtype=np.bool_,
                    #     doc='True on last step of the episode.'
                    # ),
                    # 'is_terminal': tfds.features.Scalar(
                    #     dtype=np.bool_,
                    #     doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    # 'language_embedding': tfds.features.Tensor(
                    #     shape=(1,),
                    #     dtype=np.float16,
                    #     doc='Kona language embedding. '
                    #         'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    # ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits.
        # I have 1000 demos, so I will use 800 for training and 200 for validation
        """
        return {
            'train': self._generate_examples(path=[f'data/demo_{i}.pkl' for i in range(1000)]),
            # 'val': self._generate_examples(path=[f'data/demo_{i}.pkl' for i in range(800, 1000)]),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path):
            # load raw data --> this should change for your dataset
            with open(episode_path, 'rb') as dbfile:
                db = pkl.load(dbfile)
            # breakpoint()
            # assemble episode --> here we're assuming demos
            episode = []
            for i in range(db['rgb_frames'].shape[0]):
                for j in range(5):
                    image = db['rgb_frames'][i, j]  # 0 is left shoulder camera, which was moved to roughly match the front camera
                    # image = image[:, 140:500]  # center crop 360x360 # images is already 256x256
                    image = cv2.resize(image, (224 ,224))  # size correctly
                    # images in pkl file are in BGR format, convert to RGB  # TODO
                    image = image[:, :, ::-1]

                    episode.append({
                        'observation': {
                            'image': image,
                            'state': np.concatenate((db['eef_pos'][i].squeeze(), mat2euler(quat2mat(db['eef_quat'][i]))), dtype=np.float32),
                            'gripper_state': np.array(db['gripper_state'][i: i + 1], dtype=np.float32),
                        },
                        'action': np.concatenate((
                            db['arm_action'][i][:3],
                            mat2euler(quat2mat(axisangle2quat(db['arm_action'][i][3:]))),  # convert to euler so openvla outputs euler
                            [db['gripper_action'][i]]
                            ), dtype=np.float32),
                        'language_instruction': "pick up the coke can",
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

        # # create list of all examples
        # episode_paths = glob.glob(path)
        episode_paths = path

        # for smallish datasets, use single-thread parsing
        for sample in episode_paths:
            yield _parse_example(sample)

        # # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )

