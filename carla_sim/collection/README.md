# CARLA data collection

This folder holds the exact pipeline that recorded the CARLA dataset used to train the neural simulator. It is a verbatim copy of the companion repository [carla-imitation-learning](https://github.com/JordyKieto/carla-imitation-learning) at commit `50d28e2`. Both projects train on the same demonstrations: 448 recordings, 3.32 h at 20 Hz.

Run every script from **this folder**. It needs a running CARLA 0.9.16 server and the `carla` pip client.

| Script | What it does |
|---|---|
| `drive.py` | Drive manually with the arrow keys or engage the CARLA autopilot, and record. `R` starts and stops a recording. |
| `collect.py` | Automated autopilot collection over every spawn point of a map. It balances left, right and straight maneuvers. |
| `preprocess.py` | Turns raw recordings into processed TFRecords (96×96 RGB) plus a window manifest (`windows_manifest.parquet`). |

```bash
cd carla/collection
python drive.py Town01                               # manual driving, recording on R
python collect.py --map Town01 --seconds 30          # autopilot collection
python preprocess.py                                 # -> datasets/processed/*.processed.tfrecord + manifest
```

`config.yaml` defines the sensor rig: a 224×224 FOV-90 camera, a 32-channel 50 m LiDAR binned into a 64×64 BEV, 5-d telemetry and 25 route waypoints. The processed format is what `carla_sim/data.py` reads and what `carla_sim/train.py` trains on. Record schema, sensor details and known quirks are documented in the companion repository's README.

**Known quirk that matters for the world model.** In autopilot and policy mode, the route waypoints are drawn into the CARLA world as green debug points, so they appear in the recorded camera frames. The neural simulator learned to render this route overlay (see the paper, §8).
