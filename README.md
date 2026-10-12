# VitaVision RoboSynChallenge preliminary submission

Team: VitaVision (`p2417822@mpu.edu.mo`)

This repository is the evaluation overlay for the ten-task preliminary submission. Checkpoints that differ from the released ACT weights are in the public model repository [Andrew-98/vitavision-robosyn-act](https://huggingface.co/Andrew-98/vitavision-robosyn-act). Three tasks use the released RoboSynChallenge ACT checkpoints directly.

Local 100-episode `random` results used to choose these weights:

| Task | Success | Policy | Checkpoint |
| --- | ---: | --- | --- |
| table_rearrangement | 97/100 | ACT + PhaseLoRA | `Andrew-98/vitavision-robosyn-act` subdirectory `table_rearrangement` |
| water_pouring | 95/100 | ACT prefix fine-tune | subdirectory `water_pouring` |
| mixer_operating | 87/100 | ACT prefix fine-tune | subdirectory `mixer_operating` |
| click_bell | 78/100 | released ACT | `RoboSynChallenge/ACT_sim_click_bell` |
| manipulate_pipette | 66/100 | released ACT | `RoboSynChallenge/ACT_sim_manipulate_pipette` |
| item_assembly | 63/100 | released ACT | `RoboSynChallenge/ACT_sim_item_assembly` |
| handle_basket | 40/100 | ACT retrained in 50-step segments | subdirectory `handle_basket` |
| drawer_open_place | 34/100 | ACT fine-tuned with empty-grasp recovery data | subdirectory `drawer_open_place` |
| items_handover | 6/100 | ACT suffix fine-tune | subdirectory `items_handover` |
| sample_loading | 6/100 | ACT + PhaseLoRA | subdirectory `sample_loading` |

## How to evaluate

Start from the official benchmark checkout, then copy this overlay on top of it:

```bash
git clone https://github.com/EDEM-AI/RoboSynChallenge.git
cd RoboSynChallenge
git clone https://github.com/vitem98/vitavision-robosyn.git /tmp/vitavision-robosyn
cp -a /tmp/vitavision-robosyn/policy/act policy/act
cp -a /tmp/vitavision-robosyn/policy/phaselora policy/phaselora
cp -a /tmp/vitavision-robosyn/policy/inference_timing.py policy/inference_timing.py
```

Install the official RoboSynChallenge / EmbodiChain environment, then download the fine-tuned weights:

```bash
export HF_HUB_ENABLE_HF_TRANSFER=0
huggingface-cli download Andrew-98/vitavision-robosyn-act --local-dir checkpoints/vitavision
export EMBODICHAIN_SIM_EXIT_PROCESS=0
```

Run each task for 100 random episodes. PhaseLoRA adapters load the released ACT base named in `adapter_config.json`.

```bash
bash policy/phaselora/eval.sh table_rearrangement random checkpoints/vitavision/table_rearrangement 0 --headless True --max_episodes 100 --eval_video_log False --seed 0
bash policy/act/eval.sh water_pouring random checkpoints/vitavision/water_pouring 0 --headless True --max_episodes 100 --eval_video_log False --seed 0
bash policy/act/eval.sh mixer_operating random checkpoints/vitavision/mixer_operating 0 --headless True --max_episodes 100 --eval_video_log False --seed 0
bash policy/act/eval.sh click_bell random RoboSynChallenge/ACT_sim_click_bell 0 --headless True --max_episodes 100 --eval_video_log False --seed 0
bash policy/act/eval.sh manipulate_pipette random RoboSynChallenge/ACT_sim_manipulate_pipette 0 --headless True --max_episodes 100 --eval_video_log False --seed 0
bash policy/act/eval.sh item_assembly random RoboSynChallenge/ACT_sim_item_assembly 0 --headless True --max_episodes 100 --eval_video_log False --seed 0
bash policy/act/eval.sh handle_basket random checkpoints/vitavision/handle_basket 0 --headless True --max_episodes 100 --eval_video_log False --seed 0
bash policy/act/eval.sh drawer_open_place random checkpoints/vitavision/drawer_open_place 0 --headless True --max_episodes 100 --eval_video_log False --seed 0
bash policy/act/eval.sh items_handover random checkpoints/vitavision/items_handover 0 --headless True --max_episodes 100 --eval_video_log False --seed 0
bash policy/phaselora/eval.sh sample_loading random checkpoints/vitavision/sample_loading 0 --headless True --max_episodes 100 --eval_video_log False --seed 0
```

Training data is the official simulated RoboSynChallenge demonstration set, 1000 episodes per task. No extra real-robot data was added.
