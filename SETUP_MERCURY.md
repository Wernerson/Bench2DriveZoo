# Bench2DriveZoo setup on mercury

Step-by-step instructions to reproduce the baseline setup on the mercury GPU server.
Written to be executed by Claude Code (or by hand). Mercury already has the **full
Bench2Drive-Base set** at `/scratch/brunns/data/bench2drive/Bench2Drive-Base`, so unlike
the workstation (mini set, 10 clips), a real reproduction with the official train/val
split is possible here.

## What was already done on the workstation (context)

- `mmcv/datasets/prepare_B2D.py` was patched: new `--split` CLI arg (defaults to the
  official base split) and val clips missing on disk are skipped instead of crashing.
- `data/splits/bench2drive_mini_train_val_split.json` added (only relevant for the mini set).
- HD maps + checkpoints were downloaded (same downloads are repeated below for mercury).
- The pipeline was validated end-to-end on the mini set (converter runs CPU-only).

Both changes are in the fork `git@github.com:Wernerson/Bench2DriveZoo.git` (push from the
workstation first if not yet pushed).

## 1. Clone the fork

```bash
cd /scratch/brunns
git clone git@github.com:Wernerson/Bench2DriveZoo.git
cd Bench2DriveZoo
```

The repo already tracks `data/splits/bench2drive_base_train_val_split.json` and
`data/others/b2d_motion_anchor_infos_mode6.pkl` (needed by UniAD), so no extra download.

## 2. Link the raw data

```bash
mkdir -p data/bench2drive
ln -s /scratch/brunns/data/bench2drive/Bench2Drive-Base data/bench2drive/v1
# Verify structure: must list scenario clip dirs, each containing anno/ camera/ expert_assessment/
ls data/bench2drive/v1 | head
ls data/bench2drive/v1/$(ls data/bench2drive/v1 | head -1)
```

If clips are nested one level deeper (HF tarball extraction sometimes does this), point the
symlink at the directory whose children are the `<Scenario>_<Town>_<Route>_<Weather>` dirs.

## 3. Download HD maps (~5 GB)

```bash
mkdir -p /scratch/brunns/data/bench2drive/maps
cd /scratch/brunns/data/bench2drive/maps
for t in Town01 Town02 Town03 Town04 Town05 Town06 Town07 Town10HD Town11 Town12 Town13 Town15; do
  curl -sL -C - -O "https://huggingface.co/datasets/rethinklab/Bench2Drive-Map/resolve/main/${t}_HD_map.npz"
done
cd /scratch/brunns/Bench2DriveZoo
ln -s /scratch/brunns/data/bench2drive/maps data/bench2drive/maps
```

## 4. Download checkpoints (~2 GB)

```bash
mkdir -p ckpts && cd ckpts
for f in resnet50-19c8e357.pth r101_dcn_fcos3d_pretrain.pth vad_b2d_base.pth uniad_base_b2d.pth; do
  curl -sL -C - -O "https://huggingface.co/rethinklab/Bench2DriveZoo/resolve/main/$f"
done
cd ..
```

Expected sizes (bytes): resnet50 `102502400`, r101 `225215819`, vad `699792372`, uniad `996840308`.
(If a file is ~1 KB it's a git-LFS pointer — re-download.)

## 5. Build the environment (the heavy part)

Strictly Python 3.8 + CUDA 11.8 + GCC ~9.4 (see `docs/INSTALL.md`). The install compiles
~36 CUDA kernels of the vendored mmcv/mmdet3d — run it on a node with a GPU visible so the
right compute arch is detected, or set `TORCH_CUDA_ARCH_LIST` explicitly.

```bash
conda create -n b2d_zoo python=3.8 -y
conda activate b2d_zoo
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# GCC 9.x: use system gcc-9 if present, otherwise:
#   conda install -c conda-forge gcc_linux-64=9.5 gxx_linux-64=9.5 -y
export CUDA_HOME=$CONDA_PREFIX
pip install ninja packaging
pip install -v -e .          # compiles CUDA ops; takes a while
pip install -r requirements.txt
```

Sanity check:

```bash
python -c "from mmcv.ops import iou3d_cuda; import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 6. Generate the info pkls (~1 h with 16 workers on the base set)

```bash
cd mmcv/datasets
python prepare_B2D.py --workers 16      # uses the official base split by default
cd ../..
ls -la data/infos/   # expect b2d_infos_train.pkl, b2d_infos_val.pkl, b2d_map_infos.pkl
```

CPU-only; the b2d_zoo env works. (On newer matplotlib ≥3.9 `vis_utils.py` breaks on
`cm.get_cmap` — not an issue with the py3.8 env.)

## 7. Baseline milestone 1: open-loop eval of released checkpoints (no training)

```bash
# VAD (planning L2/collision + det mAP/NDS):
bash adzoo/vad/dist_test.sh adzoo/vad/configs/VAD/VAD_base_e2e_b2d.py ckpts/vad_b2d_base.pth 1
# UniAD:
bash adzoo/uniad/uniad_dist_eval.sh adzoo/uniad/configs/stage2_e2e/base_e2e_b2d.py ckpts/uniad_base_b2d.pth 1
```

Compare against the README model-zoo table. Note: UniAD reports per-timestep L2, VAD reports
interval-averaged L2 (`docs/TRAIN_EVAL.md`) — don't compare the two directly.

## 8. Baseline milestone 2: retrain VAD (easiest model)

```bash
# 6 epochs, samples_per_gpu=1; docs use 1 GPU, more GPUs shorten wall time
bash adzoo/vad/dist_train.sh adzoo/vad/configs/VAD/VAD_base_e2e_b2d.py 1
```

BEVFormer (`adzoo/bevformer/dist_train.sh ... 4`) and UniAD (stage 1 track+map on 4 GPUs →
stage 2 e2e on 1 GPU, `adzoo/uniad/uniad_dist_train.sh`) per `docs/TRAIN_EVAL.md` if needed.

## 9. (Optional, later) closed-loop CARLA eval

Driving Score / Success Rate need CARLA 0.9.15 + the Bench2Drive leaderboard repo
(`docs/EVAL_IN_CARLA.md`, `docs/INSTALL.md` STEP 8). Skip until open-loop numbers look sane.
