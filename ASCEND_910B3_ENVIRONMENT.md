# Ascend 910B3 ARM/Ubuntu Environment Research

This note captures the verified research for running `KirSerg64/Omni-Interaction-Agent` on **Ubuntu + ARM64 + Ascend 910B3**. It is based on the current repository files plus publicly visible Ascend Quay and compatibility documentation.

## Scope and conclusion

**Short version:** I could not verify a container-only solution that makes the **full agent** run on Ascend 910B3 **without code changes**.

The main reason is a three-way mismatch:

1. the repo currently expects **Python 3.10** and a **PyTorch 2.6** stack ([`environment.yml`](environment.yml#L7-L24), [`minicpm_ft/pyproject.toml`](minicpm_ft/pyproject.toml#L9-L32), [`gander_runtime/pyproject.toml`](gander_runtime/pyproject.toml#L9-L16))
2. currently published `quay.io/ascend/cann` 910B Ubuntu/Python 3.10 images I verified are in the **CANN 9.2 beta** family
3. the official Ascend compatibility matrix maps **PyTorch 2.6** to **CANN 8.2.RC1 / 8.3.RC1**, not to CANN 9.2

In addition, the repository runtime is currently **CUDA-specific** in the inference, detached Talker, and managed ASR paths, so moving to Ascend is not just a wheel-resolution problem.

## Verified repository constraints

### Python and package expectations

The repository environment file pins:

- `python=3.10`
- `torch==2.6.0`
- `torchvision==0.21.0`
- `torchaudio==2.6.0`
- `minicpmo-utils[tts]==1.0.6`  
  ([`environment.yml`](environment.yml#L7-L24))

The `mcpmft` package metadata further requires:

- `torch>=2.6,<2.7`
- `transformers==4.51.0`
- `deepspeed>=0.19,<0.20`
- optional ASR dependency `faster-whisper>=1.2,<2`  
  ([`minicpm_ft/pyproject.toml`](minicpm_ft/pyproject.toml#L9-L32))

The runtime package also requires Python 3.10+ and FastAPI/websocket serving components  
([`gander_runtime/pyproject.toml`](gander_runtime/pyproject.toml#L9-L16)).

### CUDA-specific runtime assumptions

The current inference/runtime code paths are written for CUDA:

- `load_for_infer()` aborts if CUDA is unavailable and otherwise moves the model to `"cuda"` ([`minicpm_ft/mcpmft/infer/common.py`](minicpm_ft/mcpmft/infer/common.py#L83-L89))
- detached Talker explicitly requires a CUDA device ([`minicpm_ft/mcpmft/infer/detached_talker.py`](minicpm_ft/mcpmft/infer/detached_talker.py#L140-L145))
- runtime GPU assignment validation only accepts `cuda:<index>` semantics ([`gander_runtime/gander_runtime/cli.py`](gander_runtime/gander_runtime/cli.py#L314-L358))
- managed ASR checks for `faster_whisper` and is configured around CUDA/CPU device strings ([`gander_runtime/gander_runtime/cli.py`](gander_runtime/gander_runtime/cli.py#L273-L279), [`minicpm_ft/mcpmft/infer/asr.py`](minicpm_ft/mcpmft/infer/asr.py#L129-L142))

That means an Ascend container alone does not make the full runtime NPU-ready.

## Verified Ascend container and compatibility findings

### Quay image family

The Ascend Quay repository description for `quay.io/ascend/cann` documents Ubuntu/openEuler base images and tags of the form:

`<cann_version>-<chip_series>-<os>-<python_version>[-devel]`

Relevant verified tags include:

- `9.2.0-beta.1-910b-ubuntu22.04-py3.10`
- `9.2.0-beta.1-910b-ubuntu22.04-py3.10-devel`

Sources:

- Quay repo description: <https://quay.io/api/v1/repository/ascend/cann>
- Supported tags list: <https://raw.githubusercontent.com/Ascend/cann-container-image/main/supported_tags.md>
- Quay tags API (observed current tags): <https://quay.io/api/v1/repository/ascend/cann/tag/?limit=200&page=1&onlyActiveTags=true>

### Official TorchNPU compatibility

The official Ascend compatibility table states:

- `torch_npu==2.6.0` pairs with **PyTorch 2.6.0** and **CANN 8.2.RC1**
- `torch_npu==2.6.0.post3` pairs with **PyTorch 2.6.0** and **CANN 8.3.RC1**
- current newer active lines on CANN 9.x start at later PyTorch families such as 2.7.1, 2.9, 2.10, 2.11, 2.12

Sources:

- <https://raw.githubusercontent.com/Ascend/pytorch/master/COMPATIBILITY.en.md>
- <https://raw.githubusercontent.com/Ascend/pytorch/master/COMPATIBILITY.md>

## Why the version conflicts happen

The conflicts are not coming from one package only:

1. **Repo side:** the project pins the core stack to `torch 2.6.x`
2. **Ascend side:** currently visible public 910B Ubuntu 22.04 Python 3.10 CANN images are `9.2.0-beta.*`
3. **Compatibility side:** official Ascend guidance pairs `torch 2.6` with older `CANN 8.2/8.3`, not `9.2`
4. **Runtime side:** the repo still assumes CUDA APIs and CUDA device naming in several code paths
5. **ASR side:** `faster-whisper` is not an Ascend-native NPU path in this repository

So the environment problem is both:

- **package/version alignment**
- **runtime backend assumptions in code**

## Recommended practical Docker baseline

If the goal is to create the **least-conflicting starting point for a port**, the safest base I verified is:

```dockerfile
FROM quay.io/ascend/cann:9.2.0-beta.1-910b-ubuntu22.04-py3.10-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs curl ca-certificates \
    build-essential cmake pkg-config \
    ffmpeg libsndfile1 \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/omni
COPY . /opt/omni

RUN python -m pip install --upgrade \
    pip==25.2 setuptools==80.9.0 wheel==0.45.1

# Use an Ascend-supported public PyTorch line for CANN 9.x.
# This does not satisfy the repo's current torch<2.7 metadata pin.
RUN python -m pip install \
    torch==2.10.0 \
    torch_npu==2.10.0.post6

RUN python -m pip install \
    accelerate==1.10.1 \
    av==16.0.1 \
    fastapi==0.116.1 \
    librosa==0.10.2.post1 \
    numpy==1.26.4 \
    pillow==11.3.0 \
    pyarrow==21.0.0 \
    pyyaml==6.0.2 \
    safetensors==0.6.2 \
    scipy==1.13.1 \
    soundfile==0.13.1 \
    sentencepiece==0.2.1 \
    tensorboard==2.20.0 \
    tqdm==4.67.1 \
    transformers==4.51.0 \
    uvicorn[standard]==0.35.0 \
    websockets==15.0.1 \
    minicpmo-utils[tts]==1.0.6

# Avoid dependency re-resolution against repo metadata until the CUDA-specific
# code and torch version constraints are updated for Ascend.
RUN python -m pip install -e ./minicpm_ft --no-deps && \
    python -m pip install -e ./gander_runtime --no-deps
```

## What this Docker baseline is for

This image is intended as:

- a **research / porting baseline**
- a way to reduce package-resolution conflicts on ARM + Ascend
- a starting point for validating imports and progressively replacing CUDA assumptions

It is **not** a verified full-production image for the current unmodified repository.

## What is likely to work vs. what is blocked

### Likely workable in the container

- Python 3.10 userland
- FastAPI / websocket runtime dependencies
- general repo installation for code inspection or limited non-CUDA paths
- model/config file handling

### Still blocked without code changes

- main inference path, because it requires CUDA availability and `model.to("cuda")` ([`minicpm_ft/mcpmft/infer/common.py`](minicpm_ft/mcpmft/infer/common.py#L83-L89))
- detached Talker, because it requires a CUDA device ([`minicpm_ft/mcpmft/infer/detached_talker.py`](minicpm_ft/mcpmft/infer/detached_talker.py#L140-L145))
- managed GPU assignment logic, because it validates only CUDA-form device syntax ([`gander_runtime/gander_runtime/cli.py`](gander_runtime/gander_runtime/cli.py#L314-L358))
- managed ASR on NPU, because this repo’s ASR path is built around `faster-whisper` and CUDA/CPU device selection ([`gander_runtime/gander_runtime/cli.py`](gander_runtime/gander_runtime/cli.py#L273-L279), [`minicpm_ft/mcpmft/infer/asr.py`](minicpm_ft/mcpmft/infer/asr.py#L129-L142))

## More conservative alternative

If strict version matching matters more than using the latest public 910B image family, the better target would be:

- **CANN 8.2.RC1 or 8.3.RC1**
- **Python 3.10**
- **PyTorch 2.6.0**
- matching **`torch_npu==2.6.0`** or **`torch_npu==2.6.0.post3`**

However, during this research I did **not** verify publicly available Quay tags for `8.2/8.3 + 910b + ubuntu22.04 + py3.10`, so I am not claiming a concrete published image name for that older combination.

## Suggested next engineering steps

1. replace hard-coded CUDA checks and device strings with backend-neutral or NPU-aware logic
2. decide whether ASR stays external/CPU-only instead of being ported to Ascend
3. choose between:
   - keeping repo-side `torch 2.6.x` and finding/constructing a matching older CANN base, or
   - moving the repo to an Ascend-supported newer torch line
4. only after that, re-enable normal dependency resolution without `--no-deps`

## Validation performed for this note

- verified current repository dependency pins and runtime assumptions from the checked-in source files linked above
- verified current public `quay.io/ascend/cann` 910B Ubuntu/Python 3.10 tag family from Quay metadata
- verified official TorchNPU ↔ PyTorch ↔ CANN compatibility from the published Ascend compatibility table

