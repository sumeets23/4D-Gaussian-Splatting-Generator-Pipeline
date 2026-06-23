# 4D Gaussian Splatting Generator Pipeline


This repository is a demo-only version of the 4D Gaussian Splatting Generator Pipeline project. It contains the generated 4D Gaussian output, the Unity runtime playback scripts, and the renderer-side assets needed to preview the sequence in Unity.

It does not include the reconstruction generator pipeline, Python backend, editor tooling, training dependencies, or dataset processing stack.

## What This Demo Shows

- Playback of a generated 4D Gaussian Splatting sequence inside Unity.
- Runtime frame streaming using generated binary frame caches.
- Integration between a custom playback controller and Gaussian splat renderer assets.
- A lightweight demo project structure suitable for portfolio and showcase use.

## What Is Included

- Generated 4D output under `Assets/4D Generated/`
- Runtime playback scripts under `Assets/UnityScanLab/Scripts/FourD/`
- Shader and compute assets used for playback
- Demo scene and Unity project settings
- Package dependencies required for rendering

## Project Background

This  project is a Unity Editor plus Python pipeline for reconstructing dynamic scenes as 4D Gaussian Splatting sequences from multi-camera video. The production backend is 4C4D. The complete system supports:

multi-camera videos
-> frame extraction
-> frame sync validation
-> COLMAP calibration
-> optional MASt3R initialization
-> 4C4D training
-> checkpoint validation
-> per-frame PLY export
-> Unity import
-> runtime playback


This demo repo starts after reconstruction is already complete and focuses only on rendering and playback inside Unity.

The important design choice is that playback does not require reparsing all PLY frames every time. The importer converts source frames into runtime binary caches that are lazy-loaded during playback.

## Opening the Demo

1. Open the project in the supported Unity version configured in the repository.
2. Load the included demo scene.
3. Select the generated prefab in the scene if you want to inspect playback settings.
4. Enter Play Mode to preview the animated 4D Gaussian sequence.


## Tech Stack

- Unity 6
- C#
- Gaussian splat renderer integration
- Generated 4D Gaussian runtime assets
