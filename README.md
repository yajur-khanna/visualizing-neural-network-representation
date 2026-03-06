# XOR → 3D Warp Visualization (MATLAB)

This script visualizes **how a small neural network learns XOR** by:
1) showing the original XOR dataset in **2D**, and  
2) showing the network’s learned hidden representation in **3D**, where a **linear hyperplane** can separate the classes.

It also **records the animation to video** and converts it to **MP4** using `ffmpeg`.

---

## What you’ll see

### Left panel (2D)
- The classic XOR pattern: diagonally opposite quadrants belong to the same class.
- A straight line decision boundary in 2D cannot separate XOR.



### Right panel (3D)
- Each point is the network’s learned feature representation (the output of the 3D hidden layer).
- The grid wireframe shows how the input space is being “warped” in feature space.
- A semi-transparent plane (hyperplane) shows the final linear classifier’s decision boundary in 3D.
- As training progresses, the classes become linearly separable in this learned space.

---

## Model overview

Architecture:
- **2 → H → 3 → 1**
- Activations: `tanh`, `tanh`, `sigmoid`
- Loss: binary cross-entropy
- Optimizer: gradient descent (manual backprop)

Key idea:
- Linear layers change coordinates / representation.
- Non-linearities reshape geometry so a linear separator can work.

---

## Feature space classification

https://github.com/user-attachments/assets/87309d0f-b54f-49dd-aaab-d983222ead4a


## Files generated

- `xor_warp_training.avi` (created by MATLAB `VideoWriter`)
- `xor_warp_training.mp4` (created by converting AVI → MP4 via `ffmpeg`)

Both save into your MATLAB current working directory:
```matlab
pwd
