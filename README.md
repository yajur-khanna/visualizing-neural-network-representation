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

## XOR Geometry and Feature Space Warp

The visualization shows how a neural network learns to solve the XOR classification problem by transforming the representation of the input data.

### XOR in 2D (Input Space)

The plot on the left shows the classic **XOR dataset in two dimensions**.

In this dataset, **diagonally opposite quadrants belong to the same class**. Points in the top-left and bottom-right belong to one class, while points in the top-right and bottom-left belong to the other.

Because of this arrangement, the dataset is **not linearly separable**.  
No straight line can correctly divide the red and blue points.

You can try this yourself: draw any line on the plot, and at least one point from each class will always end up on the wrong side.

This is why XOR is one of the classic examples used to demonstrate the limitations of **linear classifiers**.

---

### Feature Space Warp (Learned Representation)

The right plot shows the **learned feature representation inside the neural network**.

The network first applies **linear transformations** that map the 2D input into a higher-dimensional feature space (3D in this example).  
After each linear transformation, a **non-linear activation function (tanh)** reshapes the geometry of the space.

The grid shown in the visualization represents how the input space is being **warped** during training. Regions of the space stretch, compress, and bend as the network learns a better representation.

Over time, the points move into positions where the two classes become **linearly separable in the transformed space**.

At that point, the final layer of the network only needs to fit a **simple hyperplane** to separate the classes.

Although the boundary is linear in this learned 3D feature space, it corresponds to a **nonlinear decision boundary in the original 2D input space**.

---

### Visualization


https://github.com/user-attachments/assets/3acb5f71-1e1d-4016-8ad5-a32524939a33

---


## Files generated

- `xor_warp_training.avi` (created by MATLAB `VideoWriter`)
- `xor_warp_training.mp4` (created by converting AVI → MP4 via `ffmpeg`)

Both save into your MATLAB current working directory:
```matlab
pwd
