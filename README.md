# DEMONS - Dynamic Environment Mapping Of Non-stationary Scenes

**DEMONS** is a Python-based framework for retrieving dynamic environment maps—such as voxel-based representations—from sensor data. It is planned to support input from radar, lidar, and stereo camera point clouds and implementations of several state-of-the-art methods for dynamic scene reconstruction.

---

## 🚀 Features (In Progress)

- Dynamic voxel grid mapping
- Support for non-stationary environments
- Visualization support using PyVista

---

## 🔧 Installation

### From GitHub

1. Clone the repository:

   ```bash
   git clone https://github.com/leopoldforkl/demons.git
   cd demons
   ```

2. Create the Conda environment:

   ```bash
   conda env create -f environment.yml
   ```

3. Activate the environment:

   ```bash
   conda activate demons-env
   ```

4. Install PyVista separately (due to version compatibility):

   ```bash
   pip install pyvista==0.38.6
   ```

5. **Note**:

    To remove the environment from your system, use:

    ```bash
    conda env remove --name demons-env
    ```

    You can list available environments with:

    ```bash
    conda env list
    ```

---

## 📁 Dataset & Dependencies

This repository leverages parts of the [View of Delft (VoD)](https://github.com/tudelft-iv/view-of-delft-dataset) development kit.

If using the dataset or associated tools, make sure to follow the license terms below.

---

## 📜 License and Credits

* **Code License**: Apache License 2.0 (see `LICENSE`)
* **Dataset**: Subject to the [VoD Research Use License](https://github.com/tudelft-iv/view-of-delft-dataset#license)

**Credits** to the View of Delft team: ToDo


### 🔖 Citation

If you use the View of Delft dataset, please cite:

```bibtex
@ARTICLE{apalffy2022,
  author={Palffy, Andras and Pool, Ewoud and Baratam, Srimannarayana and Kooij, Julian F. P. and Gavrila, Dariu M.},
  journal={IEEE Robotics and Automation Letters}, 
  title={Multi-Class Road User Detection With 3+1D Radar in the View-of-Delft Dataset}, 
  year={2022},
  volume={7},
  number={2},
  pages={4961-4968},
  doi={10.1109/LRA.2022.3147324}
}
```

---

## 🤝 Contributions

Contributions, feedback, and suggestions are welcome via GitHub issues or pull requests.

---