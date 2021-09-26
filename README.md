# Cylmarker

[![GitHub stars](https://img.shields.io/github/stars/Cartucho/cylmarker.svg?style=social&label=Stars)](https://github.com/Cartucho/cylmarker)

This marker can be used to estimate the 6DoF (location + orientation) of cylindrical objects.


<img src="https://user-images.githubusercontent.com/15831541/134814396-bf02f8b6-a3f7-4c33-a19d-171880fcc3e6.png" width="50%">


## How to run the code?

We recommend you to create a Python virtual environment:

```
python3.9 -m pip install --user virtualenv
python3.9 -m virtualenv venv
```

Then you can activate that environment and install the requirements using:
```
source venv/bin/activate
pip install -r requirements.txt
```

Now, when the `venv` is activated you can run the code using:

```
python main.py
```

Feel free to adjust the settings in [data/config.yaml](https://github.com/Cartucho/cylmarker/blob/main/data/config.yaml) file.

## How to create a new marker?

Edit the [data/config.yaml](https://github.com/Cartucho/cylmarker/blob/main/data/config.yaml) file.
Then, run the following command:

```
python main.py --task m
```

The resulting marker will be inside the folder `data/` and is ready to be used.

## How to print the marker?

Follow the instructions shown in the [data/marker_how_to_print.svg](https://github.com/Cartucho/cylmarker/blob/main/data/marker_how_to_print.svg) file.

## Paper

The paper can be found at [TODO add link when paper is published]()

If you use this marker please consider citing our paper:

```bibtex
@article{cartucho2021cylmarker,
  title={An Enhanced Marker Pattern that Achieves Improved Accuracy in Surgical Tool Tracking},
  author={Cartucho, Jo{\~a}o and Wang, Chiyu and Huang, Baoru and S. Elson, Daniel and Dariz, Ara and Giannarou, Stamatia},
  journal={Computer Methods in Biomechanics and Biomedical Engineering: Imaging \& Visualization},
  pages={1--15},
  year={2021},
  publisher={Taylor \& Francis}
}
```
