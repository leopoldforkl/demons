Folder structure:
dogms/
├── __pycache__/
├── __init__.py
├── configs/
│   ├── config_test.json
│   └── config.py
├── src/
│   ├── __pycache__/
│   ├── vod/
│   │   ├── configuration/
│   │   └── ...
│   │   ├── frame/
│   │   │   ├── __init__.py
│   │   │   ├── data_loader.py
│   │   │   ├── labels.py
│   │   │   └── transformations.py
│   │   ├── __init__.py
│   │   └── frame_loader.py
│   ├── pc_processing.py
│   ├── plotter_functions.py
│   ├── __init__.py
│   ├── test.py
│   └── utils.py
└── main.py
