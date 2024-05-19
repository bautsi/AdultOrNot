AdultOrNot/
│
├── venv/
│
├── data/
│ ├── input/
│ ├── train/
│ └── test/
│
├── scripts/
│ ├── classify_images.py # original_images -> adult / minor
│ ├── data_rename.py # 000125A22 -> A125
│ ├── random_split.py # adult / minor -> train / test
│ ├── VGG16_train.py
│ └── predict_age.py # input unique images
│
├── models/
│ └── VGG16_1.h5
│
├── train_results/ # matplotlib
│
├── Log.txt
│
└── README.md

cmd
-> cd path to this project
-> python -m venv venv
-> .\venv\Scripts\activate
-> pip install tensorflow
-> pip install matplotlib

terminal or cmd -> cd path to this project
-> .\venv\Scripts\activate (if haven't activate venv yet)
-> python .\scripts\python script you want to run