# Age Detector

Facial age estimation from a photo or live webcam feed, using a ResNet34 model
fine-tuned on the UTKFace dataset.

## What this does
- Detects a face in an image using MTCNN (`facenet-pytorch`)
- Crops and feeds it into a ResNet34 regression model
- Outputs a predicted age in years

On its validation set, the model has a mean absolute error of roughly 4.7 years.
Error increases for faces over 60 and for heavily retouched/stock-style photos,
which differ from the training data.

## Files
- `app.py` — Flask web app (upload a photo in the browser, get a predicted age)
- `live_age_detector.py` — runs the model live on your webcam feed
- `requirements.txt` — Python dependencies
- `templates/index.html` — frontend page for the web app

## Model weights
Due to GitHub's file size limit, the trained model
(`age_model_resnet34_v2.pth`, ~83MB) is hosted separately:

**[Download age_model_resnet34_v2.pth](https://drive.google.com/file/d/1L_LU61qPk6hSVx3aLzFoDYcyWkVebgnN/view?usp=sharing)**

Download it and place it in the project's root folder (same folder as `app.py`)
before running either script.

## Setup

```bash
pip install -r requirements.txt
```

## Run the web app

```bash
python app.py
```

Then open `http://127.0.0.1:7860` in your browser.

## Run the live webcam version

```bash
python live_age_detector.py
```

Press `q` to quit the webcam window.

## Limitations
This model was trained on UTKFace, which is imbalanced by age and ethnicity.
Accuracy is lower on elderly faces and faces from underrepresented groups in
the dataset. This is a research/learning project, not a production-grade tool.
