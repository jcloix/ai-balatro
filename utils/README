# utils 🛠

The `utils` module is part of the **Balatro** project.  
It provides **helper functions** to support other modules, making it easier to build, annotate, train, and identify cards.  

---

## 🚀 Features

- Simple functions to query card presence  
- Reusable code for different modules  
- Utility helpers for image loading, preprocessing, and checks  
- Keeps project logic clean and modular  

---

## 🛠 Installation

Make sure you are inside the Balatro project folder and dependencies are installed:

```bash
cd balatro
pip install -r requirements.txt
```

---

## 🎯 Usage

Import utilities into your code:

```python
from utils import is_card_displayed, load_image

if is_card_displayed("card_123"):
    print("Card 123 is currently on screen!")

image = load_image("screenshots/card_001.png")
```

---

## 📌 Example Functions

- `is_card_displayed(card_id: str) -> bool`  
  Returns `True` if the given card ID is detected on screen.  

- `load_image(path: str)`  
  Loads an image for preprocessing.  

- `preprocess_image(img)`  
  Apply transformations before feeding into the model.  

---

## 🔄 Workflow Context

```text
utils
  ↳ supports → build_dataset
  ↳ supports → annotate_dataset
  ↳ supports → train_model
  ↳ supports → identify_card
```

---

## 📌 Notes

- This module is meant to stay lightweight.  
- Additional helpers can be added as the project grows.  
- Avoid overloading `utils` with unrelated logic — keep it focused.  

---

## 🤝 Contributing

Contributions are welcome! If you have useful helper functions that could simplify the project, feel free to submit them.  

---

## 📄 License

Specify your license here (MIT, GPL, etc.).
