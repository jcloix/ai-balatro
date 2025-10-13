# Step 1: Collect data + run AHK script to help
python -m build_dataset.main  
/scripts/ahk/build_dataset.ahk 

# Step 2: Annotate labels (Optionally: Can use pretrained model for inference)
python -m utils.inference_utils --img-dir "data/dataset_default" --task-name identification --topk 2 --out-json "data/dataset_default/inference.json"

python -m streamlit run annotate_dataset/main.py

# Step 3: Generate offline augmentations
# Generate for identification task
python -m augment_dataset.main --input data/dataset_default --output data/augmented_identity --labels data/dataset_default/labels.json --no-blur --no-bc --balance-by name
python -m augment_dataset.save_augmented_labels --aug-dir "data/augmented_identity" --labels "data/augmented_identity/augmented.json"
# Generate for modifier task
python -m augment_dataset.main --input data/dataset_default --output data/augmented_modifier --labels data/dataset_default/labels.json --balance-by modifier
python -m augment_dataset.save_augmented_labels --aug-dir "data/augmented_modifier" --labels "data/augmented_modifier/augmented.json"

# Step 4: Train the model
# Train the model - identification
python -m train_model.train --tasks identification --epochs 7 --log-dir runs/identification --freeze-strategy all --optimizer simple --scheduler none --use-weighted-sampler

# Train the model - identification
python -m train_model.train --tasks modifier --epochs 10 --log-dir runs/modifier --freeze-strategy all --optimizer simple --scheduler none --use-weighted-sampler

# Step5: TODO





