# Augment identity cards
python augment_script.py --input data/dataset_default --output data/augmented_identity --labels data/dataset_default/labels.json --balance-by name 
# Augment modifier cards
python -m augment_dataset.main --input data/dataset_default --output data/augmented_modifier --labels data/dataset_default/labels.json --balance-by modifier

# Create augmented.json for both augmentation folders
python -m augment_dataset.save_augmented_labels --aug-dir data/augmented_identity --labels data/dataset_default/labels.json
python -m augment_dataset.save_augmented_labels --aug-dir data/augmented_modifier --labels data/dataset_default/labels.json


# Train the model, first freeze for 15 epochs, then unfreeze for 50 epochs
tensorboard --logdir runs/duo_head_training
python -m train_model.train --epochs 15 --log-dir runs/duo_head_training --freeze-backbone
# Train longer with adjusted strategies
tensorboard --logdir runs/duo_head_finetune
python -m train_model.train  --epochs 60  --log-dir runs/duo_head_finetune  --freeze-strategy high --use-weighted-sampler  --resume data/models/best_model.pth  --optimizer group  --scheduler cosine

# Fine tune to win again 2% - Stopped at 59, best model at 55
tensorboard --logdir runs/duo_head_finetune
python -m train_model.train --epochs 70 --patience 10 --log-dir runs/duo_head_finetune --resume data/models/best_model.pth --freeze-strategy mid --use-weighted-sampler --optimizer group --optimizer-lr-backbone 1e-5 --optimizer-lr-heads 5e-5 --optimizer-weight-decay 1e-4 --scheduler cosine --scheduler-tmax 50

tensorboard --logdir runs/duo_head_finetune_v2
python -m train_model.train --epochs 80 --patience 10 --log-dir runs/duo_head_finetune_v2 --resume data/models/best-model-duo-45.pth --freeze-strategy mid --use-weighted-sampler --optimizer group --optimizer-lr-backbone 8e-6 --optimizer-lr-heads 4e-5 --optimizer-weight-decay 1e-4 --scheduler cosine --scheduler-tmax 50



python -m train_model.train \
  --epochs 70 \
  --patience 10 \
  --log-dir runs/duo_head_finetune_v2 \
  --resume data/models/epoch_45.pth \
  --freeze-strategy mid \
  --use-weighted-sampler \
  --optimizer group \
  --optimizer-lr-backbone 8e-6 \
  --optimizer-lr-heads 4e-5 \
  --optimizer-weight-decay 1e-4 \
  --scheduler cosine \
  --scheduler-tmax 50