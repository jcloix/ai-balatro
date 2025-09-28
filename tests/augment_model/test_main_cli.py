import sys
import subprocess
import tempfile
import os
from PIL import Image

def test_cli_runs_and_creates_files():
    with tempfile.TemporaryDirectory() as input_dir, tempfile.TemporaryDirectory() as output_dir:
        img = Image.new("RGBA", (32, 32), (255, 0, 0, 255))
        img_path = os.path.join(input_dir, "card1.png")
        img.save(img_path)

        # Use the same Python executable as the current environment
        python_exe = sys.executable

        result = subprocess.run(
            [python_exe, "augment_dataset/main.py",
             "--input", input_dir,
             "--output", output_dir,
             "--n", "2",
             "--seed", "123"],
            capture_output=True,
            text=True
        )

        print(result.stdout)
        print(result.stderr)

        assert result.returncode == 0
        output_files = os.listdir(output_dir)
        assert len(output_files) == 2
        assert "card1_aug0.png" in output_files
        assert "card1_aug1.png" in output_files
