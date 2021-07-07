from pathlib import Path

from PIL import Image, UnidentifiedImageError
from PIL.ImageOps import grayscale
from tqdm import tqdm

def resize_images(input_dir: str, output_dir: str, size: int, to_grayscale=False):
    """Resize and pad .jpg images to given size.

    Args:
        input_dir (str): Directory containing images.
        output_dir (str): Directory to save to.
        size (int): Desired size.
    """
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    files = list(input_dir.rglob('*.jpg'))
    current_files = sum(1 for _ in output_dir.rglob('*.jpg'))

    for i, path in enumerate(tqdm(files, desc="Resizing images ...", total=len(files))):
        try:
            image = Image.open(path)
        except UnidentifiedImageError:
            continue
        if to_grayscale:
            image = grayscale(image)
        image.thumbnail((size, size), resample=Image.LANCZOS)
        height, width = image.size
        if height > 45 and width > 45:
            image.save(output_dir / f'{(current_files + i):05}.jpg')

def main():
    resize_images('data/raw', 'data/processed/train', size=100, to_grayscale=True)


if __name__ == "__main__":
    main()





