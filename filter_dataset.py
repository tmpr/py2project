from pathlib import Path
from shutil import copy

import numpy as np
from PIL import Image
from tqdm import tqdm

_VALID_SUFFIXES = {".jpg", ".JPG", ".JPEG", ".jpeg"}

def filter_dataset(input_dir: str, output_dir: str, logfile: str) -> int:   
    """Separates invalid from valid images and stores them.

    Arguments:
    > `input_dir`: Path to folder containing images.

    > `output_dir`: Path where to save valid images.

    > `logfile`: Path where to store logfile where invalid files are annotated.

    Returns:
    > `int`: Number of valid files in `input_dir`.
    
    """
    valid_files, invalid_files = _filter_dataset(input_dir)
    _write_logfile(logfile, invalid_files)
    _save_images(output_dir, valid_files)

    return len(valid_files)


def _save_images(output_dir, valid_files) -> None:
    dest_dir = Path(output_dir)
    dest_dir.mkdir(exist_ok=True, parents=True)

    (dest_dir/".gitignore").write_text("*")
    
    for i, file_ in tqdm(enumerate(valid_files, 1), 
            desc="Saving files", total=len(valid_files)):
        
        dest = dest_dir / f"{i:06}.jpg"
        copy(file_, dest)


def _write_logfile(logfile: str, invalid_files: list) -> None:
    p = Path(logfile)
    if p.parents:
        p.parents[0].mkdir(exist_ok=True)
    with p.open("w") as fh:
        for name, errorcode in sorted(invalid_files):
            fh.write(f"{name};{errorcode}\n")


def _filter_dataset(input_dir) -> tuple:
    ds_folder  = Path(input_dir)
    hashes     = set()
    file_paths = sorted(ds_folder.rglob("*"))
    valid_files, invalid_files = [], []
    
    for path in tqdm(file_paths, desc="Processing files", total=len(file_paths)):
        if path.is_dir(): continue
       
        try:
            _check_path(path)
            as_array = _check_image(path)
            h = hash(bytes(as_array))
            if h in hashes: 
                raise InvalidFileException(6)
            hashes.add(h)
            valid_files.append(path)
        except InvalidFileException as e:
            invalid_files.append((str(path.relative_to(input_dir)), e))
            
    return valid_files, invalid_files


def _check_path(path):
    if path.suffix not in _VALID_SUFFIXES:
        raise InvalidFileException(1)
    elif path.stat().st_size <= 10_000:
        raise InvalidFileException(2)


def _check_image(path: Path):
    try:
        img_arr = np.array(Image.open(path))
    except:
        raise InvalidFileException(3)
    if np.var(img_arr) == 0: 
        raise InvalidFileException(4)
    elif (len(img_arr.shape) != 2 or 
            img_arr.shape[0] < 100 or img_arr.shape[1] < 100):
        raise InvalidFileException(5)
    else:
        return img_arr

class InvalidFileException(Exception): 
    """Exception to raise when file does not meet required criteria.

    Raise with the number of given criterium it does not meet:
    >>>
    1. The file name ends with _.jpg, .JPG, .JPEG, or .jpeg_.
    2.  The file size is larger than 10 kB. 
    3. The file can be read as image.
    4. The image data does have variance $> 0$, i.e. there is not just 1  value in the image data.
    5.  The image data has shape $(H, W)$, with H and W larger or equal to 100 pixels.
    6.  The same image data has not been copied already.
    """
    pass

def main():
    """Predefined code to seperate data into train-, eval- and testset.
    Needs to be in the right directory. 

    Part 1-4 will be in the training set. Part 5 will be the evaluation set."""
    train_path = Path('data/interrim/train')
    for i in range(5):
        filter_dataset(Path(f'data/raw/dataset_part_{str(i)}'), train_path / str(i), Path(f'reports/logfile_part_{i}'))
    filter_dataset(Path('data/raw/dataset_part_5'), Path('data/interrim/eval'), Path(f'reports/logfile_part_5'))


if __name__ == "__main__":
    main()