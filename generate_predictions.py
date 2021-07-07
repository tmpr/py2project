import pickle as dkl
import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from datasets import ScoreSet
from functional import denormalize, disect


def generate_predictions(input_pkl_file, model_path, 
            out_path='predictions/predictions.pkl', verbose=False):
    dataset = ScoreSet(input_pkl_file)
    loader = DataLoader(dataset, shuffle=False, batch_size=1)

    model = torch.load(model_path)
    predictions = []
    for batch in loader:
        batch['input'] = batch['input'].cuda()
        batch['mean'] = batch['mean'].cuda()
        batch['std'] = batch['std'].cuda()

        out = model(batch['input'])

        reconstructed = batch['input'][0][0] + \
            batch['input'][0][1] * out.squeeze()

        denormalized = denormalize(reconstructed.squeeze().detach(),
                                   batch['mean'],
                                   batch['std'])

        denormalized = denormalized.cpu().numpy()
        denormalized = np.array(denormalized, dtype=np.uint8)

        _, _, target = disect(
            denormalized, batch['size'], batch['center'], False)

        predictions.append(target)

    with open(out_path, 'wb') as f:
        dkl.dump(predictions, f)


if __name__ == "__main__":
    generate_predictions('data/external/challenge_testset.pkl', 
                        model_path='models/final.pt', 
                        out_path='predictions/submissions.pkl')

