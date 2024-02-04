import os

import open_clip
import torch
from tqdm.auto import tqdm

from clip_finetune.templates import IWildCam_prompts, FMOW_prompts
import torch.nn.functional as F

@torch.no_grad()
def init_weights(prompts_dict, tokenizer, clip_model, device, ):
    text_embeddings = []
    clip_model.eval().to(device)
    for texts in prompts_dict.values():
        texts = tokenizer(texts).to(device)  # tokenize
        embeddings = clip_model.encode_text(texts, normalize=True)  # embed with text encoder
        mean_embedding = embeddings.mean(dim=0)  # mean of spherical embeddings
        text_embeddings.append(mean_embedding)

    text_embeddings = torch.stack(text_embeddings, )
    text_embeddings = F.normalize(text_embeddings, dim=-1)
    return text_embeddings.cpu()


if __name__ == '__main__':
    DATASET = {
    'IWildCam': (['train'], IWildCam_prompts),
    'IWildCamID': (['eval'], IWildCam_prompts),
    'IWildCamOOD': (['eval'], IWildCam_prompts),
    'IWildCamIDVal': (['eval'], IWildCam_prompts),
    'IWildCamOODVal': (['eval'], IWildCam_prompts),
    }
    DATA_SEEDS = [0]
    MODELS = [
            ('ViT-B-16', 'openai'),
            # ('ViT-L-14', 'laion2b_s32b_b82k'),
            # ('ViT-L-14-336', 'openai'),
            # ('ViT-H-14', 'laion2b_s32b_b79k'),
            # ('ViT-g-14', 'laion2b_s12b_b42k'),
            # ('ViT-bigG-14', 'laion2b_s39b_b160k'),
            # ('coca_ViT-L-14', 'laion2B-s13B-b90k')
            ]
    device = 'cuda:3'
    mixed_precision = 'bf16'

    for model_name, pretrain_ds in tqdm(MODELS):
        print('Model: ', model_name, pretrain_ds)
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrain_ds)
        model.eval().to(device)
        logit_scale = model.logit_scale.cpu()
        for dataset, val in tqdm(DATASET.items()):
            print('Dataset: ', dataset)
            splits, env_prompts = val
            tokenizer = open_clip.get_tokenizer(model_name)
            for split in splits:
                print('Split: ', split)
                for data_seed in DATA_SEEDS:
                    print('Data seed: ', data_seed)
                    save_dir = f'/tmp/clip_finetune/{model_name}-{pretrain_ds}/{dataset}:{split}/data_seed-{data_seed}/'
                    image_features = torch.load(os.path.join(save_dir, 'features.pt'), map_location='cpu')
                    image_features = F.normalize(image_features, dim=-1).detach().float().cpu()
                    # W_e shape = (num_envs, 512). Alreay normalized
                    W_e = init_weights(env_prompts, tokenizer, model, device)
                    W_e = W_e.detach().cpu().float()
                    assert W_e.shape[0] == len(env_prompts), f'W_e shape: {W_e.shape}, env_prompts: {len(env_prompts)}'
                    logits = logit_scale*image_features@W_e.T
                    probs = logits.softmax(dim=-1)
                    preds = torch.argmax(probs, dim=-1)
                    torch.save(probs, os.path.join(save_dir, 'zeroshot_env_probs.pt'))
                    torch.save(preds, os.path.join(save_dir, 'zeroshot_env_preds.pt'))
                    torch.save(W_e, os.path.join(save_dir, 'zeroshot_env_clf.pt'))
