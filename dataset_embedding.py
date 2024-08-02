import torch as T
from omegaconf import DictConfig
import hydra
import os

import tqdm
import timm

from torch.utils import data


def collate_fn(batch):
    return T.stack([b[0] for b in batch])


@hydra.main(config_path="configs/hydra", config_name="embedding-config", version_base=None)
def run(conf: DictConfig):
    model_name = conf.encoder
    
    if not os.path.exists(f"./dataset_embeddings/{model_name}"):
        print(f"create directory: ./dataset_embeddings/{model_name}")
        os.makedirs(f"./dataset_embeddings/{model_name}")
    
    # --------------- model setup ----------------
    print("initialize model ...")
    
    if f"{model_name}.pth" in os.listdir("./timm_models"):
        encoder = T.load(f"./timm_models/{model_name}.pth")
    else:
        encoder = timm.create_model(model_name, pretrained=True, features_only=True)
        T.save(encoder, f"./timm_models/{model_name}.pth")
    
    encoder = encoder.eval().to("cuda")
    
    # --------------- data setup ----------------
    print("initialize dataset ...")
    
    from motion_capture.data.datasets import COCO2017PersonKeypointsDataset, COCOPanopticsObjectDetection, HAKELarge, CelebA
    from motion_capture.data.datasets import CombinedDataset
    
    coco_dataset = COCOPanopticsObjectDetection(
        image_folder_path = "//192.168.2.206/data/datasets/COCO2017/images",
        panoptics_path = "//192.168.2.206/data/datasets/COCO2017/panoptic_annotations_trainval2017/annotations",
        image_shape_WH=conf.image_shape,
        max_number_of_instances=100
    ) # 120k images
    
    person_keypoints_dataset = COCO2017PersonKeypointsDataset(
        image_folder_path = "//192.168.2.206/data/datasets/COCO2017/images",
        annotation_folder_path = "//192.168.2.206/data/datasets/COCO2017/annotations",
        image_shape_WH = conf.image_shape,
        min_person_bbox_size = 100
    ) # 70k images
    
    hake_dataset = HAKELarge(
        annotation_path = "\\\\192.168.2.206\\data\\datasets\\HAKE\\Annotations",
        image_path = "\\\\192.168.2.206\\data\\datasets\\HAKE-large",
        image_shape_WH = conf.image_shape,
    ) # 100k images
    
    celeba_dataset = CelebA(
        annotatin_path="\\\\192.168.2.206\\data\\datasets\\CelebA\\Anno",
        image_path="\\\\192.168.2.206\\data\\datasets\\CelebA\\img\\img_align_celeba\\img_celeba",
        image_shape_WH = conf.image_shape
    )
    
    dataloader = data.DataLoader(
        dataset = CombinedDataset([
            coco_dataset,
            person_keypoints_dataset,
            hake_dataset,
            celeba_dataset
        ]),
        batch_size = conf.batch_size,
        num_workers = conf.num_workers,
        collate_fn = collate_fn, 
        shuffle = False
    )
    
    print(f"embedding (dataset size: {len(dataloader)}) ...")
    
    with T.no_grad():
        dataloader_iter = iter(dataloader)
        
        for e in tqdm.tqdm(range(len(dataloader)), total=len(dataloader)):
            
            try:
                batch = next(dataloader_iter)
                images = batch.to("cuda")
                student_embedding = encoder(images)[-1]
                T.save(student_embedding, f"./dataset_embeddings/{model_name}/{e}.pth")
                
            except Exception as e:
                print(e)
                continue
            
    # print("formatting embedding ...")
    
    # pth = "./dataset_embeddings/{model_name}"
    # num_encoding_blocks = len(os.listdir(pth))

    # embeddings = T.zeros(num_encoding_blocks * 250, 320, 7, 7)
    # valid = T.zeros(num_encoding_blocks * 250, dtype=T.bool)

    # print(f"number of encodings: {len(os.listdir(pth))}")
    # for (i, f) in tqdm.tqdm(enumerate(os.listdir(pth)), total=num_encoding_blocks, desc="Loading encodings"):
    #     ten = T.load(os.path.join(pth, f), map_location="cpu")
    #     block_size = ten.size(0)
    #     embeddings[(i * 250):(i * 250 + block_size), ...] = ten[:]
    #     valid[(i * 250):(i * 250 + block_size)] = True

    # T.save(embeddings, f"./dataset_embeddings/{model_name}-embeddings.pt")
    # T.save(valid,      f"./dataset_embeddings/{model_name}-valid.pt")

if __name__ == "__main__":
    run()
