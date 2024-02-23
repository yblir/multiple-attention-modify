import torch.utils.data as data


def self_collate(data):  # (2)
    # process_imgs, torch.tensor([video_label_int] * 8), video_path, sampled_idxs
    imgs, labels = [], []
    for d in data:
        imgs.extend(d[0])
        labels.extend(d[1])

    return imgs, labels, 0, 0


def create_base_dataloader(batch_size, dataset, num_works, split):
    """Base data loader

    Args:
        args: Dataset config args
        split (string): Load "train", "val" or "test"

    Returns:
        [dataloader]: Corresponding Dataloader
    """
    sampler = None
    if split == 'train':
        try:
            sampler = data.distributed.DistributedSampler(dataset)
        except:
            sampler = None
    else:
        sampler = None

    shuffle = True if sampler is None and split == 'train' else False
    # batch_size = getattr(args, split).batch_size
    # num_workers = args.num_workers if 'num_workers' in args else 8

    dataloader = data.DataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 sampler=sampler,
                                 num_workers=num_works,
                                 pin_memory=True,
                                 drop_last=True,
                                 collate_fn=self_collate)
    return dataloader
