import pandas as pd
import torch
import torch.nn.functional as F


def get_output_size(img, locations, roi_size):
    shapes = locations.max(2)[0]
    output_size =  torch.zeros(5)
    s = torch.unique(shapes, dim=0).squeeze()
    s = [s[i] + roi_size[i] for i in range(len(s))]
    output_size[0] = shapes.shape[0]
    output_size[1] = img.shape[1]
    output_size[2:] =  torch.tensor(s)
    return output_size.to(torch.int)


def reconstruct(img, locations, out_size, crop_size):
    reconstructed_img = torch.zeros(tuple(out_size), device=img.device)
    reshape =  list([locations.shape[0], locations.shape[2]]) + list(img.shape[1:])
    image = img.reshape(reshape)
    for i in range(out_size[0]):
        for j in range(locations.shape[2]):
            reconstructed_img[i][:, 
            locations[i][0][j]:locations[i][0][j]+crop_size[0],
            locations[i][1][j]:locations[i][1][j]+crop_size[1],
            locations[i][2][j]:locations[i][2][j]+crop_size[2]] = image[i][j,:]
    return reconstructed_img


def simple_nms(scores, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby points """
    assert(nms_radius >= 0)

    def max_pool(x):
        return F.max_pool3d(
            x,
            kernel_size=nms_radius*2+1, 
            stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    return torch.where(max_mask, scores, zeros)


def post_process_pipeline(cfg, net_output):
    img = net_output["logits"].detach()
    device = img.device
    
    locations = net_output['location'].cpu()
    scales = net_output["scale"]
    
    tomo_ids = [net_output["id"][i] for i in range(0, len(net_output["id"]), locations.shape[2])]
    dims = [net_output["dims"][i] for i in range(0, len(net_output["dims"]), locations.shape[2])]
    
    img = F.interpolate(
        img, size=(cfg.roi_size[0],cfg.roi_size[1],cfg.roi_size[2]),
        mode='trilinear', align_corners=False)

    out_size = get_output_size(img, locations, cfg.roi_size)

    rec_img = reconstruct(img, locations, out_size=out_size, crop_size=cfg.roi_size)
    
    s = rec_img.shape[-3:]
    rec_img = F.interpolate(
        rec_img, size=(s[0]//2, s[1]//2, s[2]//2), 
        mode='trilinear', align_corners=False)

    preds = rec_img.softmax(1)
    
    p1 = preds[:, 0, :][None,]
    y = simple_nms(p1, nms_radius=cfg.nms_radius)
    kps = torch.where(y.squeeze()>0)
    bzyx = torch.stack(kps,-1)
    conf = y.squeeze()[kps]

    delta = ((torch.tensor(s) - torch.tensor(cfg.new_size))//2).to(device)
    dims = torch.tensor(dims, device=device, dtype=torch.int)
    b = bzyx[:, 0].to(torch.long)
    zyx = bzyx[:, 1:]
    zyx = (((zyx * 2) - delta) / scales[b]).round().to(torch.int)  # delta to remove padding added during transforms
    z, y, x = zyx[:, 0], zyx[:, 1], zyx[:, 2]
    ids = [tomo_ids[int(bb)] for bb in b]
    conf = conf.to(torch.float32)

    dims_tensor = dims[b]
    in_bounds = ((z > 0) & (z < dims_tensor[:, 0]) &
                 (y > 0) & (y < dims_tensor[:, 1]) &
                 (x > 0) & (x < dims_tensor[:, 2]) &
                 (conf > 0.01))
    
    z = z[in_bounds]
    y = y[in_bounds]
    x = x[in_bounds]
    conf = conf[in_bounds]
    ids = [tomo_ids[i] for i in in_bounds.nonzero(as_tuple=True)[0]]
    empty_tomos = [tid for tid in tomo_ids if tid not in set(ids)]
    
    if len(empty_tomos) > 0:
        dummy_coords = torch.full((len(empty_tomos),), -1, dtype=torch.int, device=device)
        dummy_conf = torch.zeros(len(empty_tomos), dtype=torch.float32, device=device)
        z = torch.cat([z, dummy_coords])
        y = torch.cat([y, dummy_coords])
        x = torch.cat([x, dummy_coords])
        conf = torch.cat([conf, dummy_conf])
        ids.extend(empty_tomos)
    ids = torch.tensor(ids, device=device)
    output =  torch.stack([z, y, x, conf, ids], dim=1)
    return output
