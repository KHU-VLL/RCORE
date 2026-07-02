import torch
import torch.nn.functional as F
import numpy as np


def apply_CPR(x, aug_prob, alpha, lambda_mult, labels, OW_pair2idx,
              fame_operator=None, p_v_o=None, index_select="random", seen_mask=None,
              only_aug=False, only_label=False):
    batch_verb, batch_obj, batch_target = labels

    # convert CW comp label to OW label
    obj_indices = torch.argmax(batch_obj, dim=1)  # random_obj_labels is one-hot
    original_ow_pair_indices = torch.tensor(
        [OW_pair2idx[(v.item(), o.item())] for v, o in zip(batch_verb, obj_indices)],
        device=x.device, dtype=torch.long,
    )

    # one-hot
    num_ow_pairs = len(OW_pair2idx)
    OW_batch_target = torch.nn.functional.one_hot(original_ow_pair_indices, num_classes=num_ow_pairs).to(dtype=x.dtype)

    ####### center frame mixing
    B, T = x.shape[:2]

    # If utilize FAME, get motion region first
    if fame_operator is not None:
        fame_bbox_per_frame, fame_mask_per_frame = fame_operator(x, return_bbox=True)  # B, T, 4

        # --- Temporal Smoothing for BBox (Added) ---
        k_size = 5
        padding = k_size // 2
        # (B, T, 4) -> (B, 4, T)
        bbox_transposed = fame_bbox_per_frame.float().permute(0, 2, 1)

        # Apply smoothing
        smoothed_bbox = F.avg_pool1d(
            bbox_transposed,
            kernel_size=k_size,
            stride=1,
            padding=padding,
            count_include_pad=False,
        )

        # (B, 4, T) -> (B, T, 4)
        fame_bbox_per_frame = smoothed_bbox.permute(0, 2, 1)
        # ---------------------------------------------

        # --- Temporal Smoothing for Mask (Added) ---
        H, W = fame_mask_per_frame.shape[2:]
        mask_flat = fame_mask_per_frame.view(B, T, -1).permute(0, 2, 1)  # (B, H*W, T)

        smoothed_mask_flat = F.avg_pool1d(
            mask_flat,
            kernel_size=k_size,
            stride=1,
            padding=padding,
            count_include_pad=False,
        )

        # (B, H*W, T) -> (B, T, H, W)
        fame_mask_per_frame = smoothed_mask_flat.permute(0, 2, 1).view(B, T, H, W)
        # ---------------------------------------------

    # Apply augmentation with aug_prob probability
    mask = torch.rand(B, device=x.device) < aug_prob
    new_seen_mask = seen_mask.clone() if seen_mask is not None else None

    if not mask.any():
        labels = (batch_verb, batch_obj, batch_target, batch_target, new_seen_mask)
        return x, labels

    random_indices = torch.randperm(B, device=x.device)
    if B > 1:
        shift = torch.randint(1, B, (1,), device=x.device).item()
    else:
        shift = 0
    random_indices = (torch.arange(B, device=x.device) + shift) % B

    random_center_frames = x.clone()[random_indices, T // 2]

    lam = torch.tensor(np.random.beta(alpha, alpha, size=B), device=x.device, dtype=x.dtype)
    lam *= lambda_mult

    lam_img = lam.view(B, *([1] * (x.dim() - 1)))
    lam_label = lam.unsqueeze(-1)

    cpr_x = x.clone()
    if only_label:
        pass
    else:
        if fame_operator is None:
            cpr_x[mask] = cpr_x[mask] * (1.0 - lam_img[mask]) + random_center_frames[mask].unsqueeze(1) * lam_img[mask]
        else:
            masked_indices = torch.where(mask)[0]

            for b_idx in masked_indices:
                random_b_idx = random_indices[b_idx]
                l = lam[b_idx]

                src_frame = x[random_b_idx, T // 2]
                src_mask = fame_mask_per_frame[random_b_idx, T // 2].unsqueeze(0)

                bbox_to_paste = fame_bbox_per_frame[random_b_idx, T // 2].long()
                xmin_s, ymin_s, xmax_s, ymax_s = bbox_to_paste

                # (C+1, H_s, W_s)
                src_combined = torch.cat([src_frame, src_mask], dim=0)[:, ymin_s:ymax_s, xmin_s:xmax_s]

                for t in range(T):
                    bbox = fame_bbox_per_frame[b_idx, t].long()
                    xmin, ymin, xmax, ymax = bbox
                    bh, bw = ymax - ymin, xmax - xmin

                    if bh > 0 and bw > 0:
                        resized_combined = F.interpolate(
                            src_combined.unsqueeze(0),
                            size=(bh, bw),
                            mode='bilinear',
                            align_corners=False,
                        ).squeeze(0)

                        resized_rcf = resized_combined[:-1]
                        resized_mask = (resized_combined[-1:] > 0.5).float()

                        target_area = cpr_x[b_idx, t, :, ymin:ymax, xmin:xmax]
                        mix_w = l * resized_mask
                        cpr_x[b_idx, t, :, ymin:ymax, xmin:xmax] = target_area + mix_w * (resized_rcf - target_area)

    random_obj_labels = batch_obj[random_indices]
    cpr_batch_obj = batch_obj.clone()
    cpr_batch_obj[mask] = cpr_batch_obj[mask] * (1.0 - lam_label[mask]) + random_obj_labels[mask] * lam_label[mask]

    # clamp to prevent [0, 1, 0] + [0, 1, 0] = [0, 2, 0]
    cpr_batch_obj = torch.clamp(cpr_batch_obj, max=1.0)

    random_obj_indices = torch.argmax(random_obj_labels, dim=1)  # random_obj_labels is one-hot
    new_pair_indices = torch.tensor(
        [OW_pair2idx[(v.item(), o.item())] for v, o in zip(batch_verb, random_obj_indices)],
        device=x.device, dtype=torch.long,
    )

    if seen_mask is not None:
        new_seen_mask[new_pair_indices] = True

        seen_indices = torch.where(seen_mask)[0].to(x.device)
        new_cw_label_indices = torch.unique(torch.cat([seen_indices, new_pair_indices])).sort()[0]

        target_orig_local = torch.searchsorted(new_cw_label_indices, original_ow_pair_indices)
        target_new_local = torch.searchsorted(new_cw_label_indices, new_pair_indices)

        num_classes, B = new_cw_label_indices.size(0), batch_target.shape[0]
        cpr_batch_target_OW = torch.zeros((B, num_classes), device=batch_target.device)

        for b in range(B):
            if only_aug:
                cpr_batch_target_OW[b, target_orig_local[b]] = 1.0
            else:
                if mask[b]:
                    cpr_batch_target_OW[b, target_orig_local[b]] = (1.0 - lam[b])
                    cpr_batch_target_OW[b, target_new_local[b]] += lam[b]  # 중복 인덱스 대비 +=
                else:
                    cpr_batch_target_OW[b, target_orig_local[b]] = 1.0
    else:
        print("seen mask is None")
        cpr_batch_target_OW = batch_target.clone()

    cpr_labels = (batch_verb, cpr_batch_obj, batch_target, cpr_batch_target_OW, new_seen_mask)
    return cpr_x, cpr_labels
