import os
import numpy as np
from scipy import linalg
from scipy.ndimage import uniform_filter1d
import torch
from utils.motion_process import recover_from_ric
from utils.back_process import back_process
from tqdm import tqdm

#################################################################################
#                               Eval Function Loops                             #
#################################################################################
@torch.no_grad()
def evaluation_ae(out_dir, val_loader, net, writer, ep, eval_wrapper, device, best_fid=1000, best_div=0,
                  best_top1=0, best_top2=0, best_top3=0, best_matching=100,
                  eval_mean=None, eval_std=None, save=True, draw=True):
    net.eval()

    motion_annotation_list = []
    motion_pred_list = []

    R_precision_real = 0
    R_precision = 0

    nb_sample = 0
    matching_score_real = 0
    matching_score_pred = 0
    mpjpe = 0
    num_poses = 0

    for batch in tqdm(val_loader):
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token = batch

        bs, seq = motion.shape[0], motion.shape[1]
        gt = val_loader.dataset.inv_transform(motion.detach().cpu().numpy())
        bgt = []
        for j in range(bs):
            bgt.append(back_process(gt[j], is_mesh=False))
        bgt = np.stack(bgt, axis=0)
        bgt = val_loader.dataset.transform(bgt, eval_mean, eval_std)

        bgt = torch.from_numpy(bgt).to(device)
        (et, em), (_, _) = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, caption, bgt, m_length-1)

        motion = motion.float().to(device)
        with torch.no_grad():
            pred_pose_eval = net.forward(motion)
        pred = val_loader.dataset.inv_transform(pred_pose_eval.detach().cpu().numpy())
        bpred = []
        for j in range(bs):
            bpred.append(back_process(pred[j], is_mesh=False))
        bpred = np.stack(bpred, axis=0)
        bpred = val_loader.dataset.transform(bpred, eval_mean, eval_std)

        (et_pred, em_pred), (_, _) = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, caption,
                                                          torch.from_numpy(bpred).to(device), m_length-1)
        for i in range(bs):
            gtt = torch.from_numpy(gt[i, :m_length[i]]).float().reshape(-1, 22, 3)
            predd = torch.from_numpy(pred[i, :m_length[i]]).float().reshape(-1, 22, 3)
            mpjpe += torch.sum(calculate_mpjpe(gtt, predd))
            num_poses += gt.shape[0]

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample
    mpjpe = mpjpe / num_poses

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = "--> \t Eva. Re %d:, FID. %.4f, Diversity Real. %.4f, Diversity. %.4f, R_precision_real. (%.4f, %.4f, %.4f), R_precision. (%.4f, %.4f, %.4f), matching_real. %.4f, matching_pred. %.4f, MPJPE. %.4f" % \
          (ep, fid, diversity_real, diversity, R_precision_real[0], R_precision_real[1], R_precision_real[2],
           R_precision[0], R_precision[1], R_precision[2], matching_score_real, matching_score_pred, mpjpe)
    print(msg)
    if draw:
        writer.add_scalar('./Test/FID', fid, ep)
        writer.add_scalar('./Test/Diversity', diversity, ep)
        writer.add_scalar('./Test/top1', R_precision[0], ep)
        writer.add_scalar('./Test/top2', R_precision[1], ep)
        writer.add_scalar('./Test/top3', R_precision[2], ep)
        writer.add_scalar('./Test/matching_score', matching_score_pred, ep)

    if fid < best_fid:
        msg = "--> --> \t FID Improved from %.5f to %.5f !!!" % (best_fid, fid)
        if draw: print(msg)
        best_fid = fid
        if save:
            torch.save({'ae': net.state_dict(), 'ep': ep}, os.path.join(out_dir, 'net_best_fid.tar'))

    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = "--> --> \t Diversity Improved from %.5f to %.5f !!!"%(best_div, diversity)
        if draw: print(msg)
        best_div = diversity

    if R_precision[0] > best_top1:
        msg = "--> --> \t Top1 Improved from %.5f to %.5f !!!" % (best_top1, R_precision[0])
        if draw: print(msg)
        best_top1 = R_precision[0]

    if R_precision[1] > best_top2:
        msg = "--> --> \t Top2 Improved from %.5f to %.5f!!!" % (best_top2, R_precision[1])
        if draw: print(msg)
        best_top2 = R_precision[1]

    if R_precision[2] > best_top3:
        msg = "--> --> \t Top3 Improved from %.5f to %.5f !!!" % (best_top3, R_precision[2])
        if draw: print(msg)
        best_top3 = R_precision[2]

    if matching_score_pred < best_matching:
        msg = f"--> --> \t matching_score Improved from %.5f to %.5f !!!" % (best_matching, matching_score_pred)
        if draw: print(msg)
        best_matching = matching_score_pred

    net.train()
    return best_fid, best_div, best_top1, best_top2, best_top3, best_matching, mpjpe, writer

@torch.no_grad()
def evaluation_acmdm(out_dir, val_loader, ema_acmdm, ae, writer, ep, best_fid, best_div,
                        best_top1, best_top2, best_top3, best_matching, eval_wrapper, device, clip_score_old,
                        cond_scale=None, cal_mm=False, eval_mean=None, eval_std=None, after_mean=None, after_std=None, mesh_mean=None, mesh_std=None,
                        draw=True,
                        is_raw=False,
                        is_prefix=False,
                        is_control=False, index=[0], intensity=100,
                        is_mesh=False):

    ema_acmdm.eval()
    if not is_raw:
        ae.eval()

    save=False

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    multimodality = 0
    if cond_scale is None:
        if "kit" in out_dir:
            cond_scale = 2.5
        else:
            cond_scale = 2.5
    clip_score_real = 0
    clip_score_gt = 0
    skate_ratio_sum = 0
    dist_sum = 0
    traj_err = []

    nb_sample = 0
    if cal_mm:
        num_mm_batch = 3
    else:
        num_mm_batch = 0

    for i, batch in enumerate(tqdm(val_loader)):
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token = batch
        m_length = m_length.to(device)

        bs, seq = pose.shape[:2]
        if i < num_mm_batch:
            motion_multimodality_batch = []
            batch_clip_score_pred = 0
            for _ in tqdm(range(30)):
                pred_latents = ema_acmdm.generate(clip_text, m_length//4 if not is_raw else m_length, cond_scale)

                if not is_raw:
                    pred_latents = val_loader.dataset.inv_transform(pred_latents.permute(0, 2, 3, 1).detach().cpu().numpy(),
                                                                    after_mean, after_std)
                    pred_latents = torch.from_numpy(pred_latents).to(device)
                    with torch.no_grad():
                        pred_motions = ae.decode(pred_latents.permute(0,3,1,2))
                else:
                    pred_motions = pred_latents.permute(0, 2, 3, 1)
                pred_motions = val_loader.dataset.inv_transform(pred_motions.detach().cpu().numpy())
                pred_motionss = []
                for j in range(bs):
                    pred_motionss.append(back_process(pred_motions[j], is_mesh=is_mesh))
                pred_motionss = np.stack(pred_motionss, axis=0)
                pred_motions = val_loader.dataset.transform(pred_motionss, eval_mean, eval_std)
                (et_pred, em_pred), (et_pred_clip, em_pred_clip) = eval_wrapper.get_co_embeddings(word_embeddings,
                                                                                                  pos_one_hots,
                                                                                                  sent_len,
                                                                                                  clip_text,
                                                                                                  torch.from_numpy(
                                                                                                      pred_motions).to(
                                                                                                      device),
                                                                                                  m_length - 1)
                motion_multimodality_batch.append(em_pred.unsqueeze(1))
            motion_multimodality_batch = torch.cat(motion_multimodality_batch, dim=1) #(bs, 30, d)
            motion_multimodality.append(motion_multimodality_batch)
            for j in range(bs):
                single_em = em_pred_clip[j]
                single_et = et_pred_clip[j]
                clip_score = (single_em @ single_et.T).item()
                batch_clip_score_pred += clip_score
            clip_score_real += batch_clip_score_pred
        else:
            if is_control:
                pred_latents, mask_hint = ema_acmdm.generate_control(clip_text, m_length//4, pose.clone().float().to(device).permute(0,3,1,2), index, intensity,
                                                                     cond_scale)
                mask_hint = mask_hint.permute(0, 2, 3, 1).cpu().numpy()
            elif is_prefix:
                motion = pose.clone().float().to(device)
                with torch.no_grad():
                    motion = ae.encode(motion)
                amean = torch.from_numpy(after_mean).to(device)
                astd = torch.from_numpy(after_std).to(device)
                motion = ((motion.permute(0,2,3,1)-amean)/astd).permute(0,3,1,2)
                pred_latents = ema_acmdm.generate(clip_text, m_length // 4, cond_scale, motion[:, :, :5, :]) # 20+40 style
            else:
                pred_latents = ema_acmdm.generate(clip_text, m_length//4 if not (is_raw or is_mesh) else m_length, cond_scale, j=22 if not is_mesh else 28)
            if not is_raw:
                pred_latents = val_loader.dataset.inv_transform(pred_latents.permute(0,2,3,1).detach().cpu().numpy(), after_mean, after_std)
                pred_latents = torch.from_numpy(pred_latents).to(device)
                with torch.no_grad():
                    if not is_mesh:
                        pred_latents = pred_latents.permute(0, 3, 1, 2)
                    pred_motions = ae.decode(pred_latents)
            else:
                pred_motions = pred_latents.permute(0, 2, 3, 1)
            if not is_mesh:
                pred_motions = val_loader.dataset.inv_transform(pred_motions.detach().cpu().numpy())
            else:
                pred_motions = val_loader.dataset.inv_transform(pred_motions.detach().cpu().numpy(), mesh_mean, mesh_std)
                J_regressor = np.load('body_models/J_regressor.npy')
                pred_motions = np.einsum('jk,btkc->btjc', J_regressor, pred_motions)[:, :, :22]
            if is_control:
                # foot skate
                skate_ratio, skate_vel = calculate_skating_ratio(torch.from_numpy(pred_motions.transpose(0,2,3,1)))  # [batch_size]
                skate_ratio_sum += skate_ratio.sum()
                # control errors
                hint = val_loader.dataset.inv_transform(pose.clone().detach().cpu().numpy())
                hint = hint * mask_hint
                for i, (mot, h, mask) in enumerate(zip(pred_motions, hint, mask_hint)):
                    control_error = control_l2(np.expand_dims(mot, axis=0), np.expand_dims(h, axis=0),
                                               np.expand_dims(mask, axis=0))
                    mean_error = control_error.sum() / mask.sum()
                    dist_sum += mean_error
                    control_error = control_error.reshape(-1)
                    mask = mask.reshape(-1)
                    err_np = calculate_trajectory_error(control_error, mean_error, mask)
                    traj_err.append(err_np)

            pred_motionss = []
            for j in range(bs):
                pred_motionss.append(back_process(pred_motions[j], is_mesh=is_mesh))
            pred_motionss = np.stack(pred_motionss, axis=0)
            pred_motions = val_loader.dataset.transform(pred_motionss, eval_mean, eval_std)
            (et_pred, em_pred), (et_pred_clip, em_pred_clip) = eval_wrapper.get_co_embeddings(word_embeddings,
                                                                              pos_one_hots, sent_len,
                                                                              clip_text,
                                                                              torch.from_numpy(pred_motions).to(device),
                                                                              m_length-1)
            batch_clip_score_pred = 0
            for j in range(bs):
                single_em = em_pred_clip[j]
                single_et = et_pred_clip[j]
                clip_score = (single_em @ single_et.T).item()
                batch_clip_score_pred += clip_score
            clip_score_real += batch_clip_score_pred

        pose = val_loader.dataset.inv_transform(pose.detach().cpu().numpy())
        poses = []
        for j in range(bs):
            poses.append(back_process(pose[j], is_mesh=False))
        poses = np.stack(poses, axis=0)
        pose = val_loader.dataset.transform(poses, eval_mean, eval_std)
        pose = torch.from_numpy(pose).cuda().float()
        pose = pose.cuda().float()
        (et, em), (et_clip, em_clip) = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, clip_text,
                                                          pose.clone(), m_length-1)
        batch_clip_score = 0
        for j in range(bs):
            single_em = em_clip[j]
            single_et = et_clip[j]
            clip_score = (single_em @ single_et.T).item()
            batch_clip_score += clip_score
        clip_score_gt += batch_clip_score
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    clip_score_real = clip_score_real / nb_sample
    clip_score_gt = clip_score_gt / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample
    if is_control:
        # l2 dist
        dist_mean = dist_sum / nb_sample

        # Skating evaluation
        skating_score = skate_ratio_sum / nb_sample

        ### For trajecotry evaluation from GMD ###
        traj_err = np.stack(traj_err).mean(0)

    if cal_mm:
        motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        multimodality = calculate_multimodality(motion_multimodality, 10)

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = (f"--> \t Eva. Ep/Re {ep} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity."
           f" {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision},"
           f" matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
           f" multimodality. {multimodality:.4f}, clip score. {clip_score_real}"
        + (f" foot skating. {skating_score:.4f}, traj error. {traj_err[1].item():.4f}, pos error. {traj_err[3].item():.4f}, avg error. {traj_err[4].item():.4f}"
           if is_control else ""))
    print(msg)

    if draw:
        writer.add_scalar('./Test/FID', fid, ep)
        writer.add_scalar('./Test/Diversity', diversity, ep)
        writer.add_scalar('./Test/top1', R_precision[0], ep)
        writer.add_scalar('./Test/top2', R_precision[1], ep)
        writer.add_scalar('./Test/top3', R_precision[2], ep)
        writer.add_scalar('./Test/matching_score', matching_score_pred, ep)
        writer.add_scalar('./Test/clip_score', clip_score_real, ep)


    if fid < best_fid:
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        if draw: print(msg)
        best_fid, best_ep = fid, ep
        save=True


    if matching_score_pred < best_matching:
        msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
        if draw: print(msg)
        best_matching = matching_score_pred

    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        if draw: print(msg)
        best_div = diversity

    if R_precision[0] > best_top1:
        msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
        if draw: print(msg)
        best_top1 = R_precision[0]

    if R_precision[1] > best_top2:
        msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
        if draw: print(msg)
        best_top2 = R_precision[1]

    if R_precision[2] > best_top3:
        msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
        if draw: print(msg)
        best_top3 = R_precision[2]

    if clip_score_real > clip_score_old:
        msg = f"--> --> \t CLIP-score Improved from {clip_score_old:.4f} to {clip_score_real:.4f} !!!"
        if draw: print(msg)
        clip_score_old = clip_score_real

    if cal_mm:
        return best_fid, best_div, best_top1, best_top2, best_top3, best_matching, multimodality, clip_score_old, writer, save
    else:
        if is_control:
            return best_fid, best_div, best_top1, best_top2, best_top3, best_matching, 0, clip_score_old, writer, save, dist_mean, skating_score, traj_err
        else:
            return best_fid, best_div, best_top1, best_top2, best_top3, best_matching, 0, clip_score_old, writer, save, None, None, None


@torch.no_grad()
def evaluation_acmdm_another_v(out_dir, val_loader, ema_acmdm, ae, writer, ep, best_fid, best_div,
                        best_top1, best_top2, best_top3, best_matching, eval_wrapper, device, clip_score_old,
                        cond_scale=None, cal_mm=False, train_mean=None, train_std=None, after_mean=None, after_std=None,
                        draw=True,
                        is_raw=False,
                        is_prefix=False,
                        is_control=False, index=[0], intensity=100,
                        is_mesh=False):

    ema_acmdm.eval()
    if not is_raw:
        ae.eval()

    save=False

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    multimodality = 0
    if cond_scale is None:
        if "kit" in out_dir:
            cond_scale = 2.5
        else:
            cond_scale = 2.5
    clip_score_real = 0
    clip_score_gt = 0
    skate_ratio_sum = 0
    dist_sum = 0
    traj_err = []

    nb_sample = 0
    if cal_mm:
        num_mm_batch = 3
    else:
        num_mm_batch = 0

    for i, batch in enumerate(tqdm(val_loader)):
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token = batch
        m_length = m_length.to(device)

        bs, seq = pose.shape[:2]
        if i < num_mm_batch:
            motion_multimodality_batch = []
            batch_clip_score_pred = 0
            for _ in tqdm(range(30)):
                pred_latents = ema_acmdm.generate(clip_text, m_length//4 if not is_raw else m_length, cond_scale)

                if not is_raw:
                    pred_latents = val_loader.dataset.inv_transform(pred_latents.permute(0, 2, 3, 1).detach().cpu().numpy(),
                                                                    after_mean, after_std)
                    pred_latents = torch.from_numpy(pred_latents).to(device)
                    with torch.no_grad():
                        pred_motions = ae.decode(pred_latents.permute(0,3,1,2))
                else:
                    pred_motions = pred_latents.permute(0, 2, 3, 1)
                pred_motions = val_loader.dataset.inv_transform(pred_motions.detach().cpu().numpy(), train_mean, train_std)
                pred_motionss = []
                for j in range(bs):
                    pred_motionss.append(back_process(pred_motions[j], is_mesh=is_mesh))
                pred_motionss = np.stack(pred_motionss, axis=0)
                pred_motions = val_loader.dataset.transform(pred_motionss)
                (et_pred, em_pred), (et_pred_clip, em_pred_clip) = eval_wrapper.get_co_embeddings(word_embeddings,
                                                                                                  pos_one_hots,
                                                                                                  sent_len,
                                                                                                  clip_text,
                                                                                                  torch.from_numpy(
                                                                                                      pred_motions).to(
                                                                                                      device),
                                                                                                  m_length - 1)
                motion_multimodality_batch.append(em_pred.unsqueeze(1))
            motion_multimodality_batch = torch.cat(motion_multimodality_batch, dim=1) #(bs, 30, d)
            motion_multimodality.append(motion_multimodality_batch)
            for j in range(bs):
                single_em = em_pred_clip[j]
                single_et = et_pred_clip[j]
                clip_score = (single_em @ single_et.T).item()
                batch_clip_score_pred += clip_score
            clip_score_real += batch_clip_score_pred
        else:
            if is_control:
                bgt = val_loader.dataset.inv_transform(pose.clone())
                motion_gt = []
                for j in range(bs):
                    motion_gt.append(recover_from_ric(bgt[j].float(), 22).numpy())
                motion_gt = np.stack(motion_gt, axis=0)
                motion = val_loader.dataset.transform(motion_gt, train_mean, train_std)
                motion = torch.from_numpy(motion).float().to(device)
                pred_latents, mask_hint = ema_acmdm.generate_control(clip_text, m_length//4, motion.clone().permute(0,3,1,2), index, intensity,
                                                                     cond_scale)
                mask_hint = mask_hint.permute(0, 2, 3, 1).cpu().numpy()
            elif is_prefix:
                bgt = val_loader.dataset.inv_transform(pose.clone())
                motion_gt = []
                for j in range(bs):
                    motion_gt.append(recover_from_ric(bgt[j].float(), 22).numpy())
                motion_gt = np.stack(motion_gt, axis=0)
                motion = val_loader.dataset.transform(motion_gt, train_mean, train_std)
                motion = torch.from_numpy(motion).float().to(device)
                with torch.no_grad():
                    motion = ae.encode(motion)
                amean = torch.from_numpy(after_mean).to(device)
                astd = torch.from_numpy(after_std).to(device)
                motion = ((motion.permute(0,2,3,1)-amean)/astd).permute(0,3,1,2)
                pred_latents = ema_acmdm.generate(clip_text, m_length // 4, cond_scale, motion[:, :, :5, :]) # 20+40 style
            else:
                pred_latents = ema_acmdm.generate(clip_text, m_length//4 if not is_raw else m_length, cond_scale)
            if not is_raw:
                pred_latents = val_loader.dataset.inv_transform(pred_latents.permute(0,2,3,1).detach().cpu().numpy(), after_mean, after_std)
                pred_latents = torch.from_numpy(pred_latents).to(device)
                with torch.no_grad():
                    pred_motions = ae.decode(pred_latents.permute(0,3,1,2))
            else:
                pred_motions = pred_latents.permute(0, 2, 3, 1)
            pred_motions = val_loader.dataset.inv_transform(pred_motions.detach().cpu().numpy(), train_mean, train_std)
            if is_control:
                # foot skate
                skate_ratio, skate_vel = calculate_skating_ratio(torch.from_numpy(pred_motions.transpose(0,2,3,1)))  # [batch_size]
                skate_ratio_sum += skate_ratio.sum()
                # control errors
                hint = motion_gt * mask_hint
                for i, (mot, h, mask) in enumerate(zip(pred_motions, hint, mask_hint)):
                    control_error = control_l2(np.expand_dims(mot, axis=0), np.expand_dims(h, axis=0),
                                               np.expand_dims(mask, axis=0))
                    mean_error = control_error.sum() / mask.sum()
                    dist_sum += mean_error
                    control_error = control_error.reshape(-1)
                    mask = mask.reshape(-1)
                    err_np = calculate_trajectory_error(control_error, mean_error, mask)
                    traj_err.append(err_np)

            pred_motionss = []
            for j in range(bs):
                pred_motionss.append(back_process(pred_motions[j], is_mesh=is_mesh))
            pred_motionss = np.stack(pred_motionss, axis=0)
            pred_motions = val_loader.dataset.transform(pred_motionss)
            (et_pred, em_pred), (et_pred_clip, em_pred_clip) = eval_wrapper.get_co_embeddings(word_embeddings,
                                                                              pos_one_hots, sent_len,
                                                                              clip_text,
                                                                              torch.from_numpy(pred_motions).to(device),
                                                                              m_length-1)
            batch_clip_score_pred = 0
            for j in range(bs):
                single_em = em_pred_clip[j]
                single_et = et_pred_clip[j]
                clip_score = (single_em @ single_et.T).item()
                batch_clip_score_pred += clip_score
            clip_score_real += batch_clip_score_pred

        pose = pose.cuda().float()
        (et, em), (et_clip, em_clip) = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, clip_text,
                                                          pose.clone(), m_length-1)
        batch_clip_score = 0
        for j in range(bs):
            single_em = em_clip[j]
            single_et = et_clip[j]
            clip_score = (single_em @ single_et.T).item()
            batch_clip_score += clip_score
        clip_score_gt += batch_clip_score
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    clip_score_real = clip_score_real / nb_sample
    clip_score_gt = clip_score_gt / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample
    if is_control:
        # l2 dist
        dist_mean = dist_sum / nb_sample

        # Skating evaluation
        skating_score = skate_ratio_sum / nb_sample

        ### For trajecotry evaluation from GMD ###
        traj_err = np.stack(traj_err).mean(0)

    if cal_mm:
        motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        multimodality = calculate_multimodality(motion_multimodality, 10)

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = (f"--> \t Eva. Ep/Re {ep} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity."
           f" {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision},"
           f" matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
           f" multimodality. {multimodality:.4f}, clip score. {clip_score_real}"
        + (f" foot skating. {skating_score:.4f}, traj error. {traj_err[1].item():.4f}, loc error. {traj_err[3].item():.4f}, avg error. {traj_err[4].item():.4f}"
           if is_control else ""))
    print(msg)

    if draw:
        writer.add_scalar('./Test/FID', fid, ep)
        writer.add_scalar('./Test/Diversity', diversity, ep)
        writer.add_scalar('./Test/top1', R_precision[0], ep)
        writer.add_scalar('./Test/top2', R_precision[1], ep)
        writer.add_scalar('./Test/top3', R_precision[2], ep)
        writer.add_scalar('./Test/matching_score', matching_score_pred, ep)
        writer.add_scalar('./Test/clip_score', clip_score_real, ep)


    if fid < best_fid:
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        if draw: print(msg)
        best_fid, best_ep = fid, ep
        save=True


    if matching_score_pred < best_matching:
        msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
        if draw: print(msg)
        best_matching = matching_score_pred

    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        if draw: print(msg)
        best_div = diversity

    if R_precision[0] > best_top1:
        msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
        if draw: print(msg)
        best_top1 = R_precision[0]

    if R_precision[1] > best_top2:
        msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
        if draw: print(msg)
        best_top2 = R_precision[1]

    if R_precision[2] > best_top3:
        msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
        if draw: print(msg)
        best_top3 = R_precision[2]

    if clip_score_real > clip_score_old:
        msg = f"--> --> \t CLIP-score Improved from {clip_score_old:.4f} to {clip_score_real:.4f} !!!"
        if draw: print(msg)
        clip_score_old = clip_score_real

    if cal_mm:
        return best_fid, best_div, best_top1, best_top2, best_top3, best_matching, multimodality, clip_score_old, writer, save
    else:
        if is_control:
            return best_fid, best_div, best_top1, best_top2, best_top3, best_matching, 0, clip_score_old, writer, save, dist_mean, skating_score, traj_err
        else:
            return best_fid, best_div, best_top1, best_top2, best_top3, best_matching, 0, clip_score_old, writer, save, None, None, None

#################################################################################
#                                 Util Functions                                #
#################################################################################
def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

#################################################################################
#                                     Metrics                                   #
#################################################################################
def calculate_mpjpe(gt_joints, pred_joints):
    """
    gt_joints: num_poses x num_joints(22) x 3
    pred_joints: num_poses x num_joints(22) x 3
    (obtained from recover_from_ric())
    """
    assert gt_joints.shape == pred_joints.shape, f"GT shape: {gt_joints.shape}, pred shape: {pred_joints.shape}"

    # Align by root (pelvis)
    pelvis = gt_joints[:, [0]].mean(1)
    gt_joints = gt_joints - torch.unsqueeze(pelvis, dim=1)
    pelvis = pred_joints[:, [0]].mean(1)
    pred_joints = pred_joints - torch.unsqueeze(pelvis, dim=1)

    # Compute MPJPE
    mpjpe = torch.linalg.norm(pred_joints - gt_joints, dim=-1) # num_poses x num_joints=22
    mpjpe_seq = mpjpe.mean(-1) # num_poses

    return mpjpe_seq

# (X - X_train)*(X - X_train) = -2X*X_train + X*X + X_train*X_train
def euclidean_distance_matrix(matrix1, matrix2):
    """
        Params:
        -- matrix1: N1 x D
        -- matrix2: N2 x D
        Returns:
        -- dist: N1 x N2
        dist[i, j] == distance(matrix1[i], matrix2[j])
    """
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * np.dot(matrix1, matrix2.T)    # shape (num_test, num_train)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)    # shape (num_test, 1)
    d3 = np.sum(np.square(matrix2), axis=1)     # shape (num_train, )
    dists = np.sqrt(d1 + d2 + d3)  # broadcasting
    return dists

def calculate_top_k(mat, top_k):
    size = mat.shape[0]
    gt_mat = np.expand_dims(np.arange(size), 1).repeat(size, 1)
    bool_mat = (mat == gt_mat)
    correct_vec = False
    top_k_list = []
    for i in range(top_k):
#         print(correct_vec, bool_mat[:, i])
        correct_vec = (correct_vec | bool_mat[:, i])
        # print(correct_vec)
        top_k_list.append(correct_vec[:, None])
    top_k_mat = np.concatenate(top_k_list, axis=1)
    return top_k_mat


def calculate_R_precision(embedding1, embedding2, top_k, sum_all=False):
    dist_mat = euclidean_distance_matrix(embedding1, embedding2)
    argmax = np.argsort(dist_mat, axis=1)
    top_k_mat = calculate_top_k(argmax, top_k)
    if sum_all:
        return top_k_mat.sum(axis=0)
    else:
        return top_k_mat


def calculate_matching_score(embedding1, embedding2, sum_all=False):
    assert len(embedding1.shape) == 2
    assert embedding1.shape[0] == embedding2.shape[0]
    assert embedding1.shape[1] == embedding2.shape[1]

    dist = linalg.norm(embedding1 - embedding2, axis=1)
    if sum_all:
        return dist.sum(axis=0)
    else:
        return dist



def calculate_activation_statistics(activations):
    """
    Params:
    -- activation: num_samples x dim_feat
    Returns:
    -- mu: dim_feat
    -- sigma: dim_feat x dim_feat
    """
    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov


def calculate_diversity(activation, diversity_times):
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]

    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return dist.mean()


def calculate_multimodality(activation, multimodality_times):
    assert len(activation.shape) == 3
    assert activation.shape[1] > multimodality_times
    num_per_sent = activation.shape[1]

    first_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    second_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    dist = linalg.norm(activation[:, first_dices] - activation[:, second_dices], axis=2)
    return dist.mean()


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


# directly from omnicontrol
def calculate_skating_ratio(motions):
    thresh_height = 0.05  # 10
    fps = 20.0
    thresh_vel = 0.50  # 20 cm /s
    avg_window = 5  # frames

    batch_size = motions.shape[0]
    # 10 left, 11 right foot. XZ plane, y up
    # motions [bs, 22, 3, max_len]
    verts_feet = motions[:, [10, 11], :, :].detach().cpu().numpy()  # [bs, 2, 3, max_len]
    verts_feet_plane_vel = np.linalg.norm(verts_feet[:, :, [0, 2], 1:] - verts_feet[:, :, [0, 2], :-1],
                                          axis=2) * fps  # [bs, 2, max_len-1]
    # [bs, 2, max_len-1]
    vel_avg = uniform_filter1d(verts_feet_plane_vel, axis=-1, size=avg_window, mode='constant', origin=0)

    verts_feet_height = verts_feet[:, :, 1, :]  # [bs, 2, max_len]
    # If feet touch ground in agjecent frames
    feet_contact = np.logical_and((verts_feet_height[:, :, :-1] < thresh_height),
                                  (verts_feet_height[:, :, 1:] < thresh_height))  # [bs, 2, max_len - 1]
    # skate velocity
    skate_vel = feet_contact * vel_avg

    # it must both skating in the current frame
    skating = np.logical_and(feet_contact, (verts_feet_plane_vel > thresh_vel))
    # and also skate in the windows of frames
    skating = np.logical_and(skating, (vel_avg > thresh_vel))

    # Both feet slide
    skating = np.logical_or(skating[:, 0, :], skating[:, 1, :])  # [bs, max_len -1]
    skating_ratio = np.sum(skating, axis=1) / skating.shape[1]

    return skating_ratio, skate_vel

# directly from omnicontrol
def control_l2(motion, hint, hint_mask):
    # motion: b, seq, 22, 3
    # hint: b, seq, 22, 1
    loss = np.linalg.norm((motion - hint) * hint_mask, axis=-1)
    return loss

# directly from omnicontrol
def calculate_trajectory_error(dist_error, mean_err_traj, mask, strict=True):
    ''' dist_error shape [5]: error for each kps in metre
      Two threshold: 20 cm and 50 cm.
    If mean error in sequence is more then the threshold, fails
    return: traj_fail(0.2), traj_fail(0.5), all_kps_fail(0.2), all_kps_fail(0.5), all_mean_err.
        Every metrics are already averaged.
    '''
    # mean_err_traj = dist_error.mean(1)
    if strict:
        # Traj fails if any of the key frame fails
        traj_fail_02 = 1.0 - (dist_error <= 0.2).all()
        traj_fail_05 = 1.0 - (dist_error <= 0.5).all()
    else:
        # Traj fails if the mean error of all keyframes more than the threshold
        traj_fail_02 = (mean_err_traj > 0.2)
        traj_fail_05 = (mean_err_traj > 0.5)
    all_fail_02 = (dist_error > 0.2).sum() / mask.sum()
    all_fail_05 = (dist_error > 0.5).sum() / mask.sum()

    return np.array([traj_fail_02, traj_fail_05, all_fail_02, all_fail_05, dist_error.sum() / mask.sum()])
