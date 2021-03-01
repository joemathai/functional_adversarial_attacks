import torch
import numpy as np
import logging

from functional_attacks.attacks import AffineTransforms, ColorTransforms, Delta, \
    SpatialFlowFields, ThinPlateSplines, ConvolutionalKernel
from functional_attacks.loss import CWLossF6, l2_grid_smoothness

logger = logging.getLogger(__name__)


@torch.no_grad()
def validation(classifier, x, x_adv, y):
    """
    A method to run during each iteration of the pgd_attack_with_perturbation method
    :param classifier: the model that is being attacked
    :param x: original batch of images
    :param x_adv: perturbed batch of images
    :param y: ground truth values for the original images
    :return: success-rate of the attack
    """
    y_pred = classifier(x_adv).topk(1)[1].cpu().numpy()
    y = y.cpu().numpy()
    attack_successful = [y[i] != y_pred[i] for i in range(x.shape[0])]
    linf_metric = torch.norm(x.view(x.shape[0], -1) - x_adv.view(x.shape[0], -1), p=float('inf'), dim=1).cpu().numpy()
    logger.debug(f"linf: {linf_metric}")
    logger.debug(f"pgd_attack_with_perturbation state: (idx, success, gt, linf) : "
                 f"{list(zip(list(range(y.shape[0])), attack_successful, y, linf_metric))}")
    return (100 * sum(attack_successful) / x.shape[0]).item()


def pgd_attack_linf(perturbation, classifier, examples, labels, linf_budget=(8/255.0,) * 3,
                    pixel_shift_budget=2.0, num_iterations=20, perturbation_step_size=1.0 / 255.0,
                    keep_best=True, validator=validation, l2_smoothing_loss=False,
                    l2_smooth_loss_weight=1e-4):
    """
    A method to combine global and local adversarial attacks
    :param perturbation: nn.sequential model of all the transforms to be applied
    :param classifier: Classifier network to attack
    :param examples: N x C x H X W batch
    :param labels: ground truth of the batch N x 1
    :param linf_budget: budget for Lp attack and ColorTransforms attack
    :param pixel_shift_budget: the maximum shift each pixel can take in x and y axis
                               note: for Thin Plate Splines this is applied to control points only
    :param num_iterations: max num of iterations to run the PGD based iterative attack
    :param perturbation_step_size: Step size by which to modify the parameters
    :param keep_best: save and return the perturbed images with the least loss
    :param validator: A method that takes arguments (classifier, x, x_adv, y)
                      and return a value from [0.0, 100.0] indicating the success rate of the attack
    :param l2_smoothing_loss: If true applies the lsmooth loss from the paper https://arxiv.org/pdf/1801.02612.pdf
                              to both ReColor and SpatialFlowField attacks
    :param l2_smooth_loss_weight: weight for the l2_smooth_loss
    :return:
    """
    num_examples = examples.shape[0]
    best_perturbed_examples = torch.empty(*examples.shape, dtype=examples.dtype,
                                          device=examples.device, requires_grad=False).copy_(examples)
    best_loss_per_example = [float('inf') for _ in range(num_examples)]
    loss_fn = CWLossF6(classifier)

    if validator is not None:
        validation = validator(classifier, examples, best_perturbed_examples, labels)
        logger.info("iter: %d attack-success-rate: %.3f", 0, validation)

    perturbed_examples = None
    for iter_no in range(num_iterations):
        # unravel the sequential model and separate out linf from deformation methods
        # apply linf based perturbations and clip the combined perturbations
        perturbed_examples = perturbation(examples)

        # total loss
        total_loss_per_example = torch.zeros(size=(examples.shape[0],), requires_grad=True, device=examples.device)

        # apply CW Linf loss without constraint to minimize the linf budget
        cw_f6_loss_per_example = loss_fn(perturbed_examples, labels)
        total_loss_per_example = total_loss_per_example + cw_f6_loss_per_example
        logger.info("cw_f6 iter:%d loss: %.3f", iter_no, cw_f6_loss_per_example.sum().item())

        # smoothness loss for params of ReColor and SpatialFlowField
        if l2_smoothing_loss:
            for module in perturbation.modules():
                if type(module) in (ColorTransforms, SpatialFlowFields):
                    smooth_loss_per_example = l2_grid_smoothness(
                        module.xform_params - module.identity_params) * l2_smooth_loss_weight
                    total_loss_per_example = total_loss_per_example + smooth_loss_per_example
                    logger.info("[%s] smooth loss iter:%d loss: %.3f", type(module).__name__, iter_no,
                                 smooth_loss_per_example.sum().item())

        # clear the gradients of the perturbation model using the optimizer
        perturbation.zero_grad()
        # this is not exactly needed but a good practice
        classifier.zero_grad()

        # backpropagate the loss
        total_loss_per_example.sum().backward()
        logger.info("total loss iter: %d, loss: %.3f", iter_no, total_loss_per_example.sum().item())

        with torch.no_grad():
            # update the weights of the perturbation network
            for module in perturbation.modules():
                # non-signed update
                if type(module) in (torch.nn.Sequential,):
                    continue
                elif type(module) in (SpatialFlowFields, ColorTransforms,
                                      ThinPlateSplines, AffineTransforms, ConvolutionalKernel):
                    for param in module.parameters():
                        param.sub_(param.grad * perturbation_step_size)
                # signed update
                elif type(module) in (Delta,):
                    for param in module.parameters():
                        param.sub_(torch.sign(param.grad) * perturbation_step_size)
                else:
                    raise RuntimeError(f'{module} update function not defined')

            # clip the parameters to be withing the linf budget of the image
            for module in perturbation.modules():

                if type(module) in (Delta,):
                    # clip the parameters to be within pixel intensities [0, 1]
                    img_bound_clip_params = torch.clamp(module.xform_params, min=0.0, max=1.0)
                    module.xform_params.copy_(img_bound_clip_params)
                    # based on linf budget clip the parameters
                    for ch in range(3):
                        linf_clip_params = torch.clamp(
                            module.xform_params[:, ch, :, :] - module.identity_params[:, ch, :, :],
                            min=-linf_budget[ch], max=linf_budget[ch]) + module.identity_params[:, ch, :, :]
                        module.xform_params[:, ch, :, :].copy_(linf_clip_params)

                if type(module) in (ColorTransforms,):
                    # clip the parameters to be within pixel intensities [0, 1]
                    img_bound_clip_params = torch.clamp(module.xform_params, min=0.0, max=1.0)
                    module.xform_params.copy_(img_bound_clip_params)
                    # based on linf budget clip the parameters
                    for ch in range(3):
                        linf_clip_params = torch.clamp(
                            module.xform_params[:, :, :, :, ch] - module.identity_params[:, :, :, :, ch],
                            min=-linf_budget[ch], max=linf_budget[ch]) + module.identity_params[:, :, :, :, ch]
                        module.xform_params[:, :, :, :, ch].copy_(linf_clip_params)

                if type(module) in (SpatialFlowFields,):
                    # standard linf measure is not applicable here
                    # we want to clip perturbation to be within a fixed shift for, x and y co-ordinates
                    h = module.xform_params.shape[1]
                    w = module.xform_params.shape[2]
                    h_per_pixel_shift = (2.0 / h) * pixel_shift_budget  # grid generated is from [-1, 1]
                    w_per_pixel_shift = (2.0 / w) * pixel_shift_budget
                    # clip parameters to be withing the grid range of [-1, 1]
                    x_y_bound_clip_params = torch.clamp(module.xform_params, min=-1.0, max=1.0)
                    module.xform_params.copy_(x_y_bound_clip_params)
                    # based on co-ordinate shift clip the parameters
                    x_shift_clip_params = torch.unsqueeze(torch.clamp(
                        module.xform_params[:, :, :, 0] - module.identity_params[:, :, :, 0], min=-w_per_pixel_shift,
                        max=w_per_pixel_shift), dim=3)
                    y_shift_clip_params = torch.unsqueeze(torch.clamp(
                        module.xform_params[:, :, :, 1] - module.identity_params[:, :, :, 1], min=-h_per_pixel_shift,
                        max=h_per_pixel_shift), dim=3)
                    shift_clip_params = module.identity_params + torch.cat([x_shift_clip_params, y_shift_clip_params],
                                                                           dim=3)
                    module.xform_params.copy_(shift_clip_params)

                if type(module) in (ThinPlateSplines,):
                    # clip the params to be within the valid bounds of [-1, 1]
                    _, _, h, w = module.batch_shape
                    h_per_pixel_shift = (2.0 / h) * pixel_shift_budget
                    w_per_pixel_shift = (2.0 / w) * pixel_shift_budget
                    x_y_bound_clip_params = torch.clamp(module.xform_params, min=-1.0, max=1.0)
                    module.xform_params.copy_(x_y_bound_clip_params)
                    x_shift_clip_params = torch.clamp(module.xform_params[:, :, 0] - module.identity_params[:, :, 0],
                                                      min=-w_per_pixel_shift, max=w_per_pixel_shift).unsqueeze(dim=2)
                    y_shift_clip_params = torch.clamp(module.xform_params[:, :, 1] - module.identity_params[:, :, 1],
                                                      min=-h_per_pixel_shift, max=h_per_pixel_shift).unsqueeze(dim=2)
                    shift_clip_params = module.identity_params + torch.cat([x_shift_clip_params, y_shift_clip_params],
                                                                           dim=2)
                    module.xform_params.copy_(shift_clip_params)

                if type(module) in (AffineTransforms,):
                    angle = torch.atan2(module.xform_params[:, 1, 0], module.xform_params[:, 1, 1])
                    clip_angle = torch.clamp(angle, min=-np.pi/2, max=np.pi/2).unsqueeze(dim=1)
                    clip_sx = torch.clamp(module.xform_params[:, 0, 0] / torch.cos(angle), min=0.8, max=1.2).unsqueeze(dim=1)
                    clip_sy = torch.clamp(module.xform_params[:, 1, 0] / torch.sin(angle), min=0.8, max=1.2).unsqueeze(dim=1)
                    clip_tx = torch.clamp(module.xform_params[:, 0, -1],
                                          min=-pixel_shift_budget, max=pixel_shift_budget).unsqueeze(dim=1)
                    clip_ty = torch.clamp(module.xform_params[:, 1, -1],
                                          min=-pixel_shift_budget, max=pixel_shift_budget).unsqueeze(dim=1)
                    module.xform_params[:, 0, :2].copy_(
                        torch.cat([clip_sx, -1.0 * clip_sy], dim=1) * torch.cat([torch.cos(clip_angle), torch.sin(clip_angle)],
                                                                                dim=1))
                    module.xform_params[:, 1, :2].copy_(
                        torch.cat([clip_sx, clip_sy], dim=1) * torch.cat([torch.sin(clip_angle), torch.cos(clip_angle)], dim=1))
                    module.xform_params[:, :, -1].copy_(torch.cat([clip_tx, clip_ty], dim=1))

                if type(module) in (ConvolutionalKernel,):
                    weights = module.xform_params
                    batch_size, output_channels, input_channels, kx, ky = weights.shape
                    # ?? make sure L1 of the kernel adds to 1.0 (preserve the intensity of image) ??
                    weights = weights.view(batch_size, output_channels, -1) / torch.norm(weights.view(batch_size, output_channels, -1), p=1, dim=2, keepdim=True)
                    module.xform_params.copy_(weights.view(batch_size, output_channels, input_channels, kx, ky))

        # bookkeeping to track the best perturbed examples
        if keep_best:
            for i, el in enumerate(total_loss_per_example):
                cur_best_loss = best_loss_per_example[i]
                if cur_best_loss > float(el):
                    best_loss_per_example[i] = float(el)
                    best_perturbed_examples[i].copy_(perturbed_examples[i].data)

        # If validation is provided do early stopping
        if validator is not None:
            validation = validator(classifier, examples, best_perturbed_examples, labels)
            logger.info("iter: %d attack-success-rate: %.3f", iter_no + 1, validation)

    # clean up the accumulated gradients
    classifier.zero_grad()
    perturbation.zero_grad()
    if keep_best:
        return best_perturbed_examples
    return perturbed_examples.detach()
