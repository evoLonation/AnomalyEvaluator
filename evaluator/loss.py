from jaxtyping import Bool, Float, Int64, jaxtyped
import torch

@jaxtyped(typechecker=None)
def focal_loss(
    logit: Float[torch.Tensor, "N CLS=2 H W"],
    target: Bool[torch.Tensor, "N H W"],
    alpha: float | None = None,
    gamma: int = 2,
    smooth: float = 1e-5,
    size_average: bool = True,
):
    if alpha is not None:
        if alpha < 0 or alpha > 1.0:
            raise ValueError("alpha value should be in [0,1]")
    if smooth is not None:
        if smooth < 0 or smooth > 1.0:
            raise ValueError("smooth value should be in [0,1]")
    num_class = logit.shape[1]

    symbolic_decl: Float[torch.Tensor, "N CLS H W"] = logit
    logit = logit.view(*logit.shape[0:2], -1)
    logit = logit.permute(0, 2, 1).contiguous()
    logit = logit.view(-1, logit.shape[-1])
    logit_hint: Float[torch.Tensor, "N*H*W CLS"] = logit

    target = target.flatten().long()
    target_hint: Int64[torch.Tensor, "N*H*W"] = target

    alpha_tensor = torch.ones(num_class, device=logit.device)
    if isinstance(alpha, float):
        alpha_tensor = alpha_tensor * (1 - alpha)
        alpha_tensor[1] = alpha
    alpha_tensor = alpha_tensor[target]
    alpha_tensor: Float[torch.Tensor, "N*H*W"] = alpha_tensor

    one_hot_key: Bool[torch.Tensor, "N*H*W CLS"] = torch.zeros(
        target.shape[0], num_class, device=logit.device, dtype=torch.bool
    )
    one_hot_key = one_hot_key.scatter_(1, target.unsqueeze(1), 1)

    if smooth:
        one_hot_key = torch.clamp(one_hot_key, smooth, 1.0 - smooth)
    pt: Float[torch.Tensor, "N*H*W"] = (one_hot_key * logit).sum(1) + smooth
    logpt = pt.log()

    gamma = gamma

    loss = -1 * alpha_tensor * torch.pow((1 - pt), gamma) * logpt

    if size_average:
        loss = loss.mean()
    # print("focal_loss:", loss.item())
    return loss


def binary_dice_loss(
    input: Float[torch.Tensor, "N H W"], targets: Bool[torch.Tensor, "N H W"]
) -> Float[torch.Tensor, ""]:
    # 获取每个批次的大小 N
    N = targets.shape[0]
    # 平滑变量
    smooth = 1
    # 将宽高 reshape 到同一纬度
    input_flat = input.view(N, -1)
    targets_flat = targets.view(N, -1)

    intersection = input_flat * targets_flat
    N_dice_eff = (2 * intersection.sum(1) + smooth) / (
        input_flat.sum(1) + targets_flat.sum(1) + smooth
    )
    # 计算一个批次中平均每张图的损失
    loss = 1 - N_dice_eff.sum() / N
    # print("dice_loss:", loss.item())
    return loss