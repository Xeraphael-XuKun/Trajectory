"""History优化诊断：只读同一计算图，不写入参数.grad，不改变训练损失。"""
import math
import torch


def number(value):
    value = float(value)
    return value if math.isfinite(value) else None


def tensor_norm(tensors):
    # double避免极小参数平方后在float32中下溢。
    total = sum(float(t.detach().double().square().sum()) for t in tensors if t is not None)
    return number(math.sqrt(total))


def relative(a, b):
    return a / b if a is not None and b is not None and b > 0 else None


class HistoryOptimizationMonitor:
    def __init__(self, adapter, optimizer):
        self.adapter = adapter
        self.names, self.params = zip(*[(n, p) for n, p in adapter.named_parameters() if p.requires_grad])
        self.families = {}
        for i, name in enumerate(self.names):
            self.families.setdefault(name.split('.')[0], []).append(i)
        self.groups = {id(p): group for group in optimizer.param_groups for p in group['params']}

    def norms(self, gradients):
        return {family: tensor_norm([gradients[i] for i in indices])
                for family, indices in self.families.items()}

    def loss_gradients(self, identity_loss, prediction_loss, prediction_weight, scale):
        # 采用与正式backward相同的GradScaler尺度，避免AMP半精度路径的额外下溢。
        # autograd.grad不累计到.grad；retain_graph供正式backward复用。
        def measure(loss):
            gradients = torch.autograd.grad(loss * scale, self.params,
                                            retain_graph=True, allow_unused=True)
            gradients = [None if g is None else g.detach().float() / scale for g in gradients]
            return self.norms(gradients)
        identity = measure(identity_loss)
        prediction = measure(prediction_loss)
        return {'identity_grad_norm': identity,
                'prediction_raw_grad_norm': prediction,
                'prediction_weighted_grad_norm': {
                    k: None if v is None else v * prediction_weight for k, v in prediction.items()},
                'grad_scaler_scale': scale}

    @torch.no_grad()
    def parameter_summary(self):
        result = {}
        for family, indices in self.families.items():
            ps = [self.params[i] for i in indices]
            count = sum(p.numel() for p in ps)
            norm = tensor_norm(ps)
            values = {'numel': count, 'norm': norm,
                      'rms': None if norm is None else norm / math.sqrt(count)}
            if family in ('gain', 'cls_gain'):
                effective = [self.adapter.alpha * (p.tanh() if self.adapter.gain_mode == 'tanh' else p)
                             for p in ps]
                values['effective_rms'] = tensor_norm(effective) / math.sqrt(count)
            values['weight_decay'] = sorted(set(self.groups[id(p)]['weight_decay'] for p in ps))
            values['learning_rates'] = sorted(set(self.groups[id(p)]['lr'] for p in ps))
            result[family] = values
        return result

    @torch.no_grad()
    def before_clip(self):
        return self.norms([p.grad for p in self.params])

    @torch.no_grad()
    def after_clip(self):
        gradients = [p.grad for p in self.params]
        # Adam仅处理grad非None的参数；衰减项在全局clip后由优化器加入。
        decay = [None if p.grad is None else p.detach() * self.groups[id(p)]['weight_decay']
                 for p in self.params]
        summed = [None if g is None else g.detach() + w for g, w in zip(gradients, decay)]
        data_norm, decay_norm = self.norms(gradients), self.norms(decay)
        return {'data_grad_postclip_norm': data_norm, 'decay_term_norm': decay_norm,
                'adam_input_grad_norm': self.norms(summed),
                'decay_to_data_norm_ratio': {k: relative(decay_norm[k], data_norm[k]) for k in data_norm}}


def diagnostic_stop_epoch(cfg):
    """仅限制训练循环，不触碰scheduler使用的MAX_EPOCHS。"""
    stop = cfg.HISTORY.STOP_AFTER_EPOCH if cfg.HISTORY.ENABLED else 0
    if stop < 0 or stop > cfg.SOLVER.MAX_EPOCHS:
        raise ValueError('HISTORY.STOP_AFTER_EPOCH must be 0 or <= SOLVER.MAX_EPOCHS')
    return stop
