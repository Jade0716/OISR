import torch

def compute_distance(P, p, n):
    n = n / torch.norm(n)  # 归一化法向量
    return torch.norm(torch.cross(P - p, n.expand_as(P)), dim=1)

lambda_C = torch.nn.Parameter(torch.tensor(1.0, requires_grad=True, device='cuda'))

def energy_function(params, instance_data):
    total_loss = 0.0
    for i, param_tuple in enumerate(params):
        # 兼容 (sem_label, idx, x) 或直接 x
        if isinstance(param_tuple, tuple) and len(param_tuple) == 3:
            sem_label = param_tuple[0]
            x = param_tuple[2]
        else:
            sem_label = instance_data[i]["sem_label"]
            x = param_tuple
        inst = instance_data[i]

        if sem_label in {"hinge_lid", "hinge_door"}:
            p = x[:3]
            n = x[3:]
            n = n / torch.norm(n)

            P_inst = inst["points"]
            F_inst = inst["flow"]
            P_number = P_inst.shape[0]
            F_norm = torch.norm(F_inst, dim=1)
            distances = compute_distance(P_inst, p, n)

            D_ratio = distances.unsqueeze(1) / (distances.unsqueeze(0) + 1e-6)
            F_ratio = F_norm.unsqueeze(1) / (F_norm.unsqueeze(0) + 1e-6)
            EC = torch.sum(torch.abs(D_ratio - F_ratio))

            top_k = max(1, P_number // 2)
            threshold = torch.topk(F_norm, top_k, largest=True)[0][-1]
            mask = F_norm >= threshold

            P_inst = P_inst[mask]
            F_inst = F_inst[mask]
            F_norm = F_norm[mask]
            P_number_big = P_inst.shape[0]

            if P_number_big == 0:
                print("not cal")
                continue

            W = F_norm / torch.sum(F_norm)
            F_dot_n = torch.sum(F_inst * n, dim=1)
            EV = torch.sum(W * torch.abs(F_dot_n / torch.clamp(F_norm, min=1e-3)))

            loss = EV / P_number_big + EC / (P_number * P_number)

        elif sem_label in {"slider_drawer", "slider_lid"}:
            n = x[:3]
            n = n / torch.norm(n)

            P_inst = inst["points"]
            F_inst = inst["flow"]
            P_number = P_inst.shape[0]
            F_norm = torch.norm(F_inst, dim=1)
            top_k = max(1, P_number // 5)
            threshold = torch.topk(F_norm, top_k, largest=True)[0][-1]
            mask = F_norm >= threshold

            P_inst = P_inst[mask]
            F_inst = F_inst[mask]
            F_norm = F_norm[mask]
            P_number_big = P_inst.shape[0]

            F_dot_n = torch.sum(F_inst * n, dim=1)
            F_norm = torch.norm(F_inst, dim=1)
            n_norm = torch.norm(n)
            residuals = F_dot_n - n_norm * F_norm
            loss = torch.sum(torch.abs(residuals)) / P_number_big

        else:
            print("not cal")
            loss = torch.tensor(0.0)
        total_loss = total_loss + loss
    return total_loss