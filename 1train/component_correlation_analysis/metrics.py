import torch
import pdb

def cosine_similarity(matrix1, matrix2):
    n, m = matrix1.size(0), matrix2.size(0)
    res_matrix = torch.zeros((n, m), device=matrix1.device)
    
    for i in range(0, n):
        for j in range(0, m):
            res_matrix[i][j] = torch.nn.functional.cosine_similarity(matrix1[i], matrix2[j], dim=0)
    
    similarity = res_matrix.mean()
    return similarity

def euclidean_distance(matrix1, matrix2):
    n, m = matrix1.size(0), matrix2.size(0)
    distance_matrix = torch.zeros((n, m), device=matrix1.device)
    
    for i in range(0, n):
        for j in range(0, m):
            distance_matrix[i][j] = torch.sqrt(torch.sum((matrix1[i] - matrix2[j]) ** 2, dim=0))
    
    similarity = - distance_matrix.mean()
    return similarity

def dtw_distance(matrix1, matrix2):
    n, m = matrix1.size(0), matrix2.size(0)
    dtw_matrix = torch.zeros((n + 1, m + 1)).fill_(float('inf'))
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = torch.norm(matrix1[i-1] - matrix2[j-1], p=2)
            last_min = torch.min(torch.stack([dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1]]))
            dtw_matrix[i, j] = cost + last_min

    return dtw_matrix[n, m]

def pearson_correlation(matrix1, matrix2):
    n, m = matrix1.size(0), matrix2.size(0)
    res_matrix = torch.zeros((n, m), device=matrix1.device)
    
    for i in range(0, n):
        for j in range(0, m):
            mean1 = matrix1[i].mean(dim=-1, keepdim=True)
            mean2 = matrix2[j].mean(dim=-1, keepdim=True)
            
            cov = ((matrix1[i] - mean1) * (matrix2[j] - mean2)).mean(dim=-1)
            std1 = matrix1[i].std(dim=-1)
            std2 = matrix2[j].std(dim=-1)

            res_matrix[i][j] = cov / (std1 * std2)
    correlation = res_matrix.mean()
    return correlation

def mutual_information(matrix1, matrix2, bins=30):
    hist_2d = torch.histc(matrix1, bins) * torch.histc(matrix2, bins)
    pxy = hist_2d / hist_2d.sum()
    px = torch.histc(matrix1, bins) / matrix1.numel()
    py = torch.histc(matrix2, bins) / matrix2.numel()
    px_py = px.unsqueeze(1) * py.unsqueeze(0)
    mi = (pxy * torch.log(pxy / px_py + 1e-10)).sum()
    return mi


# def cosine_similarity(matrix1, matrix2):
#     similarity = torch.nn.functional.cosine_similarity(matrix1, matrix2, dim=-1)
#     return similarity

# def euclidean_distance(matrix1, matrix2):
#     distance = torch.sqrt(torch.sum((matrix1 - matrix2) ** 2, dim=-1))
#     return distance

# def dtw_distance(matrix1, matrix2):
#     n, m = matrix1.size(0), matrix2.size(0)
#     dtw_matrix = torch.zeros((n + 1, m + 1)).fill_(float('inf'))
#     dtw_matrix[0, 0] = 0

#     for i in range(1, n + 1):
#         for j in range(1, m + 1):
#             cost = torch.norm(matrix1[i-1] - matrix2[j-1], p=2)
#             last_min = torch.min(torch.stack([dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1]]))
#             dtw_matrix[i, j] = cost + last_min

#     return dtw_matrix[n, m]

# def pearson_correlation(matrix1, matrix2):
#     mean1 = matrix1.mean(dim=-1, keepdim=True)
#     mean2 = matrix2.mean(dim=-1, keepdim=True)
    
#     cov = ((matrix1 - mean1) * (matrix2 - mean2)).mean(dim=-1)
#     std1 = matrix1.std(dim=-1)
#     std2 = matrix2.std(dim=-1)
    
#     correlation = cov / (std1 * std2)
#     return correlation

# def mutual_information(matrix1, matrix2, bins=30):
#     hist_2d = torch.histc(matrix1, bins) * torch.histc(matrix2, bins)
#     pxy = hist_2d / hist_2d.sum()
#     px = torch.histc(matrix1, bins) / matrix1.numel()
#     py = torch.histc(matrix2, bins) / matrix2.numel()
#     px_py = px.unsqueeze(1) * py.unsqueeze(0)
#     mi = (pxy * torch.log(pxy / px_py + 1e-10)).sum()
#     return mi
