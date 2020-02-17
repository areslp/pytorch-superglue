import torch
import pdb

def sinkhorn_stabilized(a, b, M, reg, numItermax=100, tau=1e3, stopThr=1e-3, print_period=20):

    n_hists = 0

    # init data
    dim_a = len(a)
    dim_b = len(b)

    # we assume that no distances are null except those of the diagonal of
    # distances
    alpha, beta = torch.zeros(dim_a), torch.zeros(dim_b)

    u, v = torch.ones(dim_a) / dim_a, torch.ones(dim_b) / dim_b

    def get_K(alpha, beta):
        """log space computation"""
        return torch.exp(-(M - alpha.reshape((dim_a, 1)) - beta.reshape((1, dim_b))) / reg)

    def get_Gamma(alpha, beta, u, v):
        """log space gamma computation"""
        return torch.exp(-(M - alpha.reshape((dim_a, 1)) - beta.reshape((1, dim_b))) / reg + torch.log(1e-6 +
            u.reshape((dim_a, 1))) + torch.log(1e-6 + v.reshape((1, dim_b))))

    # print(torch.min(K))

    K = get_K(alpha, beta)
    transp = K
    loop = 1
    cpt = 0
    err = 1

    while loop:

        uprev = u
        vprev = v

        # sinkhorn update

        v = b / (torch.tensordot(K.T, u, dims=([0], [0])) + 1e-6)
        u = a / (torch.tensordot(K, v, dims=([1], [0])) + 1e-6)
        # remove numerical problems and store them in K
        if torch.abs(u).max() > tau or torch.abs(v).max() > tau:
            if n_hists:
                alpha, beta = alpha + reg * \
                    torch.max(torch.log(u), 1), beta + reg * torch.max(torch.log(1e-6 + v))
            else:
                alpha, beta = alpha + reg * torch.log(u), beta + reg * torch.log(1e-6 + v)
                if n_hists:
                    u, v = torch.ones((dim_a, n_hists)) / dim_a, torch.ones((dim_b, n_hists)) / dim_b
                else:
                    u, v = torch.ones(dim_a) / dim_a, torch.ones(dim_b) / dim_b
            K = get_K(alpha, beta)

        if cpt % print_period == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            transp = get_Gamma(alpha, beta, u, v)
            err = torch.norm((torch.sum(transp, dim=0) - b))


        if err <= stopThr:
            loop = False

        if cpt >= numItermax:
            loop = False

        if torch.any(torch.isnan(u)) or torch.any(torch.isnan(v)):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            print('Warning: numerical errors at iteration', cpt)
            u = uprev
            v = vprev
            break

        cpt = cpt + 1

    return get_Gamma(alpha, beta, u, v)

