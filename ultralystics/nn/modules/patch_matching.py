import torch
from time import time


def patch_matching(img):#, gt

    ps = 16
    h, w = img.shape[2], img.shape[3]
    maxr = h - ps + 1      #
    maxc = w - ps + 1      #
    maxrc = maxr * maxc #
    
    step = 8
    Win = 32
    nlsp = 8 # number of nonlocal patches
    r = torch.arange(0,maxr,step)
    r = torch.cat([r, torch.arange(r[-1]+1, maxr)])#.to('cuda')
    c = torch.arange(0,maxc,step)
    c = torch.cat([c, torch.arange(c[-1]+1, maxc)])#.to('cuda')
    
    lenr = len(r)       #
    lenc = len(c)       #
    lenrc = lenr*lenc   #
    ps2 = ps**2         #
    
    # X = torch.zeros(ps2*3, maxrc).to('cuda') # 모든 패치를 담고 있는 변수 (개별 패치, 모든 패치 수)
    X = torch.zeros(ps2, maxrc).to('cuda') # single patch
    k = 0
    # 패치를 한번에 저장
    for channel in range(img.shape[1]):
        for i in range(ps):
            for j in range(ps):
                blk = img[:,channel,i:h+i-ps+1,j:w+j-ps+1]
                X[k,:] = blk[:].reshape(1,-1)
                k += 1
    # index of each patch in image
    index = torch.arange(0,maxrc).reshape(maxr, maxc)#.to('cuda') # 0~31328
    
    # record the indexs of patches similar to the seed patch
    blk_arr = torch.zeros(nlsp, lenrc).to('cuda')

    # # non-local patch groups
    # nDCnlX = torch.zeros(ps2, lenrc*nlsp)
    # k = 0
    temp = torch.zeros(lenrc, ps*ps*img.shape[1], nlsp).to('cuda')
    #lenrc 3721
    for i in range(lenr):
        for j in range(lenc):
            row = r[i]
            col = c[j]
            
            off = col + row * maxc # maxr = 305 / row 0 3 6 9 ... 303 304
            off1 = j + i*lenc

            # the range indexes of the window for searching the similar patches
            rmin = max(row - Win, 0)
            rmax = min(row + Win, maxr)
            cmin = max(col - Win, 0)
            cmax = min(col + Win, maxc)

            idx = index[rmin:rmax, cmin:cmax].flatten().cuda()
            # idx = idx[:]
            if off not in idx:
                print("off not in idx")
            neighbor = X[:,idx] # 256 16 16,19,22,25,28,31,32
            seed = X[:,off].unsqueeze(1)     # 256
            dis = torch.sum((neighbor - seed) ** 2, dim=0)
            
            # seed to zero
            indices = torch.where(idx == off)
            dis += 1
            dis[indices] = 0

            _, ind = torch.sort(dis)
            indc = idx[ind[:nlsp]]
            blk_arr[:,off1] = indc  # 어떤 인덱스끼리 모였는지 표기
            temp[off1] = X[:, indc]  # 거리적으로 가까운 패치들의 모임 
            # k += 1
    return blk_arr, temp # 16 3721 / 3721 768 16 [패치수, 각 패치 인덱스], [각 패치 인덱스, 이미지, 패치수]

def aggregation(blk_arr, temp, img_shape, ps):
    aggregated_img = torch.zeros(img_shape).to('cuda')
    count = torch.zeros(img_shape).to('cuda')
    
    for i in range(blk_arr.shape[1]):
        for j in range(blk_arr.shape[0]):
            pos = blk_arr[j,i]
            row = int(pos // (img_shape[3]- ps +1))#
            col = int(pos % (img_shape[3]- ps +1 ))#
            patch = temp.permute(0,2,1)[i, j].reshape(1,ps,ps)
            aggregated_img[0,:,row:row+ps, col:col+ps] += patch
            count[0,:,row:row+ps, col:col+ps] += 1
        
            # save_image(patch, f'/media/vimlab/5A5EDA815EDA557B/restormer/patch_matching/{i}_{j}.png')
        #   plt.plot(), plt.imshow(patch.permute(1,2,0).to('cpu'))
        #   plt.xticks([]), plt.yticks([]), plt.show()
    aggregated_img /= count
    return aggregated_img

def patch_matching_using_blk_arr(img, blk_arr,ps = 16):
    
    h, w = img.shape[2], img.shape[3]
    maxr = h - ps + 1      #
    maxc = w - ps + 1      #
    maxrc = maxr * maxc #

    step = 6
    nlsp = 8 # number of nonlocal patches
    r = torch.arange(0,maxr,step)
    r = torch.cat([r, torch.arange(r[-1]+1, maxr)]).to('cuda')
    c = torch.arange(0,maxc,step)
    c = torch.cat([c, torch.arange(c[-1]+1, maxc)]).to('cuda')

    lenr = len(r)       #
    lenc = len(c)       #
    lenrc = lenr*lenc   #
    ps2 = ps**2         #

    X = torch.zeros(ps2, maxrc).to('cuda') # 모든 패치를 담고 있는 변수 (개별 패치, 모든 패치 수)
    #X_gt = torch.zeros_like(X)#!!!!!!!!!!!
    k = 0
    temp = torch.zeros(lenrc, ps*ps*img.shape[1], nlsp).to('cuda')
    # 패치를 한번에 저장
    for channel in range(img.shape[1]):
        for i in range(ps):
            for j in range(ps):
                blk = img[:,channel,i:h+i-ps+1,j:w+j-ps+1]
                X[k,:] = blk[:].reshape(1,-1)
                #blk1 = gt[:,channel,i:h+i-ps+1,j:w+j-ps+1]
                #X_gt[k,:] = blk1[:].reshape(1,-1)#!!!!!!!!!!!
                k += 1
    for off1 in range(blk_arr.shape[1]):
        indc = blk_arr[:,off1].long()   # 어떤 인덱스끼리 모였는지 표기
        temp[off1] = X[:, indc]  # 거리적으로 가까운 패치들의 모임
    return temp



def reconstruct_patches(blk_arr, temp, img_shape, ps=16):
    
    h, w = img_shape[2], img_shape[3]
    maxr = h - ps + 1      #
    maxc = w - ps + 1      #
    maxrc = maxr * maxc #

    step = 6
    r = torch.arange(0,maxr,step)
    r = torch.cat([r, torch.arange(r[-1]+1, maxr)]).to('cuda')
    c = torch.arange(0,maxc,step)
    c = torch.cat([c, torch.arange(c[-1]+1, maxc)]).to('cuda')

    lenr = len(r)       #
    lenc = len(c)       #
    lenrc = lenr*lenc   #
    ps2 = ps**2         #

    X = torch.zeros(ps2, maxrc).to('cuda') # 모든 패치를 담고 있는 변수 (개별 패치, 모든 패치 수)
    # 256 1275
    K = torch.zeros_like(X)
    #one_1 = torch.where(temp>0,1,temp)
    for off1 in range(blk_arr.shape[1]):
        indc = blk_arr[:,off1].long()   # 어떤 인덱스끼리 모였는지 표기
        X[:, indc] += temp[off1] 
        K[:, indc] += 1#one_1[off1]
    # X /= K    
    img = torch.zeros(img_shape).to('cuda')
    ones = torch.zeros_like(img)
    k = 0
    # 패치를 한번에 저장
    for i in range(ps):
        for j in range(ps):
            img[:,:,i:h+i-ps+1,j:w+j-ps+1] += X[k,:].reshape(1,1,h-ps+1,w-ps+1)
            ones[:,:,i:h+i-ps+1,j:w+j-ps+1] += K[k,:].reshape(1,1,h-ps+1,w-ps+1)
            k += 1
            # plt.plot(), plt.imshow(img.squeeze(0).permute(1,2,0).to('cpu'), cmap='gray')
            # plt.xticks([]), plt.yticks([]), plt.show()
    img /= ones
    return img