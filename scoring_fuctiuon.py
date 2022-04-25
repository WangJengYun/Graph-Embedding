
def l1_distance(a,b):
    assert len(a.shape) == len(b.shape), "Please check the dimension of a and b, becasue a is not equal b"
    return (a-b).norm(p = 1, dim = -1)**1

def l2_distance(a,b):
    assert len(a.shape) == len(b.shape), "Please check the dimension of a and b, becasue a is not equal b"
    return (a-b).norm(p = 2, dim = -1)**2

if __name__ == '__main__':
    import torch
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])
    (a-b).norm(p = 2,dim = -1)