function out = mysub2ind3D(sizeA, i1, i2, i3)
        
out = i1 + (i2-1)*sizeA(1,1) + (i3-1)*sizeA(1,1)*sizeA(1,2);