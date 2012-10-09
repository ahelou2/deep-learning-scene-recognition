#!/usr/bin/python env
import sys
if __name__ == '__main__':
    fin = open(sys.argv[1], 'r')
    fout = open(sys.argv[2], 'w')
    for line in fin:
        st = line.split(' ')
        for t in range(0, len(st)):
            if len(st[t]) > 0:
                out = str(int(float(st[t])))
                break
        j = 0
        for i in range(t+1, len(st)):
            if len(st[i]) > 0:
                j = j + 1
                out = out + ' '+str(j)+':'+str(float(st[i]))
        fout.write(out+'\n')
    fin.close()
    fout.close()
